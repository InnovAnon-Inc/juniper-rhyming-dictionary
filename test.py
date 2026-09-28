#! /usr/bin/env python3

import os
import time
import re
import wave
import numpy as np
import json
import urllib.request
from scipy.io import wavfile
import pyttsx3
import nltk
from nltk.corpus import cmudict, wordnet, words
import pronouncing
from collections import defaultdict
import random
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory

# Ensure NLTK datasets are downloaded
nltk.download('wordnet', quiet=False)
nltk.download('words', quiet=False)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def is_palindrome(s):
    return s == s[::-1]

import random
import re
from collections import defaultdict

# Dynamic word usage frequency tracker
WORD_USAGE_COUNTS = defaultdict(int)

CMU_DICT = cmudict.dict()

VALID_SHORT_WORDS = {"a", "i", "in", "on", "no", "is", "it", "or", "to", "at", "am", "an", "so", "do", "go", "me", "my", "we", "he", "be", "us", "up", "if"}

def is_valid_real_word(w):
    w_clean = w.lower()
    if not w_clean.isalpha():
        return False
    if len(w_clean) < 3 and w_clean not in VALID_SHORT_WORDS:
        return False
    if w_clean != w and w.isupper():
        return False
    return w_clean in CMU_DICT

VOCAB = [
    w.lower() for w in set(wordnet.all_lemma_names()) 
    if is_valid_real_word(w)
]
VOCAB_SET = set(VOCAB)

# Trie implementation for efficient character-level cross-boundary lookup
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

    def get_valid_prefixes(self, prefix):
        """Finds all complete words that start with prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        results = []
        def _dfs(curr_node, path):
            if curr_node.is_word:
                results.append(path)
            for ch, child in curr_node.children.items():
                _dfs(child, path + ch)

        _dfs(node, prefix)
        return results

TRIE = Trie()
for w in VOCAB:
    TRIE.insert(w)

# --- 3. ADVANCED ASYMMETRIC PALINDROME ENGINE ---
def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def get_palindrome_fudge(phrase):
    clean = re.sub(r'[^a-z]', '', phrase.lower())
    return edit_distance(clean, clean[::-1])

def is_phrase_valid(phrase):
    words = phrase.lower().split()
    return all(w in VOCAB_SET for w in words)

def letter_overlap_score(word1, word2):
    s1 = set(word1.lower())
    s2 = set(word2.lower())
    return len(s1.intersection(s2))


def get_word_weight(word, decay_rate=0.15):
    """
    Calculates a smooth, gradual weight reduction for used words.
    Prevents abrupt drop-offs while prioritizing lesser-used vocabulary.
    """
    print('get_word_weight')
    count = WORD_USAGE_COUNTS[word.lower()]
    return 1.0 / (1.0 + decay_rate * (count ** 0.75))

def record_phrase_usage(phrase):
    """Increments frequency counters for every word in a phrase."""
    print('record_phrase_usage')
    for word in phrase.lower().split():
        WORD_USAGE_COUNTS[word] += 1

def get_fuzzy_matches(prefix, max_dist=2, max_candidates=30):
    """
    Finds vocabulary words whose starting prefixes match 'prefix'
    within a Levenshtein edit distance allowance.
    """
    matches = []
    prefix_len = len(prefix)

    # Exact prefix lookup via Trie
    exact = TRIE.get_valid_prefixes(prefix)
    if exact:
        return exact[:max_candidates]

    # Fuzzy prefix lookup across vocabulary
    for w in VOCAB:
        if len(w) >= 2:
            sub = w[:prefix_len]
            if edit_distance(sub, prefix) <= max_dist:
                matches.append(w)
                if len(matches) >= max_candidates:
                    break
    return matches

import re

def extract_structural_candidates(target_word, max_candidates=30):
    """
    Dynamically extracts vocabulary candidates based on subsegment matches
    of the reversed target word, avoiding single-character candidate traps.
    """
    target_rev = target_word.lower()[::-1]
    candidates = set()
    t_len = len(target_rev)

    # 1. Look for dynamic multi-character prefix/suffix matches (length >= 2)
    for slice_len in range(t_len, 1, -1):
        sub_prefix = target_rev[:slice_len]
        matches = TRIE.get_valid_prefixes(sub_prefix)
        for m in matches:
            if m != target_word.lower() and len(m) >= 2:
                candidates.add(m)
        if len(candidates) >= max_candidates:
            break

    # 2. Subsegment matching for longer target words
    if len(candidates) < max_candidates and t_len >= 4:
        # Check interior n-grams to find natural structural bridges
        for i in range(t_len - 2):
            ngram = target_rev[i:i+3]
            for w in VOCAB:
                if len(w) >= 3 and ngram in w:
                    candidates.add(w)
                    if len(candidates) >= max_candidates:
                        break

    return list(candidates)


def find_dynamic_pivots(left_str, right_str, max_pivots=10):
    """
    Scans VOCAB dynamically for natural bridge words that minimize
    the edit distance between left_str and reversed(right_str).
    Eliminates the need for hardcoded 'connector' or 'mid' lists.
    """
    target_gap = left_str + right_str
    gap_rev = target_gap[::-1]

    pivots = []
    # Search vocabulary for words that match the structural imbalance
    for w in VOCAB:
        # Skip overly short words unless necessary
        if len(w) < 2 and len(left_str) > 2:
            continue

        combined = left_str + w + right_str
        clean_c = re.sub(r'[^a-z]', '', combined.lower())
        fudge = edit_distance(clean_c, clean_c[::-1])

        # Keep words that keep the symmetry ratio low
        if fudge / len(clean_c) <= 0.8:
            pivots.append((w, fudge))

    # Sort pivots by lowest resulting symmetry fudge score
    pivots.sort(key=lambda x: x[1])
    return [p[0] for p in pivots[:max_pivots]]


def calculate_entropy_fudge(phrase):
    """
    Calculates symmetry fudge score using dynamic length and word-diversity
    penalties instead of hardcoded string lists.
    """
    words = phrase.lower().split()
    clean_phrase = re.sub(r'[^a-z]', '', phrase.lower())

    if not clean_phrase:
        return float('inf')

    # Base Levenshtein edit distance
    raw_fudge = edit_distance(clean_phrase, clean_phrase[::-1])

    # Dynamic Penalty 1: Single-character word penalty (avoids 'a', 'i' stacking)
    short_word_count = sum(1 for w in words if len(w) == 1)
    short_penalty = short_word_count * 2.0

    # Dynamic Penalty 2: Repetition penalty (prevents repeating the same bridge word)
    unique_words = set(words)
    repetition_penalty = (len(words) - len(unique_words)) * 1.5

    # Dynamic Bonus: Reward word length diversity (longer structural words)
    length_bonus = sum(0.3 for w in words if len(w) >= 4)

    return raw_fudge + short_penalty + repetition_penalty - length_bonus


def generate_target_palindrome(target_word, max_results=10, max_fudge_ratio=1.2):
    """
    Fully dynamic pseudo-palindrome generator. Free of hardcoded word lists.
    """
    target = target_word.lower()
    if target not in VOCAB_SET:
        return {}

    results = {}
    target_rev = target[::-1]

    # 1. Dynamically pull structural vocabulary candidates
    candidates = extract_structural_candidates(target)

    # 2. Build multi-word combinations using dynamic pivot discovery
    for w2 in candidates:
        # Dynamically discover mid-words from VOCAB that bridge target and w2
        dynamic_pivots = find_dynamic_pivots(target, w2)
        if not dynamic_pivots:
            dynamic_pivots = [""]  # Allow direct concats if no pivot is needed

        for pivot in dynamic_pivots:
            phrase_variants = [
                f"{target} {pivot} {w2}".strip(),
                f"{target} {w2[::-1]} {pivot} {w2}".strip(),
            ]

            # Remainder balancing for asymmetric word lengths
            if len(target) > len(w2):
                rem = target[len(w2):][::-1]
                if rem in VOCAB_SET:
                    phrase_variants.append(f"{target} {pivot} {rem} {w2}".strip())

            for phrase in phrase_variants:
                clean_p = re.sub(r'[^a-z]', '', phrase.lower())
                fudge_score = calculate_entropy_fudge(phrase)
                fudge_ratio = fudge_score / len(clean_p)

                if fudge_ratio <= max_fudge_ratio and is_phrase_valid(phrase):
                    if phrase not in results or fudge_score < results[phrase]:
                        results[phrase] = round(fudge_score, 2)

    # 3. Dynamic Structural Fallback (Uses real dictionary words as pivots if empty)
    if not results:
        fallback_pivots = find_dynamic_pivots(target, target_rev, max_pivots=5)
        for pivot in fallback_pivots:
            fallback = f"{target} {pivot} {target_rev}".strip()
            fudge_score = calculate_entropy_fudge(fallback)
            results[fallback] = round(fudge_score, 2)

    # Return top results sorted by lowest adjusted score
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))
    return dict(list(sorted_results.items())[:max_results])


def process_dictionary_word(word, used_palindromes): # TODO ensure that we don't return the word itself by itself, and also that our return value is not already in used_palindromes
    """
    Helper function to iterate over the dictionary.
    Generates candidates, records word usage, and returns the top palindrome block.
    """
    print('process_dictionary_word')
    candidates = generate_target_palindrome(word, max_results=10)
    if not candidates:
        print(f'warning: no candidates fo word {word}')
        return None

    # Pick top candidate with lowest fudge score safely
    best_phrase = next(iter(candidates.keys()))

    # Increment word counts gradually
    record_phrase_usage(best_phrase)

    return {
        "target": word,
        "palindrome": best_phrase,
        "fudge": candidates[best_phrase],
        "word_weights": {w: round(get_word_weight(w), 3) for w in best_phrase.split()}
    }

palindromes = []
for word in ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'lazy', 'dog']:
    result = process_dictionary_word(word, palindromes)
    if not result:
        print(f'{word}')
        continue
    palindrome = result['palindrome']
    palindromes.append(palindrome)
    print(f'{word}: {palindrome}')
