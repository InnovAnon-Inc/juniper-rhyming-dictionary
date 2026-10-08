#import os
#import re
#import time
#from collections import defaultdict
#import threading
#import nltk
#from nltk.corpus import cmudict, wordnet
#
## Ensure NLTK datasets are downloaded
#nltk.download('wordnet', quiet=False)
#nltk.download('words', quiet=False)
#
#nltk_lock = threading.Lock()
#
#def safe_wordnet_all_lemma_names():
#    with nltk_lock:
#        return list(wordnet.all_lemma_names())
#
#CMU_DICT = cmudict.dict()
#
#VALID_SHORT_WORDS = {
#    "a", "i", "in", "on", "no", "is", "it", "or", "to", "at", "am", "an",
#    "so", "do", "go", "me", "my", "we", "he", "be", "us", "up", "if"
#}
#
#def is_valid_real_word(w):
#    w_clean = w.lower()
#    if not w_clean.isalpha():
#        return False
#    if len(w_clean) < 3 and w_clean not in VALID_SHORT_WORDS:
#        return False
#    if w_clean != w and w.isupper():
#        return False
#    return w_clean in CMU_DICT
#
#VOCAB = [
#    w.lower() for w in set(safe_wordnet_all_lemma_names()) 
#    if is_valid_real_word(w)
#]
#VOCAB_SET = set(VOCAB)
#
## Dynamic word usage frequency tracker
#WORD_USAGE_COUNTS = defaultdict(int)
#
#class TrieNode:
#    def __init__(self):
#        self.children = {}
#        self.is_word = False
#
#class Trie:
#    def __init__(self):
#        self.root = TrieNode()
#
#    def insert(self, word):
#        node = self.root
#        for char in word:
#            if char not in node.children:
#                node.children[char] = TrieNode()
#            node = node.children[char]
#        node.is_word = True
#
#    def get_valid_prefixes(self, prefix):
#        """Finds all complete words that start with prefix."""
#        node = self.root
#        for char in prefix:
#            if char not in node.children:
#                return []
#            node = node.children[char]
#        
#        results = []
#        def _dfs(curr_node, path):
#            if curr_node.is_word:
#                results.append(path)
#            for ch, child in curr_node.children.items():
#                _dfs(child, path + ch)
#
#        _dfs(node, prefix)
#        return results
#
#TRIE = Trie()
#for w in VOCAB:
#    TRIE.insert(w)
#
#def edit_distance(s1, s2):
#    if len(s1) < len(s2):
#        return edit_distance(s2, s1)
#    if len(s2) == 0:
#        return len(s1)
#
#    previous_row = range(len(s2) + 1)
#    for i, c1 in enumerate(s1):
#        current_row = [i + 1]
#        for j, c2 in enumerate(s2):
#            insertions = previous_row[j + 1] + 1
#            deletions = current_row[j] + 1
#            substitutions = previous_row[j] + (c1 != c2)
#            current_row.append(min(insertions, deletions, substitutions))
#        previous_row = current_row
#    return previous_row[-1]
#
#def get_palindrome_fudge(phrase):
#    clean = re.sub(r'[^a-z]', '', phrase.lower())
#    return edit_distance(clean, clean[::-1])
#
#def is_phrase_valid(phrase):
#    words = phrase.lower().split()
#    return all(w in VOCAB_SET for w in words)
#
#def letter_overlap_score(word1, word2):
#    s1 = set(word1.lower())
#    s2 = set(word2.lower())
#    return len(s1.intersection(s2))
#
#def get_word_weight(word, decay_rate=0.15):
#    count = WORD_USAGE_COUNTS[word.lower()]
#    return 1.0 / (1.0 + decay_rate * (count ** 0.75))
#
#def record_phrase_usage(phrase):
#    for word in phrase.lower().split():
#        WORD_USAGE_COUNTS[word] += 1
#
#def get_fuzzy_matches(prefix, max_dist=2, max_candidates=30):
#    prefix_len = len(prefix)
#    exact = TRIE.get_valid_prefixes(prefix)
#    if exact:
#        return exact[:max_candidates]
#
#    matches = []
#    for w in VOCAB:
#        if len(w) >= 2:
#            sub = w[:prefix_len]
#            if edit_distance(sub, prefix) <= max_dist:
#                matches.append(w)
#                if len(matches) >= max_candidates:
#                    break
#    return matches
#
#def extract_structural_candidates(target_word, max_candidates=30):
#    target_rev = target_word.lower()[::-1]
#    candidates = set()
#    t_len = len(target_rev)
#
#    for slice_len in range(t_len, 1, -1):
#        sub_prefix = target_rev[:slice_len]
#        matches = TRIE.get_valid_prefixes(sub_prefix)
#        for m in matches:
#            if m != target_word.lower() and len(m) >= 2:
#                candidates.add(m)
#        if len(candidates) >= max_candidates:
#            break
#
#    if len(candidates) < max_candidates and t_len >= 4:
#        for i in range(t_len - 2):
#            ngram = target_rev[i:i+3]
#            for w in VOCAB:
#                if len(w) >= 3 and ngram in w:
#                    candidates.add(w)
#                    if len(candidates) >= max_candidates:
#                        break
#
#    return list(candidates)
#
#def find_dynamic_pivots(left_str, right_str, max_pivots=10):
#    pivots = []
#    for w in VOCAB:
#        if len(w) < 2 and len(left_str) > 2:
#            continue
#
#        combined = left_str + w + right_str
#        clean_c = re.sub(r'[^a-z]', '', combined.lower())
#        fudge = edit_distance(clean_c, clean_c[::-1])
#
#        if fudge / len(clean_c) <= 0.8:
#            pivots.append((w, fudge))
#
#    pivots.sort(key=lambda x: x[1])
#    return [p[0] for p in pivots[:max_pivots]]
#
#def calculate_entropy_fudge(phrase):
#    words = phrase.lower().split()
#    clean_phrase = re.sub(r'[^a-z]', '', phrase.lower())
#
#    if not clean_phrase:
#        return float('inf')
#
#    raw_fudge = edit_distance(clean_phrase, clean_phrase[::-1])
#    short_word_count = sum(1 for w in words if len(w) == 1)
#    short_penalty = short_word_count * 2.0
#    unique_words = set(words)
#    repetition_penalty = (len(words) - len(unique_words)) * 1.5
#    length_bonus = sum(0.3 for w in words if len(w) >= 4)
#
#    return raw_fudge + short_penalty + repetition_penalty - length_bonus
#
#def generate_target_palindrome(target_word, used_palindromes, max_results=10, max_fudge_ratio=1.2):
#    target = target_word.lower()
#    if target not in VOCAB_SET:
#        return {}
#
#    results = {}
#    target_rev = target[::-1]
#    candidates = extract_structural_candidates(target)
#
#    for w2 in candidates:
#        dynamic_pivots = find_dynamic_pivots(target, w2)
#        if not dynamic_pivots:
#            dynamic_pivots = [""]
#
#        for pivot in dynamic_pivots:
#            phrase_variants = [
#                f"{target} {pivot} {w2}".strip(),
#                f"{target} {w2[::-1]} {pivot} {w2}".strip(),
#            ]
#
#            if len(target) > len(w2):
#                rem = target[len(w2):][::-1]
#                if rem in VOCAB_SET:
#                    phrase_variants.append(f"{target} {pivot} {rem} {w2}".strip())
#
#            for phrase in phrase_variants:
#                clean_p = re.sub(r'[^a-z]', '', phrase.lower())
#                fudge_score = calculate_entropy_fudge(phrase)
#                fudge_ratio = fudge_score / len(clean_p)
#
#                if fudge_ratio <= max_fudge_ratio and is_phrase_valid(phrase):
#                    if phrase not in results or fudge_score < results[phrase]:
#                        results[phrase] = round(fudge_score, 2)
#
#    if not results:
#        fallback_pivots = find_dynamic_pivots(target, target_rev, max_pivots=5)
#        for pivot in fallback_pivots:
#            fallback = f"{target} {pivot} {target_rev}".strip()
#            fudge_score = calculate_entropy_fudge(fallback)
#            results[fallback] = round(fudge_score, 2)
#
#    filtered_results = [
#        result for result in results
#        if result != target_word and result not in used_palindromes
#    ]
#
#    sorted_results = dict(sorted({k: results[k] for k in filtered_results}.items(), key=lambda item: item[1]))
#    return dict(list(sorted_results.items())[:max_results])
#
#def process_dictionary_word(word, used_palindromes):
#    candidates = generate_target_palindrome(word, used_palindromes, max_results=10)
#    if not candidates:
#        return None
#
#    best_phrase = next(iter(candidates.keys()))
#    record_phrase_usage(best_phrase)
#
#    return {
#        "target": word,
#        "palindrome": best_phrase,
#        "fudge": candidates[best_phrase],
#        "word_weights": {w: round(get_word_weight(w), 3) for w in best_phrase.split()}
#    }
#
#PALINDROME_CACHE_DIR = os.path.join(os.path.dirname(__file__), "palindrome_cache")
#os.makedirs(PALINDROME_CACHE_DIR, exist_ok=True)
#
#def get_cached_data(word, cache_dir=PALINDROME_CACHE_DIR):
#    filepath = os.path.join(cache_dir, f"{word}.txt")
#    if os.path.exists(filepath):
#        with open(filepath, 'r', encoding='utf-8') as f:
#            content = f.read().strip()
#            return content if content else None
#    return None
#
#def write_cached_data(word, cache_dir, data):
#    filepath = os.path.join(cache_dir, f"{word}.txt")
#    with open(filepath, 'w', encoding='utf-8') as f:
#        f.write(str(data))
#
#def palindrome_worker():
#    print("[Worker] Palindrome cache worker started.")
#    used_palindromes = []
#
#    while True:
#        for word in VOCAB:
#            cached = get_cached_data(word, PALINDROME_CACHE_DIR)
#            if cached is None:
#                try:
#                    result = process_dictionary_word(word, used_palindromes)
#                    if result and result.get('palindrome'):
#                        pal = result['palindrome']
#                        write_cached_data(word, PALINDROME_CACHE_DIR, pal)
#                        used_palindromes.append(pal)
#                    else:
#                        write_cached_data(word, PALINDROME_CACHE_DIR, "NONE")
#                except Exception as e:
#                    print(f"[Worker] Palindrome error for {word}: {e}")
#
#                time.sleep(0.05)
#        time.sleep(10.0)
import os
import re
import json
import time
from collections import defaultdict, Counter
import threading
from functools import lru_cache
import nltk
from nltk.corpus import cmudict, wordnet

# Ensure NLTK datasets are downloaded
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)

nltk_lock = threading.Lock()

def safe_wordnet_all_lemma_names():
    with nltk_lock:
        return list(wordnet.all_lemma_names())

CMU_DICT = cmudict.dict()

VALID_SHORT_WORDS = {
    "a", "i", "in", "on", "no", "is", "it", "or", "to", "at", "am", "an",
    "so", "do", "go", "me", "my", "we", "he", "be", "us", "up", "if"
}

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
    w.lower() for w in set(safe_wordnet_all_lemma_names())
    if is_valid_real_word(w)
]
VOCAB_SET = set(VOCAB)

WORD_USAGE_COUNTS = defaultdict(int)

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

@lru_cache(maxsize=131072)
def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def min_edit_distance_lower_bound(s):
    """Mathematical lower bound on edit distance for string symmetry based on odd character parity."""
    counts = Counter(s)
    odd_counts = sum(1 for count in counts.values() if count % 2 != 0)
    return max(0, (odd_counts - 1) // 2)

def is_phrase_valid(phrase):
    words = phrase.lower().split()
    return all(w in VOCAB_SET for w in words)

def get_word_weight(word, decay_rate=0.15):
    count = WORD_USAGE_COUNTS[word.lower()]
    return 1.0 / (1.0 + decay_rate * (count ** 0.75))

def record_phrase_usage(phrase):
    for word in phrase.lower().split():
        WORD_USAGE_COUNTS[word] += 1

def extract_structural_candidates(target_word, max_candidates=50):
    target_rev = target_word.lower()[::-1]
    candidates = set()
    t_len = len(target_rev)

    for slice_len in range(t_len, 1, -1):
        sub_prefix = target_rev[:slice_len]
        matches = TRIE.get_valid_prefixes(sub_prefix)
        for m in matches:
            if m != target_word.lower() and len(m) >= 2:
                candidates.add(m)
        if len(candidates) >= max_candidates:
            break

    if len(candidates) < max_candidates and t_len >= 4:
        for i in range(t_len - 2):
            ngram = target_rev[i:i+3]
            for w in VOCAB:
                if len(w) >= 3 and ngram in w:
                    candidates.add(w)
                    if len(candidates) >= max_candidates:
                        break

    return list(candidates)

def find_dynamic_pivots(left_str, right_str, max_pivots=15):
    pivots = []
    for w in VOCAB:
        if len(w) < 2 and len(left_str) > 2:
            continue

        combined = left_str + w + right_str
        clean_c = re.sub(r'[^a-z]', '', combined.lower())

        # Exact lower-bound check before calling edit_distance
        lower_bound = min_edit_distance_lower_bound(clean_c)
        if lower_bound / len(clean_c) > 0.8:
            continue

        fudge = edit_distance(clean_c, clean_c[::-1])

        if fudge / len(clean_c) <= 0.8:
            pivots.append((w, fudge))

    pivots.sort(key=lambda x: x[1])
    return [p[0] for p in pivots[:max_pivots]]

def calculate_entropy_fudge(phrase):
    words = phrase.lower().split()
    clean_phrase = re.sub(r'[^a-z]', '', phrase.lower())

    if not clean_phrase:
        return float('inf')

    raw_fudge = edit_distance(clean_phrase, clean_phrase[::-1])
    short_word_count = sum(1 for w in words if len(w) == 1)
    short_penalty = short_word_count * 2.0
    unique_words = set(words)
    repetition_penalty = (len(words) - len(unique_words)) * 1.5
    length_bonus = sum(0.3 for w in words if len(w) >= 4)

    return raw_fudge + short_penalty + repetition_penalty - length_bonus

def generate_target_palindrome(target_word, used_palindromes, max_results=20, max_fudge_ratio=1.2):
    target = target_word.lower()
    if target not in VOCAB_SET:
        return {}

    results = {}
    target_rev = target[::-1]
    candidates = extract_structural_candidates(target)

    for w2 in candidates:
        dynamic_pivots = find_dynamic_pivots(target, w2)
        if not dynamic_pivots:
            dynamic_pivots = [""]

        for pivot in dynamic_pivots:
            phrase_variants = [
                f"{target} {pivot} {w2}".strip(),
                f"{target} {w2[::-1]} {pivot} {w2}".strip(),
            ]

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

    if not results:
        fallback_pivots = find_dynamic_pivots(target, target_rev, max_pivots=10)
        for pivot in fallback_pivots:
            fallback = f"{target} {pivot} {target_rev}".strip()
            fudge_score = calculate_entropy_fudge(fallback)
            results[fallback] = round(fudge_score, 2)

    filtered_results = [
        result for result in results
        if result != target_word and result not in used_palindromes
    ]

    sorted_results = dict(sorted({k: results[k] for k in filtered_results}.items(), key=lambda item: item[1]))
    return dict(list(sorted_results.items())[:max_results])

def process_dictionary_word(word, used_palindromes):
    candidates = generate_target_palindrome(word, used_palindromes, max_results=20)
    if not candidates:
        return None

    best_phrase = next(iter(candidates.keys()))
    record_phrase_usage(best_phrase)

    return {
        "target": word,
        "palindrome": best_phrase,
        "fudge": candidates[best_phrase],
        "all_candidates": candidates
    }

PALINDROME_CACHE_DIR = os.path.join(os.path.dirname(__file__), "palindrome_cache")
os.makedirs(PALINDROME_CACHE_DIR, exist_ok=True)

def get_cached_data(word, cache_dir=PALINDROME_CACHE_DIR):
    filepath = os.path.join(cache_dir, f"{word}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def update_cached_data(word, new_entries, cache_dir=PALINDROME_CACHE_DIR, max_stored=10):
    """Accumulates and updates top ranked pseudo-palindromes across multiple runs."""
    filepath = os.path.join(cache_dir, f"{word}.json")
    existing = []

    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    existing = data
        except Exception:
            pass

    seen_phrases = {entry["phrase"]: entry for entry in existing if isinstance(entry, dict)}

    for phrase, fudge in new_entries.items():
        if phrase not in seen_phrases or fudge < seen_phrases[phrase]["fudge"]:
            seen_phrases[phrase] = {"phrase": phrase, "fudge": fudge}

    sorted_entries = sorted(seen_phrases.values(), key=lambda x: x["fudge"])[:max_stored]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sorted_entries, f, indent=2)

def palindrome_worker():
    print("[Worker] Exhaustive multi-run palindrome worker started.")
    used_palindromes = []
    run_count = 0

    while True:
        run_count += 1
        print(f"[Worker] Starting sweep run #{run_count}")
        for word in VOCAB:
            try:
                result = process_dictionary_word(word, used_palindromes)
                if result and result.get('all_candidates'):
                    update_cached_data(word, result['all_candidates'], PALINDROME_CACHE_DIR)
                    used_palindromes.append(result['palindrome'])
            except Exception as e:
                print(f"[Worker] Palindrome error for {word}: {e}")

            time.sleep(0.001)
        time.sleep(5.0)
