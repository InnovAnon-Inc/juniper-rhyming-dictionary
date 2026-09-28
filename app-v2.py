#! /usr/bin/env python3
import asyncio
import json
import math
import os
import re
import threading
import time
import random
from collections import defaultdict
import numpy as np
import pyttsx3
import websockets
import urllib.request
from flask import Flask, render_template_string, request, jsonify
import nltk
from nltk.corpus import cmudict, wordnet, words
import pronouncing

# Ensure NLTK datasets are downloaded safely
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)
nltk.download('cmudict', quiet=True)

nltk_lock = threading.Lock()

def safe_wordnet_synsets(word):
    with nltk_lock:
        return wordnet.synsets(word)

def safe_wordnet_all_lemma_names():
    with nltk_lock:
        return list(wordnet.all_lemma_names())

app = Flask(__name__)

# ==========================================
# CONFIGURATION & DISK CACHES
# ==========================================
UPSTREAM_CHIMES_WS = "ws://127.0.0.1:65402"  # Chimes v2 WS endpoint
FALLBACK_FREQ = 432.0                       # Fallback pitch if chimes offline
BPM_NARRATION = 120                          # 120 BPM = 8th notes relative to 60 BPM chimes
DOT_DURATION_16TH = (60.0 / BPM_NARRATION) / 2.0  # 0.125s per 16th note

CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
PALINDROME_CACHE_DIR = os.path.join(os.path.dirname(__file__), "palindrome_cache")
LLM_CACHE_DIR = os.path.join(os.path.dirname(__file__), "llm_cache")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PALINDROME_CACHE_DIR, exist_ok=True)
os.makedirs(LLM_CACHE_DIR, exist_ok=True)

current_chord_tone_1 = FALLBACK_FREQ
latest_chimes_state = {}

# ==========================================
# VOCABULARY, TRIE & FREQUENCY TRACKING
# ==========================================
CMU_DICT = cmudict.dict()
VALID_SHORT_WORDS = {"a", "i", "in", "on", "no", "is", "it", "or", "to", "at", "am", "an", "so", "do", "go", "me", "my", "we", "he", "be", "us", "up", "if"}
WORD_USAGE_COUNTS = defaultdict(int)

def is_valid_real_word(w):
    w_clean = w.lower()
    if not w_clean.isalpha():
        return False
    if len(w_clean) < 3 and w_clean not in VALID_SHORT_WORDS:
        return False
    if w_clean != w and w.isupper():
        return False
    return w_clean in CMU_DICT

VOCAB = [w.lower() for w in set(safe_wordnet_all_lemma_names()) if is_valid_real_word(w)]
VOCAB_SET = set(VOCAB)

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

# ==========================================
# ASYMMETRIC SEMI-PALINDROME ENGINE
# ==========================================
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

def is_phrase_valid(phrase):
    words_list = phrase.lower().split()
    return all(w in VOCAB_SET for w in words_list)

def get_word_weight(word, decay_rate=0.15):
    count = WORD_USAGE_COUNTS[word.lower()]
    return 1.0 / (1.0 + decay_rate * (count ** 0.75))

def record_phrase_usage(phrase):
    for word in phrase.lower().split():
        WORD_USAGE_COUNTS[word] += 1

def extract_structural_candidates(target_word, max_candidates=30):
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

def find_dynamic_pivots(left_str, right_str, max_pivots=10):
    pivots = []
    for w in VOCAB:
        if len(w) < 2 and len(left_str) > 2:
            continue
        combined = left_str + w + right_str
        clean_c = re.sub(r'[^a-z]', '', combined.lower())
        fudge = edit_distance(clean_c, clean_c[::-1])
        if len(clean_c) > 0 and (fudge / len(clean_c)) <= 0.8:
            pivots.append((w, fudge))

    pivots.sort(key=lambda x: x[1])
    return [p[0] for p in pivots[:max_pivots]]

def calculate_entropy_fudge(phrase):
    words_list = phrase.lower().split()
    clean_phrase = re.sub(r'[^a-z]', '', phrase.lower())

    if not clean_phrase:
        return float('inf')

    raw_fudge = edit_distance(clean_phrase, clean_phrase[::-1])
    short_word_count = sum(1 for w in words_list if len(w) == 1)
    short_penalty = short_word_count * 2.0
    unique_words = set(words_list)
    repetition_penalty = (len(words_list) - len(unique_words)) * 1.5
    length_bonus = sum(0.3 for w in words_list if len(w) >= 4)

    return raw_fudge + short_penalty + repetition_penalty - length_bonus

def generate_target_palindrome(target_word, used_palindromes, max_results=10, max_fudge_ratio=1.2):
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
                if not clean_p:
                    continue
                fudge_score = calculate_entropy_fudge(phrase)
                fudge_ratio = fudge_score / len(clean_p)

                if fudge_ratio <= max_fudge_ratio and is_phrase_valid(phrase):
                    if phrase not in results or fudge_score < results[phrase]:
                        results[phrase] = round(fudge_score, 2)

    if not results:
        fallback_pivots = find_dynamic_pivots(target, target_rev, max_pivots=5)
        for pivot in fallback_pivots:
            fallback = f"{target} {pivot} {target_rev}".strip()
            fudge_score = calculate_entropy_fudge(fallback)
            results[fallback] = round(fudge_score, 2)

    filtered = {
        k: v for k, v in results.items() 
        if k != target_word and k not in used_palindromes
    }
    sorted_results = dict(sorted(filtered.items(), key=lambda item: item[1]))
    return dict(list(sorted_results.items())[:max_results])

def process_dictionary_word(word, used_palindromes):
    candidates = generate_target_palindrome(word, used_palindromes, max_results=10)
    if not candidates:
        return None

    best_phrase = next(iter(candidates.keys()))
    record_phrase_usage(best_phrase)

    return {
        "target": word,
        "palindrome": best_phrase,
        "fudge": candidates[best_phrase],
        "word_weights": {w: round(get_word_weight(w), 3) for w in best_phrase.split()}
    }

# ==========================================
# DISK CACHING HELPERS & WORKER THREADS
# ==========================================
def get_cached_data(word, cache_dir):
    filepath = os.path.join(cache_dir, f"{word}.txt")
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content if content else None
    return None

def write_cached_data(word, cache_dir, data):
    filepath = os.path.join(cache_dir, f"{word}.txt")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(str(data))

def palindrome_worker():
    print("[WORKER] Palindrome background cache worker started.")
    used_palindromes = []
    for word in VOCAB:
        cached = get_cached_data(word, PALINDROME_CACHE_DIR)
        if cached is None:
            try:
                res = process_dictionary_word(word, used_palindromes)
                if res and res.get('palindrome'):
                    pal = res['palindrome']
                    write_cached_data(word, PALINDROME_CACHE_DIR, pal)
                    used_palindromes.append(pal)
                else:
                    write_cached_data(word, PALINDROME_CACHE_DIR, "NONE")
            except Exception as e:
                print(f"[WORKER] Palindrome error for {word}: {e}")
            time.sleep(0.05)

def llm_example_worker(narrator_ref):
    print("[WORKER] LLM usage-example background cache worker started.")
    for word in VOCAB:
        cached = get_cached_data(word, LLM_CACHE_DIR)
        if cached is None:
            synsets = safe_wordnet_synsets(word)
            definition = synsets[0].definition() if synsets else None
            try:
                example = narrator_ref._generate_ollama_example(word, definition)
                if example:
                    write_cached_data(word, LLM_CACHE_DIR, example)
                else:
                    write_cached_data(word, LLM_CACHE_DIR, "NONE")
            except Exception as e:
                print(f"[WORKER] LLM error for {word}: {e}")
            time.sleep(1.0)

# ==========================================
# UPSTREAM CHIMES V2 WEBSOCKET LISTENER
# ==========================================
async def listen_to_chimes_v2():
    global current_chord_tone_1, latest_chimes_state
    print(f"[RHYMING DIC V2] Connecting to chimes-v2 on {UPSTREAM_CHIMES_WS}...")
    while True:
        try:
            async with websockets.connect(UPSTREAM_CHIMES_WS) as ws:
                print("[RHYMING DIC V2] Synchronized with chimes-v2 stream.")
                async for msg in ws:
                    data = json.loads(msg)
                    latest_chimes_state = data
                    lh_freqs = data.get("left_hand", {}).get("frequencies", [])
                    if lh_freqs:
                        current_chord_tone_1 = float(lh_freqs[0])
        except (websockets.exceptions.ConnectionClosedError, OSError):
            current_chord_tone_1 = FALLBACK_FREQ
            await asyncio.sleep(2.0)

def start_ws_client_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(listen_to_chimes_v2())

# ==========================================
# 16TH-NOTE MORSE CODE GENERATOR
# ==========================================
class MorseV2AudioGenerator:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.morse_code = {
            'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
            'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
            'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
            'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
            'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
            '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
            '9': '----.', '0': '-----'
        }

    def generate_tone(self, duration, freq):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        envelope = np.ones_like(t)
        fade = int(0.005 * self.sample_rate)
        if len(t) > 2 * fade:
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
        return (np.sin(2 * np.pi * freq * t) * envelope * 0.5 * 32767).astype(np.int16)

    def text_to_morse_audio_v2(self, text):
        global current_chord_tone_1
        freq = current_chord_tone_1
        dot_len = DOT_DURATION_16TH
        
        now = time.time()
        sub_second_drift = now - math.floor(now)
        jitter_adjustment = (0.5 - sub_second_drift) * 0.05
        
        farnsworth_char_pause = max(0.05, (dot_len * 3) + jitter_adjustment)
        farnsworth_word_pause = max(0.10, (dot_len * 7) + jitter_adjustment)

        audio_parts = []
        words_list = text.upper().split()

        for w_idx, word in enumerate(words_list):
            for char in word:
                if char in self.morse_code:
                    symbols = self.morse_code[char]
                    for s_idx, sym in enumerate(symbols):
                        dur = dot_len if sym == '.' else dot_len * 3
                        audio_parts.append(self.generate_tone(dur, freq))
                        if s_idx < len(symbols) - 1:
                            audio_parts.append(np.zeros(int(self.sample_rate * dot_len), dtype=np.int16))
                    audio_parts.append(np.zeros(int(self.sample_rate * farnsworth_char_pause), dtype=np.int16))
            if w_idx < len(words_list) - 1:
                audio_parts.append(np.zeros(int(self.sample_rate * farnsworth_word_pause), dtype=np.int16))

        if audio_parts:
            return np.concatenate(audio_parts)
        return np.array([], dtype=np.int16)

# ==========================================
# WORDLIST TRAVERSAL & RHYME STEPPER ENGINE
# ==========================================
class WordListNavigationEngine:
    def __init__(self):
        self.vocab = VOCAB
        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}

    def get_rhymes_and_similar(self, word):
        word_clean = word.lower()
        rhyme_candidates = pronouncing.rhymes(word_clean)
        valid_rhymes = [r for r in rhyme_candidates if r in VOCAB_SET]

        # Step to similar phonetic/semantic wordlists via WordNet
        synsets = safe_wordnet_synsets(word_clean)
        similar_words = set()
        for syn in synsets[:3]:
            for lemma in syn.lemmas():
                name = lemma.name().lower()
                if name != word_clean and name in VOCAB_SET:
                    similar_words.add(name)

        return {
            "target": word_clean,
            "rhymes": valid_rhymes[:15],
            "similar_words": list(similar_words)[:15]
        }

    def step_next_wordlist(self, current_word):
        info = self.get_rhymes_and_similar(current_word)
        if info["rhymes"]:
            next_word = random.choice(info["rhymes"])
        elif info["similar_words"]:
            next_word = random.choice(info["similar_words"])
        else:
            curr_idx = self.word_to_index.get(current_word.lower(), 0)
            next_word = self.vocab[(curr_idx + 1) % len(self.vocab)]
        return next_word

nav_engine = WordListNavigationEngine()

# ==========================================
# NARRATION & FORMATTING ENGINE
# ==========================================
class NarrationEngineV2:
    def __init__(self):
        self.morse_gen = MorseV2AudioGenerator()

    def _generate_ollama_example(self, word, definition):
        try:
            prompt = f"Provide one concise, illustrative sentence using the word '{word}' (Definition: {definition}). Output ONLY the sentence."
            data = json.dumps({"model": "llama3", "prompt": prompt, "stream": False}).encode('utf-8')
            req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result.get("response", "").strip()
        except Exception:
            return f"The word '{word}' is used in formal writing."

    def speak_spelling_bee_format(self, word):
        synsets = safe_wordnet_synsets(word)
        definition = synsets[0].definition() if synsets else "No definition available."
        spelled = ", ".join(list(word.upper()))
        narration_text = f"Your word is {word}. Spelled: {spelled}. Definition: {definition}."
        morse_audio = self.morse_gen.text_to_morse_audio_v2(word)

        return {
            "format": "spelling_bee",
            "word": word,
            "spelled": spelled,
            "definition": definition,
            "narration_text": narration_text,
            "chord_tone_1_hz": current_chord_tone_1,
            "morse_samples": len(morse_audio)
        }

    def speak_dictionary_format(self, word):
        synsets = safe_wordnet_synsets(word)
        pos = synsets[0].pos() if synsets else "n/a"
        definition = synsets[0].definition() if synsets else "No definition available."
        
        # Check LLM cache or generate
        llm_example = get_cached_data(word, LLM_CACHE_DIR)
        if not llm_example or llm_example == "NONE":
            llm_example = self._generate_ollama_example(word, definition)

        # Check Palindrome cache
        palindrome_res = get_cached_data(word, PALINDROME_CACHE_DIR)
        if not palindrome_res or palindrome_res == "NONE":
            pal_dict = generate_target_palindrome(word, [], max_results=1)
            palindrome_res = next(iter(pal_dict.keys())) if pal_dict else "None found"

        rel = nav_engine.get_rhymes_and_similar(word)

        return {
            "format": "dictionary",
            "word": word,
            "pos": pos,
            "definition": definition,
            "usage_example": llm_example,
            "semi_palindrome": palindrome_res,
            "rhymes": rel["rhymes"],
            "similar_words": rel["similar_words"],
            "chord_tone_1_hz": current_chord_tone_1
        }

narrator_v2 = NarrationEngineV2()

# ==========================================
# FLASK WEB INTERFACE & REST API
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Rhyming Dictionary & Dynamic Wordlist Engine V2</title>
    <style>
        body { background-color: #121214; color: #e0e0e0; font-family: sans-serif; padding: 25px; }
        h1 { color: #4db6ac; margin-bottom: 5px; }
        .card { background: #1e1e24; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .badge { background: #00897b; color: #fff; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
        input[type="text"] { background: #282830; border: 1px solid #444; color: #fff; padding: 10px; border-radius: 4px; width: 250px; }
        button { background: #00e676; border: none; color: #000; font-weight: bold; padding: 10px 18px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #00c853; }
        pre { background: #18181c; padding: 15px; border-radius: 6px; color: #81dfe6; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>Rhyming Dictionary & Palindrome Engine V2</h1>
    <p>Synced with <code>chimes-v2</code> | Current Tonic CT1: <strong id="ct1-val">--</strong> Hz</p>

    <div class="card">
        <h3>Word Exploration & Stepping</h3>
        <input type="text" id="target-word" value="rhyme" placeholder="Enter word...">
        <button onclick="fetchWordData('spelling_bee')">Spelling Bee Format</button>
        <button onclick="fetchWordData('dictionary')">Dictionary Format</button>
        <button onclick="stepNext()">Step to Next Wordlist</button>
    </div>

    <div class="card">
        <h3>Active Output Payload</h3>
        <pre id="output">Select a mode or step through wordlists...</pre>
    </div>

    <script>
        async function fetchWordData(mode) {
            const word = document.getElementById('target-word').value.trim();
            const endpoint = mode === 'spelling_bee' ? '/api/narrate' : '/api/dictionary';
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ word: word })
            });
            const data = await res.json();
            document.getElementById('output').innerText = JSON.stringify(data, null, 2);
            if (data.chord_tone_1_hz) {
                document.getElementById('ct1-val').innerText = data.chord_tone_1_hz.toFixed(2);
            }
        }

        async function stepNext() {
            const word = document.getElementById('target-word').value.trim();
            const res = await fetch('/api/step', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ current_word: word })
            });
            const data = await res.json();
            document.getElementById('target-word').value = data.next_word;
            fetchWordData('dictionary');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/narrate', methods=['POST'])
def narrate_word():
    data = request.json or {}
    word = data.get('word', 'rhyme').lower()
    res = narrator_v2.speak_spelling_bee_format(word)
    return jsonify(res)

@app.route('/api/dictionary', methods=['POST'])
def dictionary_word():
    data = request.json or {}
    word = data.get('word', 'rhyme').lower()
    res = narrator_v2.speak_dictionary_format(word)
    return jsonify(res)

@app.route('/api/palindrome', methods=['POST'])
def get_palindrome():
    data = request.json or {}
    word = data.get('word', 'rhyme').lower()
    palindromes = generate_target_palindrome(word, [], max_results=5)
    return jsonify({
        "word": word,
        "semi_palindromes": palindromes
    })

@app.route('/api/step', methods=['POST'])
def step_wordlist():
    data = request.json or {}
    current_word = data.get('current_word', 'rhyme').lower()
    next_word = nav_engine.step_next_wordlist(current_word)
    return jsonify({
        "previous_word": current_word,
        "next_word": next_word,
        "details": nav_engine.get_rhymes_and_similar(next_word)
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "Rhyming Dictionary & Palindrome V2 Server Online",
        "upstream_chimes": UPSTREAM_CHIMES_WS,
        "chord_tone_1_hz": current_chord_tone_1,
        "vocabulary_size": len(VOCAB),
        "chimes_state": latest_chimes_state
    })

if __name__ == '__main__':
    # Start background synchronization and worker threads
    threading.Thread(target=start_ws_client_thread, daemon=True).start()
    threading.Thread(target=palindrome_worker, daemon=True).start()
    threading.Thread(target=llm_example_worker, args=(narrator_v2,), daemon=True).start()

    print("Running Merged Rhyming Dictionary V2 Server on http://0.0.0.0:5020")
    app.run(host='0.0.0.0', port=5020)
