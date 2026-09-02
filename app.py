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
from nltk.corpus import wordnet, words
import pronouncing
from collections import defaultdict
import random
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory

# Ensure NLTK datasets are downloaded
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================
# 1. Phonetic Rhyme Engine
# ==========================================
class UnifiedPhonicsEngine:
    def __init__(self, max_word_length=20):
        self.max_word_length = max_word_length
        self.vowel_phonemes = {
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
            'EH', 'ER', 'EY', 'IH', 'IY', 'OW',
            'OY', 'UH', 'UW'
        }
        self.word_profiles = {}
        self.rhyme_matrix = defaultdict(lambda: defaultdict(list))
        self.phone_to_words = defaultdict(list)
        self._build_indices()

    def _clean_word(self, word):
        return re.sub(r'[^a-z]', '', word.lower())

    def _extract_phonetic_parts(self, phones_str):
        tokens = phones_str.split()
        stresses = "".join([char for token in tokens for char in token if char.isdigit()])
        onset = tokens[0] if tokens else ""

        stressed_idx = -1
        primary_vowel = ""
        for i, t in enumerate(tokens):
            clean_p = "".join([c for c in t if not c.isdigit()])
            if clean_p in self.vowel_phonemes:
                if '1' in t or stressed_idx == -1:
                    stressed_idx = i
                    primary_vowel = clean_p
                    if '1' in t:
                        break

        if stressed_idx == -1:
            return None

        rhyme_tokens = ["".join([c for c in t if not c.isdigit()]) for t in tokens[stressed_idx:]]
        rhyme_tail = "_".join(rhyme_tokens)

        return {
            "stress": stresses,
            "syllables": len(stresses),
            "onset": onset,
            "vowel": primary_vowel,
            "rhyme_tail": rhyme_tail
        }

    def _build_indices(self):
        all_words = pronouncing.search(".*")
        for word in all_words:
            clean = self._clean_word(word)
            if not clean or len(clean) > self.max_word_length or clean in self.word_profiles:
                continue

            phones_list = pronouncing.phones_for_word(clean)
            if not phones_list:
                continue

            raw_phones = phones_list[0]
            parts = self._extract_phonetic_parts(raw_phones)
            if not parts:
                continue

            self.word_profiles[clean] = parts
            self.rhyme_matrix[parts["stress"]][parts["rhyme_tail"]].append(clean)
            
            clean_phones = re.sub(r'\d+', '', raw_phones)
            self.phone_to_words[clean_phones].append(clean)

    def get_homophones(self, target_word):
        clean = self._clean_word(target_word)
        phones_list = pronouncing.phones_for_word(clean)
        if not phones_list:
            return []
        clean_phones = re.sub(r'\d+', '', phones_list[0])
        matches = self.phone_to_words.get(clean_phones, [])
        return [w for w in matches if w != clean]

    def get_group_for_word(self, target_word):
        clean = self._clean_word(target_word)
        profile = self.word_profiles.get(clean)
        if not profile:
            phones = pronouncing.phones_for_word(clean)
            if phones:
                profile = self._extract_phonetic_parts(phones[0])
        
        if profile:
            stress = profile["stress"]
            tail = profile["rhyme_tail"]
            word_list = self.rhyme_matrix[stress].get(tail, [clean])
            label = f"Stress {stress}, Tail {tail}"
            return label, word_list
        return None, None

    def generate_rhyme_groups(self, max_syllables=8, min_rhymes=3):
        groups = []
        for stress in sorted(self.rhyme_matrix.keys(), key=lambda s: (len(s), s)):
            if len(stress) > max_syllables:
                continue
            for tail, word_list in self.rhyme_matrix[stress].items():
                if len(word_list) >= min_rhymes:
                    label = f"Stress {stress}, Tail {tail}"
                    groups.append((label, word_list))
        random.shuffle(groups)
        return groups


# ==========================================
# 2. Morse Audio Generator
# ==========================================
class MorseAudioGenerator:
    def __init__(self, freq=432, sample_rate=44100):
        self.freq = freq
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

    def _generate_tone(self, duration_ms):
        t = np.linspace(0, duration_ms / 1000.0, int(self.sample_rate * (duration_ms / 1000.0)), False)
        tone = np.sin(2 * np.pi * self.freq * t)
        fade_len = int(self.sample_rate * 0.005)
        if len(tone) > 2 * fade_len:
            tone[:fade_len] *= np.linspace(0, 1, fade_len)
            tone[-fade_len:] *= np.linspace(1, 0, fade_len)
        return tone

    def spell_to_morse_wav(self, word, filename, dot_ms=60, silence_ms=800):
        dash_ms = dot_ms * 3
        elem_space = np.zeros(int(self.sample_rate * (dot_ms / 1000.0)))
        char_space = np.zeros(int(self.sample_rate * (dash_ms / 1000.0)))

        audio_chunks = []
        for char in word.upper():
            if char in self.morse_code:
                pattern = self.morse_code[char]
                for symbol in pattern:
                    audio_chunks.append(self._generate_tone(dot_ms if symbol == '.' else dash_ms))
                    audio_chunks.append(elem_space)
                audio_chunks.append(char_space)

        # Pad with silence at end
        audio_chunks.append(np.zeros(int(self.sample_rate * (silence_ms / 1000.0))))

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            scaled = np.int16(full_audio / np.max(np.abs(full_audio)) * 32767)
            wavpath = os.path.join(CACHE_DIR, filename)
            wavfile.write(wavpath, self.sample_rate, scaled)
            return filename
        return None


# ==========================================
# 3. Synchronized Audio State Manager
# ==========================================
class SyncedNarratorState:
    def __init__(self, engine_ref, speech_rate=120, ollama_url="http://127.0.0.1:11434", model_name="qwen3"):
        self.phonics_engine = engine_ref
        self.morse_gen = MorseAudioGenerator(freq=432)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', speech_rate)
        
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.valid_english = set(words.words())
        self.pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 's': 'adjective', 'r': 'adverb'}
        
        self.current_state = {
            "group_label": "Initializing...",
            "wordlist": [],
            "current_word": "",
            "step": "Idle",
            "metadata": {},
            "audio_file": None,
            "audio_id": 0
        }
        self.priority_queue = []
        self.lock = threading.Lock()

    def _pad_wav_with_silence(self, filepath, silence_duration_sec=1.0):
        """Reads a generated WAV and appends trailing silence so text isn't cut off."""
        try:
            sr, data = wavfile.read(filepath)
            silence_samples = int(sr * silence_duration_sec)
            
            if data.ndim == 1:
                silence = np.zeros(silence_samples, dtype=data.dtype)
            else:
                silence = np.zeros((silence_samples, data.shape[1]), dtype=data.dtype)

            padded_data = np.concatenate((data, silence))
            wavfile.write(filepath, sr, padded_data)
        except Exception as e:
            print(f"Error padding audio: {e}")

    def _generate_tts_wav(self, text, filename):
        filepath = os.path.join(CACHE_DIR, filename)
        self.tts_engine.save_to_file(text, filepath)
        self.tts_engine.runAndWait()
        self._pad_wav_with_silence(filepath, silence_duration_sec=0.8)
        return filename

    def _get_wav_duration(self, filename):
        filepath = os.path.join(CACHE_DIR, filename)
        try:
            with wave.open(filepath, 'r') as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        except Exception:
            return 2.5

    def _generate_ollama_example(self, word):
        prompt = f"Write one very short, simple, kid-friendly sentence using the word '{word}'. Output only the sentence."
        payload = json.dumps({"model": self.model_name, "prompt": prompt, "stream": False}).encode('utf-8')
        req = urllib.request.Request(f"{self.ollama_url}/api/generate", data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data.get("response", "").strip()
        except Exception:
            return None

    def clean_and_deduplicate_list(self, raw_words):
        filtered = []
        for w in raw_words:
            w_clean = w.lower().strip()
            has_definition = bool(wordnet.synsets(w_clean))
            #if w_clean in self.valid_english or has_definition:
            if w_clean in self.valid_english and has_definition:
                filtered.append(w_clean)
        return list(dict.fromkeys(filtered))

    def get_word_details(self, word):
        synsets = wordnet.synsets(word)
        homophones = self.phonics_engine.get_homophones(word)
        
        if not synsets:
            example = self._generate_ollama_example(word)
            return {
                "pos": "word",
                "definition": f"The word is {word}.",
                "example": example,
                "synonyms": [],
                "antonyms": [],
                "hypernyms": [],
                "hyponyms": [],
                "homophones": homophones
            }

        syn = synsets[0]
        pos_full = self.pos_map.get(syn.pos(), 'word')
        definition = syn.definition()
        examples = syn.examples()
        example = examples[0] if examples else self._generate_ollama_example(word)

        synonyms, antonyms, hypernyms, hyponyms = set(), set(), set(), set()
        for s in synsets:
            for lemma in s.lemmas():
                clean_lemma = lemma.name().replace('_', ' ')
                if clean_lemma.lower() != word.lower():
                    synonyms.add(clean_lemma)
                if lemma.antonyms():
                    for ant in lemma.antonyms():
                        antonyms.add(ant.name().replace('_', ' '))
            for hyp in s.hypernyms():
                for lemma in hyp.lemmas():
                    hypernyms.add(lemma.name().replace('_', ' '))
            for hyp in s.hyponyms():
                for lemma in hyp.lemmas():
                    hyponyms.add(lemma.name().replace('_', ' '))

        return {
            "pos": pos_full,
            "definition": definition,
            "example": example,
            "synonyms": list(synonyms)[:5],
            "antonyms": list(antonyms)[:5],
            "hypernyms": list(hypernyms)[:5],
            "hyponyms": list(hyponyms)[:5],
            "homophones": homophones[:5]
        }

    def _broadcast_phrase(self, step_name, text, file_prefix, is_morse=False, word=""):
        filename = f"{file_prefix}.wav"
        
        if is_morse:
            self.morse_gen.spell_to_morse_wav(word, filename)
        else:
            self._generate_tts_wav(text, filename)

        duration = self._get_wav_duration(filename)

        with self.lock:
            self.current_state["step"] = step_name
            self.current_state["audio_file"] = filename
            self.current_state["audio_id"] += 1

        # Sleep exact duration of audio file plus safety margin
        time.sleep(duration + 0.3)

    def narrate_group(self, group_label, raw_word_list):
        group_words = self.clean_and_deduplicate_list(raw_word_list)
        if not group_words:
            return

        words_str = ", ".join(group_words)
        
        with self.lock:
            self.current_state["group_label"] = group_label
            self.current_state["wordlist"] = group_words

        for word in group_words:
            details = self.get_word_details(word)

            with self.lock:
                self.current_state["current_word"] = word
                self.current_state["metadata"] = details

            self._broadcast_phrase("Announcing Group List", f"Group list: {words_str}.", "group_list")
            self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
            self._broadcast_phrase("Morse Code Spelling", "", "morse", is_morse=True, word=word)
            self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")

            if details['pos']:
                self._broadcast_phrase("Part of Speech", f"Part of speech: {details['pos']}.", "pos")

            if details['definition']:
                self._broadcast_phrase("Definition", f"Definition: {details['definition']}", "def")

            if details['example']:
                self._broadcast_phrase("Example Sentence", f"Example: {details['example']}", "example")

            if details['synonyms']:
                self._broadcast_phrase("Synonyms", f"Synonyms: {', '.join(details['synonyms'])}.", "synonyms")

            if details['antonyms']:
                self._broadcast_phrase("Antonyms", f"Antonyms: {', '.join(details['antonyms'])}.", "antonyms")

            #if details['homophones']:
            #    self._broadcast_phrase("Homophones", f"Homophones: {', '.join(details['homophones'])}.", "homophones")

            if details['hypernyms']:
                self._broadcast_phrase("Hypernyms", f"Hypernyms: {', '.join(details['hypernyms'])}.", "hypernyms")

            if details['hyponyms']:
                self._broadcast_phrase("Hyponyms", f"Hyponyms: {', '.join(details['hyponyms'])}.", "hyponyms")

            self._broadcast_phrase("Repeating Word", f"Word: {word}.", "repeat_word")
            self._broadcast_phrase("Morse Code Spelling", "", "morse_repeat", is_morse=True, word=word)
            self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")

    def enqueue_priority_word(self, word):
        label, word_list = self.phonics_engine.get_group_for_word(word)
        if label and word_list:
            with self.lock:
                self.priority_queue.insert(0, (label, word_list))
            return True, label
        return False, "Word not found in dictionary."


# ==========================================
# 4. Flask Application & Background Worker
# ==========================================
app = Flask(__name__)

phonics_engine = UnifiedPhonicsEngine(max_word_length=20)
narrator_state = SyncedNarratorState(engine_ref=phonics_engine, speech_rate=120, model_name="qwen3")

def narration_worker():
    rhyme_groups = phonics_engine.generate_rhyme_groups(max_syllables=8, min_rhymes=3)
    group_idx = 0

    while True:
        next_group = None
        with narrator_state.lock:
            if narrator_state.priority_queue:
                next_group = narrator_state.priority_queue.pop(0)

        if not next_group:
            if not rhyme_groups:
                rhyme_groups = phonics_engine.generate_rhyme_groups(max_syllables=8, min_rhymes=3)
                group_idx = 0
            next_group = rhyme_groups[group_idx % len(rhyme_groups)]
            group_idx += 1

        label, word_list = next_group
        narrator_state.narrate_group(label, word_list)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(CACHE_DIR, filename)

@app.route("/api/state")
def get_state():
    with narrator_state.lock:
        return jsonify(narrator_state.current_state)

@app.route("/api/search", methods=["POST"])
def search_word():
    data = request.get_json() or {}
    word = data.get("word", "").strip().lower()
    if not word:
        return jsonify({"status": "error", "message": "No word provided"}), 400

    success, message = narrator_state.enqueue_priority_word(word)
    if success:
        return jsonify({"status": "success", "message": f"Queued group for word '{word}' ({message})"})
    return jsonify({"status": "error", "message": message}), 404

if __name__ == "__main__":
    t = threading.Thread(target=narration_worker, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5003, debug=False)
