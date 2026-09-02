#! /usr/bin/env python3

import os
import time
import re
import wave
import numpy as np
import json
import urllib.request
import subprocess
from scipy.io import wavfile
import pyttsx3
import nltk
from nltk.corpus import wordnet, words
import pronouncing
from collections import defaultdict
import random
import itertools
import threading
from flask import Flask, render_template, jsonify, send_from_directory

# Ensure NLTK datasets are downloaded
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Map standard stress patterns to classical poetic feet
POETIC_FEET_MAP = {
    "01": "Iambic",
    "10": "Trochaic",
    "001": "Anapestic",
    "100": "Dactylic",
    "11": "Spondaic",
    "00": "Pyrrhic",
    "010": "Amphibrachic",
    "101": "Amphimacer",
    "1000": "1st Paeon",
    "0100": "2nd Paeon",
    "0010": "3rd Paeon",
    "0001": "4th Paeon",
}

def identify_meter_name(stress_str):
    """Maps binary/ternary stress strings to classical poetic meter names."""
    # Simplify secondary stress (2) to primary/unstressed where applicable for matching
    simplified = stress_str.replace('2', '1')
    if simplified in POETIC_FEET_MAP:
        return POETIC_FEET_MAP[simplified]
    
    # Check if pattern repeats (e.g., 0101 -> Double Iambic / Iambic Dimeter)
    for unit_len in [2, 3]:
        if len(simplified) % unit_len == 0:
            unit = simplified[:unit_len]
            if unit in POETIC_FEET_MAP and unit * (len(simplified) // unit_len) == simplified:
                count = len(simplified) // unit_len
                feet_names = {1: "Monometer", 2: "Dimeter", 3: "Trimeter", 4: "Tetrameter"}
                return f"{POETIC_FEET_MAP[unit]} {feet_names.get(count, 'Meter')}"
                
    return "Composite Meter"


# ==========================================
# 1. Advanced Phonetic & Super Group Engine
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
            if not clean or len(clean) <= 1 or len(clean) > self.max_word_length or clean in self.word_profiles:
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


class SuperGroupEngine:
    def __init__(self, phonics_engine):
        self.phonics = phonics_engine

    def _partition_stress_string(self, target_stress_str):
        results = []
        available_stresses = set(self.phonics.rhyme_matrix.keys())

        def backtrack(remaining, current_path):
            if not remaining:
                results.append(current_path)
                return
            for i in range(1, len(remaining) + 1):
                chunk = remaining[:i]
                if chunk in available_stresses:
                    backtrack(remaining[i:], current_path + [chunk])

        backtrack(target_stress_str, [])
        return results

    def build_super_groups_for_length(self, target_syllable_length):
        super_groups = []
        possible_stresses = [''.join(p) for p in itertools.product(['0', '1', '2'], repeat=target_syllable_length)]

        for target_stress_str in possible_stresses:
            partitions = self._partition_stress_string(target_stress_str)
            if not partitions:
                continue

            meter_name = identify_meter_name(target_stress_str)

            for stress_sequence in partitions[:3]:
                combination_groups = []
                valid = True

                for s in stress_sequence:
                    tails = list(self.phonics.rhyme_matrix[s].keys())
                    if not tails:
                        valid = False
                        break
                    chosen_tail = random.choice(tails)
                    words = self.phonics.rhyme_matrix[s][chosen_tail]
                    combination_groups.append({
                        "label": f"Foot Pattern [{s}] (Rhyme Tail: {chosen_tail})",
                        "words": words
                    })

                if valid and combination_groups:
                    partition_str = " + ".join(stress_sequence)
                    super_groups.append({
                        "theme": f"{meter_name} Structure | Sequence '{target_stress_str}' [{partition_str}]",
                        "target_stress": target_stress_str,
                        "meter_name": meter_name,
                        "sub_groups": combination_groups
                    })

        random.shuffle(super_groups)
        return super_groups


# ==========================================
# 2. Lightweight & Fast TTS Engine Wrapper
# ==========================================
class FastTTSEngine:
    """Supports Piper TTS binary fallbacking to pyttsx3 for light CPU footprints."""
    def __init__(self, speech_rate=120):
        self.piper_path = "/usr/local/bin/piper" # Adjust path to your piper installation
        self.piper_model = "en_US-lessac-medium.onnx"
        self.use_piper = os.path.exists(self.piper_path) and os.path.exists(self.piper_model)
        
        if not self.use_piper:
            self.pyttsx_engine = pyttsx3.init()
            self.pyttsx_engine.setProperty('rate', speech_rate)

    def generate_wav(self, text, filepath):
        if self.use_piper:
            cmd = f'echo "{text}" | {self.piper_path} --model {self.piper_model} --output_file {filepath}'
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            self.pyttsx_engine.save_to_file(text, filepath)
            self.pyttsx_engine.runAndWait()


# ==========================================
# 3. Morse Code Audio Synthesizer
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

        audio_chunks.append(np.zeros(int(self.sample_rate * (silence_ms / 1000.0))))

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            scaled = np.int16(full_audio / np.max(np.abs(full_audio)) * 32767)
            wavpath = os.path.join(CACHE_DIR, filename)
            wavfile.write(wavpath, self.sample_rate, scaled)
            return filename
        return None


# ==========================================
# 4. Synchronized Audio State Manager
# ==========================================
class SyncedNarratorState:
    def __init__(self, engine_ref, speech_rate=120, ollama_url="http://127.0.0.1:11434", model_name="qwen3"):
        self.phonics_engine = engine_ref
        self.super_engine = SuperGroupEngine(engine_ref)
        self.morse_gen = MorseAudioGenerator(freq=432)
        self.tts = FastTTSEngine(speech_rate=speech_rate)
        
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.valid_english = set(words.words())
        self.pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 's': 'adjective', 'r': 'adverb'}
        
        self.current_state = {
            "super_group_theme": "Initializing...",
            "group_label": "Initializing...",
            "wordlist": [],
            "current_word": "",
            "step": "Idle",
            "metadata": {},
            "audio_file": None,
            "audio_id": 0
        }
        self.lock = threading.Lock()

    def _pad_wav_with_silence(self, filepath, silence_duration_sec=0.8):
        try:
            sr, data = wavfile.read(filepath)
            silence_samples = int(sr * silence_duration_sec)
            silence = np.zeros(silence_samples, dtype=data.dtype) if data.ndim == 1 else np.zeros((silence_samples, data.shape[1]), dtype=data.dtype)
            padded_data = np.concatenate((data, silence))
            wavfile.write(filepath, sr, padded_data)
        except Exception as e:
            print(f"Error padding audio: {e}")

    def _generate_tts_wav(self, text, filename):
        filepath = os.path.join(CACHE_DIR, filename)
        self.tts.generate_wav(text, filepath)
        self._pad_wav_with_silence(filepath, silence_duration_sec=0.8)
        return filename

    def _get_wav_duration(self, filename):
        filepath = os.path.join(CACHE_DIR, filename)
        try:
            with wave.open(filepath, 'r') as f:
                return f.getnframes() / float(f.getframerate())
        except Exception:
            return 2.5

    def clean_and_deduplicate_list(self, raw_words):
        filtered = []
        for w in raw_words:
            w_clean = w.lower().strip()
            if len(w_clean) > 1 and w_clean in self.valid_english and bool(wordnet.synsets(w_clean)):
                filtered.append(w_clean)
        return list(dict.fromkeys(filtered))

    def get_word_details(self, word):
        synsets = wordnet.synsets(word)
        if not synsets:
            return {"word": word, "senses": []}

        senses = []
        for syn in synsets[:3]:
            pos_full = self.pos_map.get(syn.pos(), 'word')
            definition = syn.definition()
            raw_examples = syn.examples()
            examples = [e for e in raw_examples if word.lower() in e.lower()][:1]

            synonyms, antonyms = set(), set()
            for lemma in syn.lemmas():
                clean_lemma = lemma.name().replace('_', ' ')
                if clean_lemma.lower() != word.lower():
                    synonyms.add(clean_lemma)
                if lemma.antonyms():
                    for ant in lemma.antonyms():
                        antonyms.add(ant.name().replace('_', ' '))

            senses.append({
                "pos": pos_full,
                "definition": definition,
                "examples": examples,
                "synonyms": list(synonyms)[:3],
                "antonyms": list(antonyms)[:3]
            })

        return {"word": word, "senses": senses}

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

        time.sleep(duration + 0.3)

    def narrate_super_group(self, super_group):
        theme = super_group["theme"]
        sub_groups = super_group["sub_groups"]

        with self.lock:
            self.current_state["super_group_theme"] = theme
            self.current_state["step"] = "Starting Super Group"

        for group in sub_groups:
            group_label = group["label"]
            group_words = self.clean_and_deduplicate_list(group["words"])
            
            if not group_words:
                continue

            words_str = ", ".join(group_words)
            with self.lock:
                self.current_state["group_label"] = group_label
                self.current_state["wordlist"] = group_words

            self._broadcast_phrase("Announcing Group List", f"Sub-group rhyming list: {words_str}.", "group_list")

            for word in group_words:
                details = self.get_word_details(word)

                with self.lock:
                    self.current_state["current_word"] = word
                    self.current_state["metadata"] = details

                self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
                self._broadcast_phrase("Morse Code Spelling", "", "morse", is_morse=True, word=word)

                # Combined concise narration (eliminates separate POS/Def repeating calls)
                for idx, sense in enumerate(details["senses"], start=1):
                    concise_phrase = f"As a {sense['pos']}: {sense['definition']}."
                    self._broadcast_phrase("Definition", concise_phrase, "def")
                    
                    for ex in sense["examples"]:
                        self._broadcast_phrase("Example Sentence", f"Example: {ex}", "example")

                self._broadcast_phrase("Repeating Word", f"Word: {word}.", "repeat_word")


# ==========================================
# 5. Flask Application & Outer Loop Execution
# ==========================================
app = Flask(__name__)

phonics_engine = UnifiedPhonicsEngine(max_word_length=20)
narrator_state = SyncedNarratorState(engine_ref=phonics_engine, speech_rate=120)

def narration_worker():
    current_meter_len = 1
    max_meter_len = 6

    while True:
        super_groups = narrator_state.super_engine.build_super_groups_for_length(current_meter_len)
        
        if not super_groups:
            current_meter_len = (current_meter_len % max_meter_len) + 1
            continue

        next_super_group = random.choice(super_groups)
        current_meter_len = (current_meter_len % max_meter_len) + 1

        narrator_state.narrate_super_group(next_super_group)


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

if __name__ == "__main__":
    t = threading.Thread(target=narration_worker, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5003, debug=False)
