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
nltk.download('wordnet', quiet=False)
nltk.download('words', quiet=False)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def is_palindrome(s):
    return s == s[::-1]

# ==========================================
# Metrical Foot Prosodic Dictionary
# ==========================================
# Mapping binary/numeric stress strings ('1' = stressed, '0'/'2' = unstressed)
# to classical Greek/Latin metrical feet.
METRICAL_FEET = {
    # --- Disyllables (2 Syllables) ---
    "01": "Iamb",
    "10": "Trochee",
    "11": "Spondee",
    "00": "Pyrrhic",

    # --- Trisyllables (3 Syllables) ---
    "100": "Dactyl",
    "001": "Anapest",
    "010": "Amphibrach",
    "101": "Amphimacer (Cretic)",
    "110": "Antibacchius",
    "011": "Bacchius",
    "111": "Molossus",
    "000": "Tribrach",

    # --- Tetrasyllables (4 Syllables) ---
    "1000": "Primus Paeon",
    "0100": "Secundus Paeon",
    "0010": "Tertius Paeon",
    "0001": "Quartus Paeon",
    "1100": "Major Ionic (Double Trochee)",
    "0011": "Minor Ionic (Double Iamb)",
    "1001": "Choriamb",
    "0110": "Antispast",
    "1010": "Ditrochee",
    "0101": "Diiamb",
    "1110": "Epitrite I",
    "1101": "Epitrite II",
    "1011": "Epitrite III",
    "0111": "Epitrite IV",
    "1111": "Dispondee",
    "0000": "Proceleusmatic",

    # --- Common Pentasyllables (5 Syllables) ---
    "01010": "Pentameter Iambic Catalectic",
    "10101": "Pentameter Trochaic Catalectic",
    "100100": "Hexapody Dactylic Catalectic"
}

def identify_metrical_foot(stress_pattern):
    """Normalize stress string ('0', '1', '2') to binary ('0', '1') and lookup foot name."""
    # Convert secondary stress ('2') to unstressed ('0') for foot matching
    normalized = "".join(['1' if c == '1' else '0' for c in stress_pattern])
    return METRICAL_FEET.get(normalized, None)

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

#    def generate_rhyme_groups(self, max_syllables=8, min_rhymes=3):
#        groups = []
#        for stress in sorted(self.rhyme_matrix.keys(), key=lambda s: (len(s), s)):
#            if len(stress) > max_syllables:
#                continue
#            for tail, word_list in self.rhyme_matrix[stress].items():
#                if len(word_list) >= min_rhymes:
#                    label = f"Stress {stress}, Tail {tail}"
#                    groups.append((label, word_list))
#        random.shuffle(groups)
#        return groups
    def generate_rhyme_groups(self, max_syllables=20, min_rhymes=3):
        groups = []
        for stress in sorted(self.rhyme_matrix.keys(), key=lambda s: (len(s), s)):
            if len(stress) > max_syllables:
                continue
            for tail, word_list in self.rhyme_matrix[stress].items():
                if len(word_list) >= min_rhymes:
                    label = f"Stress {stress}, Tail {tail}"
                    groups.append({
                        "label": label,
                        "stress": stress,
                        "tail": tail,
                        "words": word_list
                    })
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

    #def spell_to_morse_wav(self, word, filename, dot_ms=60, silence_ms=800):
    def spell_to_morse_wav(self, word, filename, dot_ms=100, silence_ms=1000):
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
import asyncio
import websockets

def note_name_to_freq(note_str, a4_freq=432.0):
    """Converts a note string like 'C4', 'F#5' to Hz based on standard equal temperament."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # Normalize flat notes to sharp equivalents
    flats = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}

    name = note_str[:-1]
    octave = int(note_str[-1])
    if name in flats:
        name = flats[name]

    semitones_from_c0 = note_names.index(name) + (octave + 1) * 12
    a4_midi = 69
    semitones_from_a4 = semitones_from_c0 - a4_midi
    return a4_freq * (2.0 ** (semitones_from_a4 / 12.0))

def get_nearest_chord_tone_freq(chord_notes, target_freq=432.0, a4_freq=432.0):
    """Finds the pitch in chord_notes closest in Hz to target_freq."""
    if not chord_notes:
        return target_freq

    freqs = [note_name_to_freq(n, a4_freq) for n in chord_notes]
    return min(freqs, key=lambda f: abs(f - target_freq))

STATE_FILE = os.path.join(os.path.dirname(__file__), "narrator_state.json")

class SyncedNarratorState:
    #def __init__(self, engine_ref, speech_rate=120, ollama_url="http://127.0.0.1:11434", model_name="qwen3", ws_url="ws://127.0.0.1:65432"):
    def __init__(self, engine_ref, speech_rate=120, ollama_url="http://127.0.0.1:11435", model_name="qwen3", ws_url="ws://127.0.0.1:65432"):
        self.phonics_engine = engine_ref
        self.morse_gen = MorseAudioGenerator(freq=432)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', speech_rate)

        self.tts_engine.setProperty('voice', 'en-us')
        
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.valid_english = set(words.words())
        self.pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 's': 'adjective', 'r': 'adverb'}
        
        # Chord Sync Properties
        self.ws_url = ws_url
        self.current_chord = []
        self.a4_freq = 432.0
        
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

        # Start WebSocket Client Thread
        threading.Thread(target=self._start_ws_client, daemon=True).start()

    def _load_persisted_state(self):
        """Loads state from JSON file if available."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    saved_data = json.load(f)
                    self.current_state.update(saved_data.get("current_state", {}))
                    print(f"Loaded previous state. Resuming at word: '{self.current_state.get('current_word')}' in group: '{self.current_state.get('group_label')}'")
            except Exception as e:
                print(f"Error loading state file: {e}")

    def _save_persisted_state(self):
        """Saves current state to JSON file."""
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({"current_state": self.current_state}, f, indent=2)
        except Exception as e:
            print(f"Error saving state file: {e}")

    def _start_ws_client(self):
        """Runs an async loop inside a background thread to stay connected to chimes.py WS."""
        async def listen():
            while True:
                try:
                    async with websockets.connect(self.ws_url) as ws:
                        while True:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            with self.lock:
                                self.current_chord = data.get("chord", [])
                                self.a4_freq = data.get("a4_freq", 432.0)
                except Exception as e:
                    # Retry connection after a short delay if chimes.py restarts
                    print('start_ws_client', e)
                    await asyncio.sleep(2.0)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(listen())

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
        except Exception as e:
            print('get wav duration', e)
            return 2.5

    def _generate_ollama_example(self, word, definition):
        if not definition:
            print('no definition for word', word)
            return None

        prompt = (
            f"The word is '{word}', "
            f"with definition '{definition}'. "
            f"Use the word '{word}' in a sentence. "
            #f"The example sentence should contain the word '{word}' in precisely that form. "
            f"Output only the sentence."
        )
        
        # Build OpenAI-compatible chat completion payload
        payload = json.dumps({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful dictionary assistant who provides usage examples for words whose examples are missing from the NLTK wordnet. Output only the requested sentence."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 60
        }).encode('utf-8')

        # Endpoint for llama-server OpenAI completion API
        api_url = f"{self.ollama_url.rstrip('/')}/v1/chat/completions"
        
        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer local"  # Matches --api-key local
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                # Extract sentence from OpenAI response schema
                content = res_data["choices"][0]["message"]["content"].strip()
                # Remove extra quotes if model wraps output in quotes
                print('example',content)
                return content.strip('"\'')
        except Exception as e:
            print('generate llama-server example error:', e)
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

#    def get_word_details(self, word): # FIXME multiple definitions
#        synsets = wordnet.synsets(word)
#        homophones = self.phonics_engine.get_homophones(word)
#        
#        if not synsets:
#            example = self._generate_ollama_example(word, None)
#            return {
#                "pos": None, #"word",
#                "definition": None, #f"The word is {word}.",
#                "example": example,
#                "synonyms": [],
#                "antonyms": [],
#                "hypernyms": [],
#                "hyponyms": [],
#                "homophones": homophones
#            }
#
#        syn = synsets[0]
#        pos_full = self.pos_map.get(syn.pos(), 'word')
#        definition = syn.definition()
#        examples = syn.examples()
#        example = examples[0] if examples else self._generate_ollama_example(word, definition)
#
#        synonyms, antonyms, hypernyms, hyponyms = set(), set(), set(), set()
#        for s in synsets:
#            for lemma in s.lemmas():
#                clean_lemma = lemma.name().replace('_', ' ')
#                if clean_lemma.lower() != word.lower():
#                    synonyms.add(clean_lemma)
#                if lemma.antonyms():
#                    for ant in lemma.antonyms():
#                        antonyms.add(ant.name().replace('_', ' '))
#            for hyp in s.hypernyms():
#                for lemma in hyp.lemmas():
#                    hypernyms.add(lemma.name().replace('_', ' '))
#            for hyp in s.hyponyms():
#                for lemma in hyp.lemmas():
#                    hyponyms.add(lemma.name().replace('_', ' '))
#
#        return {
#            "pos": pos_full,
#            "definition": definition,
#            "example": example,
#            "synonyms": list(synonyms),#[:5],
#            "antonyms": list(antonyms),#[:5],
#            "hypernyms": list(hypernyms),#[:5],
#            "hyponyms": list(hyponyms),#[:5],
#            #"homophones": homophones,#[:5]
#        }

    def _broadcast_phrase(self, step_name, text, file_prefix, is_morse=False, word=""):
        filename = f"{file_prefix}.wav"

        if is_morse:
            with self.lock:
                chord = list(self.current_chord)
                a4 = self.a4_freq

            if chord:
                # Set Morse generator frequency to the nearest chord tone
                self.morse_gen.freq = get_nearest_chord_tone_freq(chord, target_freq=432.0, a4_freq=a4)
            else:
                self.morse_gen.freq = 432.0

            self.morse_gen.spell_to_morse_wav(word, filename)
        else:
            self._generate_tts_wav(text, filename)

        duration = self._get_wav_duration(filename)

        with self.lock:
            self.current_state["step"] = step_name
            self.current_state["audio_file"] = filename
            self.current_state["audio_id"] += 1

        # 1. Let the audio duration play through
        time.sleep(duration)

        # 2. Calculate jitter delay to land precisely on the next 1.0s tick grid
        now = time.time()
        remainder = now - int(now)

        sleep_to_next_tick = 1.0 - remainder if remainder > 0 else 0.0

        if sleep_to_next_tick < 0.1:
            sleep_to_next_tick += 1.0

        time.sleep(sleep_to_next_tick)

#    def narrate_group(self, group_label, raw_word_list):
#        group_words = self.clean_and_deduplicate_list(raw_word_list)
#        if not group_words:
#            return
#
#        # Extract stress pattern from group label (e.g. "Stress 10, Tail ...")
#        stress_pattern = ""
#        if "Stress " in group_label:
#            stress_pattern = group_label.split("Stress ")[1].split(",")[0].strip()
#
#        foot_name = identify_metrical_foot(stress_pattern) if stress_pattern else None
#
#        # Build group list narration text with optional foot framing
#        if foot_name:
#            #words_announcement = f"Metrical foot: {foot_name}. Group list: {', '.join(group_words)}. Metrical foot: {foot_name}."
#            words_announcement = f"{foot_name}. Group list: {', '.join(group_words)}. Metrical foot: {foot_name}."
#        else:
#            words_announcement = f"Group list: {', '.join(group_words)}."
#        
#        with self.lock:
#            self.current_state["group_label"] = group_label
#            self.current_state["wordlist"] = group_words
#
#        for word in group_words:
#            details = self.get_word_details(word)
#
#            with self.lock:
#                self.current_state["current_word"] = word
#                self.current_state["metadata"] = details
#
#            self._broadcast_phrase("Announcing Group List", words_announcement, "group_list")
#            self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
#            if foot_name:
#                self._broadcast_phrase("Metrical Foot", f"{foot_name}", "metrical_foot")
#            self._broadcast_phrase("Morse Code Spelling", "", "morse", is_morse=True, word=word)
#            self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
#
#            if details['pos']:
#                self._broadcast_phrase("Part of Speech", f"Part of speech: {details['pos']}.", "pos")
#
#            if details['definition']:
#                self._broadcast_phrase("Definition", f"Definition: {details['definition']}", "def")
#
#            if details['example']:
#                self._broadcast_phrase("Example Sentence", f"Example: {details['example']}", "example")
#
#            if details['synonyms']:
#                self._broadcast_phrase("Synonyms", f"Synonyms: {', '.join(details['synonyms'])}.", "synonyms")
#
#            if details['antonyms']:
#                self._broadcast_phrase("Antonyms", f"Antonyms: {', '.join(details['antonyms'])}.", "antonyms")
#
#            if details['hypernyms']:
#                self._broadcast_phrase("Hypernyms", f"Hypernyms: {', '.join(details['hypernyms'])}.", "hypernyms")
#
#            if details['hyponyms']:
#                self._broadcast_phrase("Hyponyms", f"Hyponyms: {', '.join(details['hyponyms'])}.", "hyponyms")
#
#            self._broadcast_phrase("Repeating Word", f"Word: {word}.", "repeat_word")
#            if foot_name:
#                self._broadcast_phrase("Metrical Foot", f"{foot_name}", "metrical_foot")
#            self._broadcast_phrase("Morse Code Spelling", "", "morse_repeat", is_morse=True, word=word)
#            self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
    def get_word_details(self, word):
        synsets = wordnet.synsets(word)
        homophones = self.phonics_engine.get_homophones(word)
        
        if not synsets:
            example = self._generate_ollama_example(word, None)
            return {
                "senses": [{
                    "pos": None,
                    "definition": None,
                    "example": example,
                    "synonyms": [],
                    "antonyms": [],
                    "hypernyms": [],
                    "hyponyms": []
                }],
                "homophones": homophones
            }

        senses = []
        for syn in synsets:
            pos_full = self.pos_map.get(syn.pos(), 'word')
            definition = syn.definition()
            examples = syn.examples()
            example = examples[0] if examples else self._generate_ollama_example(word, definition)

            synonyms, antonyms, hypernyms, hyponyms = set(), set(), set(), set()
            
            for lemma in syn.lemmas():
                clean_lemma = lemma.name().replace('_', ' ')
                if clean_lemma.lower() != word.lower():
                    synonyms.add(clean_lemma)
                if lemma.antonyms():
                    for ant in lemma.antonyms():
                        antonyms.add(ant.name().replace('_', ' '))
                        
            for hyp in syn.hypernyms():
                for lemma in hyp.lemmas():
                    hypernyms.add(lemma.name().replace('_', ' '))
                    
            for hyp in syn.hyponyms():
                for lemma in hyp.lemmas():
                    hyponyms.add(lemma.name().replace('_', ' '))

            senses.append({
                "pos": pos_full,
                "definition": definition,
                "example": example,
                "synonyms": list(synonyms),
                "antonyms": list(antonyms),
                "hypernyms": list(hypernyms),
                "hyponyms": list(hyponyms)
            })

        return {
            "senses": senses,
            "homophones": homophones
        }

    def narrate_group(self, group_label, raw_word_list):
        group_words = self.clean_and_deduplicate_list(raw_word_list)
        if not group_words:
            return

        stress_pattern = ""
        if "Stress " in group_label:
            stress_pattern = group_label.split("Stress ")[1].split(",")[0].strip()

        foot_name = identify_metrical_foot(stress_pattern) if stress_pattern else None

        if foot_name:
            words_announcement = f"{foot_name}. Group list: {', '.join(group_words)}. Metrical foot: {foot_name}."
        else:
            words_announcement = f"Group list: {', '.join(group_words)}."
        
        with self.lock:
            self.current_state["group_label"] = group_label
            self.current_state["wordlist"] = group_words

            self._save_persisted_state()

        for word in group_words:
            details = self.get_word_details(word)

            with self.lock:
                self.current_state["current_word"] = word
                self.current_state["metadata"] = details

                self._save_persisted_state()

            self._broadcast_phrase("Announcing Group List", words_announcement, "group_list")
            self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
            if foot_name:
                self._broadcast_phrase("Metrical Foot", f"{foot_name}", "metrical_foot")
            self._broadcast_phrase("Morse Code Spelling", "", "morse", is_morse=True, word=word)
            self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
            if is_palindrome(word):
                self._broadcast_phrase("Palindrome Detected", f"Palindrome", "is_palindrome")

            # Iterate over all synset senses for the current word
            senses = details.get("senses", [])
            num_senses = len(senses)

            for idx, sense in enumerate(senses, 1):
                if num_senses > 1:
                    self._broadcast_phrase("Announcing Group List", words_announcement, "group_list")
                    self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
                    if foot_name:
                        self._broadcast_phrase("Metrical Foot", f"{foot_name}", "metrical_foot")
                    self._broadcast_phrase("Morse Code Spelling", "", "morse", is_morse=True, word=word)
                    self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
                    self._broadcast_phrase("Definition Index", f"Definition {idx} of {num_senses}.", f"def_idx_{idx}")

                if sense['pos']:
                    self._broadcast_phrase("Part of Speech", f"Part of speech: {sense['pos']}.", f"pos_{idx}")

                if sense['definition']:
                    self._broadcast_phrase("Definition", f"Definition: {sense['definition']}", f"def_{idx}")

                if sense['example']:
                    self._broadcast_phrase("Example Sentence", f"Example: {sense['example']}", f"example_{idx}")

                if sense['synonyms']:
                    self._broadcast_phrase("Synonyms", f"Synonyms: {', '.join(sense['synonyms'])}.", f"synonyms_{idx}")

                if sense['antonyms']:
                    self._broadcast_phrase("Antonyms", f"Antonyms: {', '.join(sense['antonyms'])}.", f"antonyms_{idx}")

                if sense['hypernyms']:
                    self._broadcast_phrase("Hypernyms", f"Hypernyms: {', '.join(sense['hypernyms'])}.", f"hypernyms_{idx}")

                if sense['hyponyms']:
                    self._broadcast_phrase("Hyponyms", f"Hyponyms: {', '.join(sense['hyponyms'])}.", f"hyponyms_{idx}")

            self._broadcast_phrase("Repeating Word", f"Word: {word}.", "repeat_word")
            if foot_name:
                self._broadcast_phrase("Metrical Foot", f"{foot_name}", "metrical_foot")
            self._broadcast_phrase("Morse Code Spelling", "", "morse_repeat", is_morse=True, word=word)
            self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
            if is_palindrome(word):
                self._broadcast_phrase("Palindrome Detected", f"Palindrome", "is_palindrome")


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
#narrator_state = SyncedNarratorState(engine_ref=phonics_engine, speech_rate=120, model_name="qwen3")
narrator_state = SyncedNarratorState(
    engine_ref=phonics_engine,
    speech_rate=120,
    ollama_url="http://127.0.0.1:11435",
    model_name="Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)

#def narration_worker():
#    #rhyme_groups = phonics_engine.generate_rhyme_groups(max_syllables=8, min_rhymes=3)
#    rhyme_groups = phonics_engine.generate_rhyme_groups(max_syllables=20, min_rhymes=3)
#    group_idx = 0
#
#    while True:
#        next_group = None
#        with narrator_state.lock:
#            if narrator_state.priority_queue:
#                next_group = narrator_state.priority_queue.pop(0)
#
#        if not next_group:
#            if not rhyme_groups:
#                #rhyme_groups = phonics_engine.generate_rhyme_groups(max_syllables=8, min_rhymes=3)
#                rhyme_groups = phonics_engine.generate_rhyme_groups(max_syllables=20, min_rhymes=3)
#                group_idx = 0
#            next_group = rhyme_groups[group_idx % len(rhyme_groups)]
#            group_idx += 1
#
#        label, word_list = next_group
#        narrator_state.narrate_group(label, word_list)
def levenshtein_distance(s1, s2):
    """Calculates edit distance between two stress strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
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

def calculate_group_distance(g1, g2):
    """
    Measures phonetic closeness between two groups.
    A distance of 1 means 1 digit added/removed/swapped in stress, or identical stress with adjacent tail.
    """
    stress_dist = levenshtein_distance(g1["stress"], g2["stress"])
    tail_dist = 0 if g1["tail"] == g2["tail"] else 1
    return stress_dist + tail_dist

def narration_worker():
    all_groups = phonics_engine.generate_rhyme_groups(max_syllables=20, min_rhymes=3)
    if not all_groups:
        return

    visited = set()

    saved_label = narrator_state.current_state.get("group_label")
    current_group = next((g for g in all_groups if g["label"] == saved_label), None)

    if not current_group:
        # Pick a starting group (e.g., shortest stress pattern)
        #current_group = min(all_groups, key=lambda g: len(g["stress"]))
        current_group = max(all_groups, key=lambda g: len(g["stress"]))

    while True:
        # 1. Priority Queue Handling (User manual search override)
        next_group_data = None
        with narrator_state.lock:
            if narrator_state.priority_queue:
                label, word_list = narrator_state.priority_queue.pop(0)
                # Parse priority item back into dict format if needed
                stress = label.split("Stress ")[1].split(",")[0].strip() if "Stress " in label else ""
                tail = label.split("Tail ")[1].strip() if "Tail " in label else ""
                next_group_data = {
                    "label": label,
                    "stress": stress,
                    "tail": tail,
                    "words": word_list
                }

        # 2. Step-wise Traversal if no user priority
        if not next_group_data:
            visited.add(current_group["label"])

            # Reset history if all groups have been explored
            if len(visited) >= len(all_groups):
                visited.clear()

            # Find unvisited neighbors sorted by smallest edit distance
            candidates = [g for g in all_groups if g["label"] not in visited]

            if candidates:
                # Calculate distances from current_group
                scored_candidates = [
                    (calculate_group_distance(current_group, cand), cand)
                    for cand in candidates
                ]

                # Find minimum distance available (ideally dist == 1)
                min_dist = min(dist for dist, _ in scored_candidates)
                best_steps = [cand for dist, cand in scored_candidates if dist == min_dist]

                # Pick randomly among the closest minimal-step neighbors
                current_group = random.choice(best_steps)
            else:
                current_group = random.choice(all_groups)

            next_group_data = current_group

        # 3. Execute Narration
        label = next_group_data["label"]
        word_list = next_group_data["words"]
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
