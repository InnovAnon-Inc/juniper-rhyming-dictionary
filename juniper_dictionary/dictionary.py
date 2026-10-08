import os
import time
import re
import wave
import json
import random
import threading
import urllib.request
from collections import defaultdict
import numpy as np
from scipy.io import wavfile
import pyttsx3
import nltk
from nltk.corpus import wordnet
import pronouncing

nltk.download('wordnet', quiet=True)
nltk_lock = threading.Lock()

def safe_wordnet_synsets(word):
    with nltk_lock:
        return wordnet.synsets(word)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def is_palindrome(s):
    return s == s[::-1]

def fetch_microservice_json(url, timeout=1.0):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Juniper-Dict/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        pass
    return {}

# ==========================================
# Metrical Foot Prosodic Dictionary
# ==========================================
METRICAL_FEET = {
    "01": "Iamb", "10": "Trochee", "11": "Spondee", "00": "Pyrrhic",
    "100": "Dactyl", "001": "Anapest", "010": "Amphibrach", "101": "Amphimacer",
    "110": "Antibacchius", "011": "Bacchius", "111": "Molossus", "000": "Tribrach",
    "1000": "Primus Paeon", "0100": "Secundus Paeon", "0010": "Tertius Paeon", "0001": "Quartus Paeon",
    "1100": "Major Ionic", "0011": "Minor Ionic", "1001": "Choriamb", "0110": "Antispast",
    "1010": "Ditrochee", "0101": "Diiamb"
}

def identify_metrical_foot(stress_pattern):
    normalized = "".join(['1' if c == '1' else '0' for c in stress_pattern])
    return METRICAL_FEET.get(normalized, None)

# ==========================================
# 1. Phonetic Rhyme Engine
# ==========================================
class UnifiedPhonicsEngine:
    def __init__(self, max_word_length=20):
        self.max_word_length = max_word_length
        self.vowel_phonemes = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
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
                    if '1' in t: break

        if stressed_idx == -1: return None

        rhyme_tokens = ["".join([c for c in t if not c.isdigit()]) for t in tokens[stressed_idx:]]
        return {
            "stress": stresses,
            "syllables": len(stresses),
            "onset": onset,
            "vowel": primary_vowel,
            "rhyme_tail": "_".join(rhyme_tokens),
            "raw_pronunciation": phones_str
        }

    def _build_indices(self):
        all_words = pronouncing.search(".*")
        for word in all_words:
            clean = self._clean_word(word)
            if not clean or len(clean) > self.max_word_length or clean in self.word_profiles:
                continue
            phones_list = pronouncing.phones_for_word(clean)
            if not phones_list: continue

            parts = self._extract_phonetic_parts(phones_list[0])
            if not parts: continue

            self.word_profiles[clean] = parts
            self.rhyme_matrix[parts["stress"]][parts["rhyme_tail"]].append(clean)
            self.phone_to_words[re.sub(r'\d+', '', phones_list[0])].append(clean)

    def get_homophones(self, target_word):
        clean = self._clean_word(target_word)
        phones_list = pronouncing.phones_for_word(clean)
        if not phones_list: return []
        matches = self.phone_to_words.get(re.sub(r'\d+', '', phones_list[0]), [])
        return [w for w in matches if w != clean]

    def get_group_for_word(self, target_word):
        clean = self._clean_word(target_word)
        profile = self.word_profiles.get(clean)
        if profile:
            stress, tail = profile["stress"], profile["rhyme_tail"]
            return f"Stress {stress}, Tail {tail}", self.rhyme_matrix[stress].get(tail, [clean])
        return None, None

    def generate_rhyme_groups(self, max_syllables=20, min_rhymes=3):
        groups = []
        for stress in sorted(self.rhyme_matrix.keys(), key=lambda s: (len(s), s)):
            if len(stress) > max_syllables: continue
            for tail, word_list in self.rhyme_matrix[stress].items():
                if len(word_list) >= min_rhymes:
                    groups.append({
                        "label": f"Stress {stress}, Tail {tail}",
                        "stress": stress,
                        "tail": tail,
                        "words": word_list
                    })
        return groups

# ==========================================
# 2. Morse Audio Generator (125ms 16th notes)
# ==========================================
class MorseAudioGenerator:
    def __init__(self, sample_rate=44100):
        self.freq = 216.0 
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

    def spell_to_morse_wav(self, word, filename, dot_ms=125, silence_ms=1000):
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
            wavfile.write(os.path.join(CACHE_DIR, filename), self.sample_rate, scaled)
            return filename
        return None

# ==========================================
# 3. Synchronized Audio State Manager
# ==========================================
#def get_nearest_chord_tone_freq(freqs, target_freq=216.0):
def get_nearest_chord_tone_freq(freqs, target_freq=432.0):
    if not freqs: return target_freq
    # Compare against frequencies dropped by an octave to find the nearest match in the lower register
    #lower_octave_freqs = [f / 2.0 for f in freqs]
    #return min(lower_octave_freqs, key=lambda f: abs(f - target_freq))
    lower_octave_freqs = [f for f in freqs]
    result = min(lower_octave_freqs, key=lambda f: abs(f - target_freq))
    return result / 2

class SyncedNarratorState:
    def __init__(self, engine_ref, speech_rate=120):
        self.phonics_engine = engine_ref
        self.morse_gen = MorseAudioGenerator()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', speech_rate)
        
        self.current_chimes_state = {}
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

    def update_chimes_state(self, payload):
        with self.lock:
            self.current_chimes_state = payload

    def _pad_wav_with_silence(self, filepath, silence_duration_sec=1.0):
        try:
            sr, data = wavfile.read(filepath)
            silence_samples = int(sr * silence_duration_sec)
            silence = np.zeros(silence_samples, dtype=data.dtype) if data.ndim == 1 else np.zeros((silence_samples, data.shape[1]), dtype=data.dtype)
            wavfile.write(filepath, sr, np.concatenate((data, silence)))
        except Exception:
            pass

    def _generate_tts_wav(self, text, filename):
        filepath = os.path.join(CACHE_DIR, filename)
        self.tts_engine.save_to_file(text, filepath)
        self.tts_engine.runAndWait()
        self._pad_wav_with_silence(filepath, silence_duration_sec=0.8)

    def _get_wav_duration(self, filename):
        try:
            with wave.open(os.path.join(CACHE_DIR, filename), 'r') as f:
                return f.getnframes() / float(f.getframerate())
        except Exception:
            return 2.5

    def _broadcast_phrase(self, step_name, text, file_prefix, is_morse=False, word=""):
        filename = f"{file_prefix}.wav"

        if is_morse:
            with self.lock:
                freqs = self.current_chimes_state.get("left_hand", {}).get("frequencies", [])
            self.morse_gen.freq = get_nearest_chord_tone_freq(freqs, target_freq=216.0)
            self.morse_gen.spell_to_morse_wav(word, filename, dot_ms=125) # strictly 125ms dits
        else:
            self._generate_tts_wav(text, filename)

        duration = self._get_wav_duration(filename)

        with self.lock:
            self.current_state["step"] = step_name
            self.current_state["audio_file"] = filename
            self.current_state["audio_id"] += 1

        time.sleep(duration)

        # Jitter sleep to align perfectly to the 125ms (16th note) grid for 60/120bpm sync
        now = time.time()
        remainder = now % 0.125
        sleep_to_next_tick = 0.125 - remainder if remainder > 0 else 0.0
        if sleep_to_next_tick < 0.02: 
            sleep_to_next_tick += 0.125
        time.sleep(sleep_to_next_tick)

    def get_word_details(self, word):
        synsets = safe_wordnet_synsets(word)
        homophones = self.phonics_engine.get_homophones(word)
        profile = self.phonics_engine.word_profiles.get(word, {})

        ex_data = fetch_microservice_json(f"http://127.0.0.1:5018/example/{word}")
        pal_data = fetch_microservice_json(f"http://127.0.0.1:5010/api/palindrome/{word}")

        example = ex_data.get("example")
        palindrome = pal_data.get("palindrome")

        senses = []
        for syn in synsets:
#            senses.append({
#                "pos": syn.pos(),
#                "definition": syn.definition(),
#                "example": example or (syn.examples()[0] if syn.examples() else None),
#                "synonyms": [l.name().replace('_', ' ') for l in syn.lemmas() if l.name().lower() != word],
#                "antonyms": [a.name().replace('_', ' ') for l in syn.lemmas() for a in l.antonyms()]
#            })
            senses.append({
    "pos": syn.pos(),
    "definition": syn.definition(),
    "example": example or (syn.examples()[0] if syn.examples() else None),
    "synonyms": list(set(l.name().replace('_', ' ') for l in syn.lemmas() if l.name().lower() != word)),
    "antonyms": list(set(a.name().replace('_', ' ') for l in syn.lemmas() for a in l.antonyms())),
    "hyponyms": list(set(h.name().split('.')[0].replace('_', ' ') for h in syn.hyponyms())),
    "hypernyms": list(set(h.name().split('.')[0].replace('_', ' ') for h in syn.hypernyms()))
})

        if not senses:
            senses = [{"pos": None, "definition": None, "example": example, "synonyms": [], "antonyms": []}]

        return {
            "senses": senses,
            "homophones": homophones,
            "palindrome": palindrome,
            "pronunciation": profile.get("raw_pronunciation", ""),
            "stress_pattern": profile.get("stress", ""),
            "rhyme_tail": profile.get("rhyme_tail", "")
        }

    def spelling_bee(self, word, foot_name):
        self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
        if foot_name:
            self._broadcast_phrase("Metrical Foot", f"{foot_name}", "metrical_foot")
        self._broadcast_phrase("Morse Code Spelling", "", "morse", is_morse=True, word=word)
        self._broadcast_phrase("Saying Word", f"Word: {word}.", "say_word")
        if is_palindrome(word):
            self._broadcast_phrase("Palindrome Detected", "Palindrome", "is_palindrome")

    def narrate_group(self, group_label, raw_word_list):
        group_words = list(dict.fromkeys([w.lower().strip() for w in raw_word_list if safe_wordnet_synsets(w.lower().strip())]))
        if not group_words: return

        stress_pattern = group_label.split("Stress ")[1].split(",")[0].strip() if "Stress " in group_label else ""
        foot_name = identify_metrical_foot(stress_pattern)

        announcement = f"{foot_name}. Group list: {', '.join(group_words)}." if foot_name else f"Group list: {', '.join(group_words)}."
        
        with self.lock:
            self.current_state["group_label"] = group_label
            self.current_state["wordlist"] = group_words

        for word in group_words:
            details = self.get_word_details(word)
            with self.lock:
                self.current_state["current_word"] = word
                self.current_state["metadata"] = details

            self._broadcast_phrase("Announcing Group List", announcement, "group_list")
            self.spelling_bee(word, foot_name)

            for idx, sense in enumerate(details["senses"], 1):
                if len(details["senses"]) > 1:
                    self.spelling_bee(word, foot_name)
                    self._broadcast_phrase("Definition Index", f"Definition {idx}.", f"def_idx_{idx}")

                if details.get("palindrome"):
                    self._broadcast_phrase("Saying Pseudo-palindrome", f"Example: {details['palindrome']}.", "say_palindrome")
                    self._broadcast_phrase("Pseudo-palindromic example", "", "morse_repeat", is_morse=True, word=details['palindrome'])
                    self._broadcast_phrase("Saying Pseudo-palindrome", f"{details['palindrome']}", "say_palindrome")

#                if sense['pos']: self._broadcast_phrase("Part of Speech", f"Part of speech: {sense['pos']}.", f"pos_{idx}")
#                if sense['definition']: self._broadcast_phrase("Definition", f"Definition: {sense['definition']}", f"def_{idx}")
#                if sense['example']: self._broadcast_phrase("Example Sentence", f"Example: {sense['example']}", f"example_{idx}")
#                if sense['synonyms']: self._broadcast_phrase("Synonyms", f"Synonyms: {', '.join(sense['synonyms'])}.", f"synonyms_{idx}")
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
                if sense['hyponyms']: 
                    self._broadcast_phrase("Hyponyms", f"Hyponyms: {', '.join(sense['hyponyms'])}.", f"hyponyms_{idx}")
                if sense['hypernyms']: 
                    self._broadcast_phrase("Hypernyms", f"Hypernyms: {', '.join(sense['hypernyms'])}.", f"hypernyms_{idx}")

            self.spelling_bee(word, foot_name)

    def enqueue_priority_word(self, word):
        label, word_list = self.phonics_engine.get_group_for_word(word)
        if label and word_list:
            with self.lock:
                self.priority_queue.insert(0, (label, word_list))
            return True, label
        return False, "Word not found in dictionary."

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]

def calculate_group_distance(g1, g2):
    return levenshtein_distance(g1["stress"], g2["stress"]) + (0 if g1["tail"] == g2["tail"] else 1)

def narration_worker(narrator_state, phonics_engine):
    all_groups = phonics_engine.generate_rhyme_groups(max_syllables=20, min_rhymes=3)
    if not all_groups: return

    visited = set()
    current_group = max(all_groups, key=lambda g: len(g["stress"]))

    while True:
        next_group_data = None
        with narrator_state.lock:
            if narrator_state.priority_queue:
                label, words = narrator_state.priority_queue.pop(0)
                next_group_data = {"label": label, "stress": label.split("Stress ")[1].split(",")[0].strip(), "tail": label.split("Tail ")[1].strip(), "words": words}

        if not next_group_data:
            visited.add(current_group["label"])
            if len(visited) >= len(all_groups): visited.clear()
            
            candidates = [g for g in all_groups if g["label"] not in visited]
            if candidates:
                scored = [(calculate_group_distance(current_group, cand), cand) for cand in candidates]
                min_dist = min(dist for dist, _ in scored)
                current_group = random.choice([cand for dist, cand in scored if dist == min_dist])
            else:
                current_group = random.choice(all_groups)
            next_group_data = current_group

        narrator_state.narrate_group(next_group_data["label"], next_group_data["words"])
