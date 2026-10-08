import os
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Ensure pyttsx3 does not attempt to initialize audio hardware during test collection
with patch("pyttsx3.init"):
    from juniper_dictionary.dictionary import (
        MorseAudioGenerator,
        SyncedNarratorState,
        UnifiedPhonicsEngine,
        calculate_group_distance,
        get_nearest_chord_tone_freq,
        identify_metrical_foot,
        is_palindrome,
        levenshtein_distance,
    )


# ==========================================
# 1. Pure Helper Function Tests
# ==========================================

class TestHelperFunctions:
    def test_is_palindrome(self):
        assert is_palindrome("racecar") is True
        assert is_palindrome("kayak") is True
        assert is_palindrome("juniper") is False
        assert is_palindrome("") is True

#    def test_identify_metrical_foot(self):
#        assert identify_metrical_foot("01") == "Iamb"
#        assert identify_metrical_foot("10") == "Trochee"
#        assert identify_metrical_foot("11") == "Spondee"
#        assert identify_metrical_foot("00") == "Pyrrhic"
#        assert identify_metrical_foot("100") == "Dactyl"
#        assert identify_metrical_foot("001") == "Anapest"
#        assert identify_metrical_foot("0010") == "Tertius Paeon"
#        #assert identify_metrical_foot("9999") == "Pyrrhic"  # Non-'1' chars normalize to '0'
#        assert identify_metrical_foot("9999") == "Pyrrhic"
#        assert identify_metrical_foot("99") == "Dispondee"
#        assert identify_metrical_foot("111111") is None

#    # test_dictionary.py
#    def test_identify_metrical_foot(self):
#        assert identify_metrical_foot("01") == "Iamb"
#        assert identify_metrical_foot("10") == "Trochee"
#        assert identify_metrical_foot("11") == "Spondee"
#        assert identify_metrical_foot("00") == "Pyrrhic"
#        assert identify_metrical_foot("100") == "Dactyl"
#        assert identify_metrical_foot("001") == "Anapest"
#        assert identify_metrical_foot("0010") == "Tertius Paeon"
#        assert identify_metrical_foot("99") == "Pyrrhic"  # Non-'1' chars normalize to '0'
#        assert identify_metrical_foot("111111") is None

    def test_identify_metrical_foot(self):
        # Standard foot patterns
        assert identify_metrical_foot("01") == "Iamb"
        assert identify_metrical_foot("10") == "Trochee"
        assert identify_metrical_foot("11") == "Spondee"
        assert identify_metrical_foot("00") == "Pyrrhic"
        assert identify_metrical_foot("100") == "Dactyl"
        assert identify_metrical_foot("001") == "Anapest"
        assert identify_metrical_foot("0010") == "Tertius Paeon"
    
        # Secondary stress ('2') normalizes to primary stress ('1')
        assert identify_metrical_foot("02") == "Iamb"
        assert identify_metrical_foot("20") == "Trochee"
        assert identify_metrical_foot("21") == "Spondee"
    
        # Invalid characters and unlisted patterns return None
        assert identify_metrical_foot("9999") is None
        assert identify_metrical_foot("111111") is None

    def test_levenshtein_distance(self):
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("same", "same") == 0
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("10", "01") == 2

    def test_calculate_group_distance(self):
        g1 = {"stress": "01", "tail": "AE1_T"}
        g2 = {"stress": "01", "tail": "AE1_T"}
        assert calculate_group_distance(g1, g2) == 0

        g3 = {"stress": "01", "tail": "IH1_T"}
        assert calculate_group_distance(g1, g3) == 1  # Same stress, different tail (+1)

        g4 = {"stress": "10", "tail": "AE1_T"}
        assert calculate_group_distance(g1, g4) == 2  # Levenshtein("01", "10") = 2, same tail (+0)

    def test_get_nearest_chord_tone_freq(self):
        assert get_nearest_chord_tone_freq([], target_freq=432.0) == 432.0
        
        freqs = [220.0, 440.0, 880.0]
        # Target 432.0 is closest to 440.0; return value is halved -> 220.0
        assert get_nearest_chord_tone_freq(freqs, target_freq=432.0) == 220.0


# ==========================================
# 2. Phonetic Engine Tests
# ==========================================

class TestUnifiedPhonicsEngine:
    @pytest.fixture
    def engine(self):
        with patch("pronouncing.search", return_value=["cat", "bat", "hat", "rat"]):
            with patch("pronouncing.phones_for_word", side_effect=lambda w: {
                "cat": ["K AE1 T"],
                "bat": ["B AE1 T"],
                "hat": ["HH AE1 T"],
                "rat": ["R AE1 T"]
            }.get(w, [])):
                return UnifiedPhonicsEngine(max_word_length=20)

    def test_clean_word(self, engine):
        assert engine._clean_word("Cat!!") == "cat"
        assert engine._clean_word("  TEST-123  ") == "test"

    def test_extract_phonetic_parts(self, engine):
        parts = engine._extract_phonetic_parts("K AE1 T")
        assert parts["stress"] == "1"
        assert parts["syllables"] == 1
        assert parts["onset"] == "K"
        assert parts["vowel"] == "AE"
        assert parts["rhyme_tail"] == "AE_T"

    def test_get_group_for_word(self, engine):
        label, words = engine.get_group_for_word("cat")
        assert label == "Stress 1, Tail AE_T"
        assert set(words) == {"cat", "bat", "hat", "rat"}

    def test_get_group_for_unknown_word(self, engine):
        label, words = engine.get_group_for_word("nonexistentword123")
        assert label is None
        assert words is None

    def test_generate_rhyme_groups(self, engine):
        groups = engine.generate_rhyme_groups(max_syllables=5, min_rhymes=3)
        assert len(groups) == 1
        assert groups[0]["stress"] == "1"
        assert groups[0]["tail"] == "AE_T"
        assert len(groups[0]["words"]) == 4


# ==========================================
# 3. Morse Audio Generator Tests
# ==========================================

class TestMorseAudioGenerator:
    @pytest.fixture
    def morse_gen(self):
        return MorseAudioGenerator(sample_rate=44100)

    def test_generate_tone(self, morse_gen):
        tone = morse_gen._generate_tone(125)
        assert isinstance(tone, np.ndarray)
        expected_samples = int(44100 * (125 / 1000.0))
        assert len(tone) == expected_samples

    def test_spell_to_morse_wav(self, morse_gen):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("juniper_dictionary.dictionary.CACHE_DIR", tmp_dir):
                output_file = morse_gen.spell_to_morse_wav("SOS", "test_sos.wav", dot_ms=125)
                assert output_file == "test_sos.wav"
                assert os.path.exists(os.path.join(tmp_dir, "test_sos.wav"))


# ==========================================
# 4. Synced Narrator State Tests
# ==========================================

class TestSyncedNarratorState:
    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock(spec=UnifiedPhonicsEngine)
        engine.get_group_for_word.side_effect = lambda w: (
            ("Stress 1, Tail AE_T", ["cat", "bat", "hat"]) if w == "cat" else (None, None)
        )
        engine.word_profiles = {
            "cat": {
                "raw_pronunciation": "K AE1 T",
                "stress": "1",
                "rhyme_tail": "AE_T"
            }
        }
        engine.get_homophones.return_value = []
        return engine

    @pytest.fixture
    def narrator_state(self, mock_engine):
        with patch("pyttsx3.init") as mock_tts_init:
            mock_tts_instance = MagicMock()
            mock_tts_init.return_value = mock_tts_instance
            state = SyncedNarratorState(engine_ref=mock_engine, speech_rate=120)
            return state

    def test_initial_state(self, narrator_state):
        assert narrator_state.current_state["step"] == "Idle"
        assert narrator_state.current_state["current_word"] == ""
        assert narrator_state.current_chimes_state == {}

    def test_update_chimes_state(self, narrator_state):
        payload = {"left_hand": {"key": "C", "mode": "Lydian", "frequencies": [261.63, 329.63]}}
        narrator_state.update_chimes_state(payload)
        assert narrator_state.current_chimes_state == payload

    def test_enqueue_priority_word_success(self, narrator_state):
        success, label = narrator_state.enqueue_priority_word("cat")
        assert success is True
        assert label == "Stress 1, Tail AE_T"
        assert len(narrator_state.priority_queue) == 1
        assert narrator_state.priority_queue[0] == ("Stress 1, Tail AE_T", ["cat", "bat", "hat"])

    def test_enqueue_priority_word_failure(self, narrator_state):
        success, message = narrator_state.enqueue_priority_word("unknownword")
        assert success is False
        assert message == "Word not found in dictionary."
        assert len(narrator_state.priority_queue) == 0

    @patch("juniper_dictionary.dictionary.fetch_microservice_json")
    @patch("juniper_dictionary.dictionary.safe_wordnet_synsets")
    def test_get_word_details(self, mock_synsets, mock_microservice, narrator_state):
        mock_microservice.side_effect = lambda url: (
            {"example": "The cat sat on the mat."} if "example" in url else {"palindrome": None}
        )
        
        mock_synset = MagicMock()
        mock_synset.pos.return_value = "n"
        mock_synset.definition.return_value = "A small domesticated carnivorous mammal."
        mock_synset.examples.return_value = []
        
        lemma_cat = MagicMock()
        lemma_cat.name.return_value = "cat"
        lemma_feline = MagicMock()
        lemma_feline.name.return_value = "feline"
        lemma_feline.antonyms.return_value = []
        mock_synset.lemmas.return_value = [lemma_cat, lemma_feline]
        
        mock_synset.hyponyms.return_value = []
        mock_synset.hypernyms.return_value = []
        mock_synsets.return_value = [mock_synset]

        details = narrator_state.get_word_details("cat")
        
        assert details["pronunciation"] == "K AE1 T"
        assert details["stress_pattern"] == "1"
        assert details["rhyme_tail"] == "AE_T"
        assert len(details["senses"]) == 1
        assert details["senses"][0]["pos"] == "n"
        assert details["senses"][0]["definition"] == "A small domesticated carnivorous mammal."
        assert "feline" in details["senses"][0]["synonyms"]
        assert "cat" not in details["senses"][0]["synonyms"]
