#!/usr/bin/env python3
import math

# ==============================================================================
# 24-EDO MICROTONAL MUSIC THEORY & RHYTHM ENGINE
# ==============================================================================
A4_FREQ = 432.0                # Master pitch reference
EDO_STEPS = 24
SYSTEM_DRONE_PC = 0
CHORD_DURATION_TICKS = 60      # 60s per chord shift

NOTE_NAMES_24 = [
    'C',  'C𝄳', 'Db', 'D𝄲', 
    'D',  'D𝄳', 'Eb', 'E𝄲', 
    'E',  'E𝄳', 'F',  'F𝄳', 
    'F#', 'G𝄲', 'G',  'G𝄳', 
    'Ab', 'A𝄲', 'A',  'A𝄳', 
    'Bb', 'B𝄲', 'B',  'B𝄳'
]

CHROMATIC_SOLFEGE_MAP_24 = {
    0:  "Do",    1:  "Do+",   2:  "Ra",    3:  "Ra+",   4:  "Re",    5:  "Re+",
    6:  "Me",    7:  "Me+",   8:  "Mi",    9:  "Mi+",   10: "Fa",    11: "Fa+",
    12: "Fi",    13: "Fi+",   14: "So",    15: "So+",   16: "Le",    17: "Le+",
    18: "La",    19: "La+",   20: "Te",    21: "Te+",   22: "Ti",    23: "Ti+"
}

ALL_FAMILIES = [
    "Major", "Harmonic Minor", "Melodic Minor",
    "Harmonic Major", "Double Harmonic Major",
    "Neapolitan Major", "Neapolitan Minor"
]

PARENT_SCALES_24 = {
    "Major":                 [0, 4, 8, 10, 14, 18, 22],
    "Harmonic Minor":        [0, 4, 6, 10, 14, 16, 22],
    "Melodic Minor":         [0, 4, 6, 10, 14, 18, 22],
    "Harmonic Major":        [0, 4, 8, 10, 14, 16, 22],
    "Double Harmonic Major": [0, 2, 8, 10, 14, 16, 22],
    "Neapolitan Major":      [0, 2, 6, 10, 14, 18, 22],
    "Neapolitan Minor":      [0, 2, 6, 10, 14, 16, 22],
}

MODE_ORDERING = {
    "Major":                 [4, 1, 5, 2, 6, 3, 7],
    "Harmonic Minor":        [6, 3, 7, 1, 4, 5, 2],
    "Melodic Minor":         [3, 4, 1, 5, 2, 6, 7],
    "Harmonic Major":        [2, 1, 5, 4, 3, 7, 6],
    "Double Harmonic Major": [4, 1, 5, 2, 6, 3, 7],
    "Neapolitan Major":      [4, 7, 1, 5, 2, 6, 3],
    "Neapolitan Minor":      [4, 7, 1, 5, 2, 3, 6],
}

MODE_NAMES = {
    "Major": ["Ionian", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", "Locrian"],
    "Harmonic Minor": ["Harmonic Minor", "Locrian 6", "Ionian #5", "Dorian #4", "Phrygian Dominant", "Lydian #2", "Super Locrian bb7"],
    "Melodic Minor": ["Melodic Minor", "Dorian b2", "Lydian Augmented", "Lydian Dominant", "Mixolydian b6", "Half-Diminished", "Altered Scale"],
    "Harmonic Major": ["Harmonic Major", "Dorian b5", "Phrygian b4", "Lydian b3", "Mixolydian b2", "Lydian Augmented #2", "Locrian bb7"],
    "Double Harmonic Major": ["Double Harmonic Major", "Lydian #2 #6", "Ultra Phrygian", "Hungarian Minor", "Harmonic Minor b5", "Ionian #2 #5", "Locrian bb3 bb7"],
    "Neapolitan Major": ["Neapolitan Major", "Lydian #6", "Major Augmented #5", "Lydian Dominant b6", "Major Locrian", "Half-Diminished b4", "Altered Dominant bb3"],
    "Neapolitan Minor": ["Neapolitan Minor", "Lydian #6 #3", "Major #5", "Hungarian Gypsy", "Locrian Major", "Ionian #2", "Ultra Locrian"]
}

CHORD_PROGRESSION_ORDER = [0, 3, 4, 5, 2, 1, 6]
FIFTH_STEP_24 = 14

def edo24_to_freq_432(step_val: int) -> float:
    a4_step = 138
    return A4_FREQ * math.pow(2.0, (step_val - a4_step) / 24.0)

def get_fixed_do_solfege_24(step_val: int, drone_pc: int = SYSTEM_DRONE_PC) -> str:
    semitones_above_drone = (step_val - drone_pc) % EDO_STEPS
    return CHROMATIC_SOLFEGE_MAP_24.get(semitones_above_drone, "Do")

def identify_7th_chord_24(formatted_steps: list) -> str:
    root_step = formatted_steps[0]
    root_name = NOTE_NAMES_24[root_step % EDO_STEPS]
    intervals = tuple(sorted((s - root_step) % EDO_STEPS for s in formatted_steps[1:]))

    quality_map = {
        (8, 14, 22):  "Maj7",
        (6, 14, 20):  "m7",
        (8, 14, 20):  "7",
        (6, 12, 20):  "m7b5",
        (6, 12, 18):  "dim7",
        (8, 16, 22):  "Maj7#5",
        (8, 16, 20):  "7#5",
        (6, 14, 22):  "m(Maj7)",
        (8, 12, 20):  "7b5",
        (8, 12, 22):  "Maj7b5",
        (6, 12, 22):  "m(Maj7)b5",
        (10, 14, 20): "7sus4",
        (4, 14, 20):  "7sus2",
    }
    quality = quality_map.get(intervals, "7th Custom (24-EDO)")
    return f"{root_name} {quality}"

def get_parallel_mode_pitches_24(parent_name: str, mode_degree: int, tonic_step: int) -> list:
    scale = PARENT_SCALES_24[parent_name]
    num_notes = len(scale)
    mode_offset = scale[mode_degree - 1]
    mode_indices = [(i + mode_degree - 1) % num_notes for i in range(num_notes)]

    mode_pitches = []
    for idx in mode_indices:
        interval = (scale[idx] - mode_offset) % EDO_STEPS
        mode_pitches.append(tonic_step + interval)

    return mode_pitches

def generate_diatonic_7th_chords_24(
    scale_pitches: list,
    meta: dict,
    tonic_step: int,
    perceived_drone_pc: int,
    octave_offset: int = 0
) -> list:
    chords = []
    num_notes = len(scale_pitches)

    for i in CHORD_PROGRESSION_ORDER:
        chord_steps = [
            scale_pitches[i % num_notes],
            scale_pitches[(i + 2) % num_notes],
            scale_pitches[(i + 4) % num_notes],
            scale_pitches[(i + 6) % num_notes]
        ]

        root_pc = chord_steps[0] % EDO_STEPS
        base_root_step = (120 + (octave_offset * EDO_STEPS)) + root_pc

        formatted_steps = [base_root_step]
        prev_step = base_root_step

        for step_val in chord_steps[1:]:
            pc = step_val % EDO_STEPS
            interval = (pc - root_pc) % EDO_STEPS
            if interval == 0:
                interval = EDO_STEPS
            candidate = base_root_step + interval

            while candidate <= prev_step:
                candidate += EDO_STEPS

            formatted_steps.append(candidate)
            prev_step = candidate

        formatted_notes = []
        fixed_solfege = []

        for s in formatted_steps:
            name = NOTE_NAMES_24[s % EDO_STEPS]
            octave = (s // EDO_STEPS) - 1
            formatted_notes.append(f"{name}{octave}")
            solf = get_fixed_do_solfege_24(s, drone_pc=perceived_drone_pc)
            fixed_solfege.append(solf)

        chord_name = identify_7th_chord_24(formatted_steps)

        chords.append({
            "duration": CHORD_DURATION_TICKS,
            "notes": formatted_notes,
            "steps": formatted_steps,
            "solfege": fixed_solfege,
            "fixed_solfege": fixed_solfege,
            "chord_name": chord_name,
            "meta": meta
        })
    return chords

def generate_parallel_family_block_24(
    family_name: str,
    tonic_step: int,
    perceived_drone_pc: int,
    octave_offset: int = 0
) -> list:
    block = []
    tonic_name = NOTE_NAMES_24[tonic_step % EDO_STEPS]

    for mode_deg in MODE_ORDERING[family_name]:
        pitches = get_parallel_mode_pitches_24(family_name, mode_deg, tonic_step)
        mode_label = MODE_NAMES[family_name][mode_deg - 1]

        scale_solfege = [get_fixed_do_solfege_24(p, drone_pc=perceived_drone_pc) for p in pitches]
        scale_solfege_str = " - ".join(scale_solfege)

        scale_notes = [NOTE_NAMES_24[p % EDO_STEPS] for p in pitches]
        scale_notes_str = " - ".join(scale_notes)

        meta = {
            "key": f"{tonic_name} Parallel {family_name}",
            "mode": f"Mode {mode_deg}: {tonic_name} {mode_label}",
            "tonic_name": tonic_name,
            "tonic_step": tonic_step,
            "scale_pitches": pitches,
            "scale_solfege": scale_solfege_str,
            "scale_notes": scale_notes_str
        }

        block.extend(generate_diatonic_7th_chords_24(
            pitches,
            meta,
            tonic_step=tonic_step,
            perceived_drone_pc=perceived_drone_pc,
            octave_offset=octave_offset
        ))
    return block

def build_descending_circle_of_fifths_progression(
    families: list,
    octave_offset: int = 0
) -> list:
    progression = []
    current_tonic_step = 120
    family_idx = 0
    num_families = len(families)
    total_passes = 84

    for _ in range(total_passes):
        family = families[family_idx % num_families]
        active_tonic_pc = current_tonic_step % EDO_STEPS
        progression.extend(
            generate_parallel_family_block_24(
                family,
                current_tonic_step,
                perceived_drone_pc=active_tonic_pc,
                octave_offset=octave_offset
            )
        )
        current_tonic_step = (current_tonic_step - FIFTH_STEP_24) % (EDO_STEPS * 10)
        family_idx += 1

    return progression

def compute_tone_rhythms(minute_tick: int, is_7th_allowed: bool) -> list:
    return [
        {
            "tone": "Root",
            "interval_beats": 3,
            "active": (minute_tick % 3 == 0)
        },
        {
            "tone": "3rd",
            "interval_beats": 4,
            "active": (minute_tick % 4 == 0)
        },
        {
            "tone": "5th",
            "interval_beats": 5,
            "active": (minute_tick % 5 == 0)
        },
        {
            "tone": "7th",
            "interval_beats": "polygon",
            "active": is_7th_allowed
        }
    ]

def compute_tone_rhythms_rh(minute_tick: int, is_7th_allowed: bool) -> list:
    return [
        {
            "tone": "Root",
            "interval_beats": 3,
            "active": ((minute_tick + 30) % 3 == 0)
        },
        {
            "tone": "3rd",
            "interval_beats": 4,
            "active": ((minute_tick + 30) % 4 == 0)
        },
        {
            "tone": "5th",
            "interval_beats": 5,
            "active": ((minute_tick + 30) % 5 == 0)
        },
        {
            "tone": "7th",
            "interval_beats": "polygon",
            "active": is_7th_allowed
        }
    ]

#!/usr/bin/env python3
import math

class EDOEngine:
    def __init__(self, edo_steps: int = 31, a4_freq: float = 432.0):
        self.edo_steps = edo_steps
        self.a4_freq = a4_freq
        
        # Perfect 5th generator in N-EDO
        self.fifth_step = round(self.edo_steps * math.log2(1.5))
        
        # Diatonic scale via chain of 5ths (F, C, G, D, A, E, B) from C=0
        raw_diatonic = [
            (-self.fifth_step) % self.edo_steps,    # F (degree 4)
            0,                                       # C (degree 1)
            self.fifth_step % self.edo_steps,       # G (degree 5)
            (2 * self.fifth_step) % self.edo_steps, # D (degree 2)
            (3 * self.fifth_step) % self.edo_steps, # A (degree 6)
            (4 * self.fifth_step) % self.edo_steps, # E (degree 3)
            (5 * self.fifth_step) % self.edo_steps  # B (degree 7)
        ]
        self.major_scale = sorted(raw_diatonic)
        
        # Diatonic step sizes & chromatic alteration delta
        self.W = self.major_scale[1] - self.major_scale[0]  # Major second (C->D)
        self.H = self.major_scale[3] - self.major_scale[2]  # Diatonic semitone (E->F)
        self.delta = self.W - self.H                       # Chromatic semitone (flat/sharp)
        
        # Dynamic Interval Definitions
        self.m3 = self.major_scale[2] - self.delta
        self.M3 = self.major_scale[2]
        self.P4 = self.major_scale[3]
        self.d5 = self.major_scale[4] - self.delta
        self.P5 = self.major_scale[4]
        self.a5 = self.major_scale[4] + self.delta
        self.m7 = self.major_scale[6] - self.delta
        self.M7 = self.major_scale[6]
        self.d7 = self.major_scale[6] - (2 * self.delta)
        
        # Parent Scale Families parameterized for N-EDO
        s = self.major_scale
        d = self.delta
        self.parent_scales = {
            "Major":                 [s[0], s[1],     s[2],     s[3], s[4], s[5],     s[6]],
            "Harmonic Minor":        [s[0], s[1],     s[2] - d, s[3], s[4], s[5] - d, s[6]],
            "Melodic Minor":         [s[0], s[1],     s[2] - d, s[3], s[4], s[5],     s[6]],
            "Harmonic Major":        [s[0], s[1],     s[2],     s[3], s[4], s[5] - d, s[6]],
            "Double Harmonic Major": [s[0], s[1] - d, s[2],     s[3], s[4], s[5] - d, s[6]],
            "Neapolitan Major":      [s[0], s[1] - d, s[2],     s[3], s[4], s[5],     s[6]],
            "Neapolitan Minor":      [s[0], s[1] - d, s[2] - d, s[3], s[4], s[5] - d, s[6]],
        }
        
        # Master Pitch Reference: Anchor A4 (4 octaves + A step)
        a_pc = (3 * self.fifth_step) % self.edo_steps
        self.a4_step = (4 * self.edo_steps) + a_pc

    def edo_to_freq(self, step_val: int) -> float:
        return self.a4_freq * math.pow(2.0, (step_val - self.a4_step) / float(self.edo_steps))

    def get_note_name(self, step_val: int) -> str:
        pc = step_val % self.edo_steps
        octave = (step_val // self.edo_steps) - 1
        
        if self.edo_steps == 12:
            names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            return f"{names[pc]}{octave}"
        elif self.edo_steps == 24:
            names = ['C', 'C𝄳', 'Db', 'D𝄲', 'D', 'D𝄳', 'Eb', 'E𝄲', 'E', 'E𝄳', 'F', 'F𝄳',
                     'F#', 'G𝄲', 'G', 'G𝄳', 'Ab', 'A𝄲', 'A', 'A𝄳', 'Bb', 'B𝄲', 'B', 'B𝄳']
            return f"{names[pc]}{octave}"
        else:
            deg_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
            closest_deg = min(range(7), key=lambda i: min((pc - self.major_scale[i]) % self.edo_steps, (self.major_scale[i] - pc) % self.edo_steps))
            diff = (pc - self.major_scale[closest_deg]) % self.edo_steps
            if diff > self.edo_steps // 2:
                diff -= self.edo_steps
            acc = f"+{diff}" if diff > 0 else (f"{diff}" if diff < 0 else "")
            return f"{deg_names[closest_deg]}{acc}_{octave}"

    def get_solfege(self, step_val: int, drone_pc: int = 0) -> str:
        pc = (step_val - drone_pc) % self.edo_steps
        solfege_base = ["Do", "Re", "Mi", "Fa", "So", "La", "Ti"]
        for idx, deg_pc in enumerate(self.major_scale):
            if pc == deg_pc:
                return solfege_base[idx]
        closest = min(range(7), key=lambda i: (pc - self.major_scale[i]) % self.edo_steps)
        offset = (pc - self.major_scale[closest]) % self.edo_steps
        return f"{solfege_base[closest]}+{offset}" if offset > 0 else solfege_base[closest]

    def identify_7th_chord(self, formatted_steps: list) -> str:
        root_step = formatted_steps[0]
        root_name = self.get_note_name(root_step)
        intervals = sorted(((s - root_step) % self.edo_steps) for s in formatted_steps[1:])
        i3, i5, i7 = intervals[0], intervals[1], intervals[2]
        
        if i3 == self.M3 and i5 == self.P5 and i7 == self.M7:
            quality = "Maj7"
        elif i3 == self.m3 and i5 == self.P5 and i7 == self.m7:
            quality = "m7"
        elif i3 == self.M3 and i5 == self.P5 and i7 == self.m7:
            quality = "7"
        elif i3 == self.m3 and i5 == self.d5 and i7 == self.m7:
            quality = "m7b5"
        elif i3 == self.m3 and i5 == self.d5 and i7 == self.d7:
            quality = "dim7"
        elif i3 == self.M3 and i5 == self.a5 and i7 == self.M7:
            quality = "Maj7#5"
        elif i3 == self.M3 and i5 == self.a5 and i7 == self.m7:
            quality = "7#5"
        elif i3 == self.m3 and i5 == self.P5 and i7 == self.M7:
            quality = "m(Maj7)"
        else:
            quality = f"7th Custom ({self.edo_steps}-EDO)"
            
        return f"{root_name} {quality}"

    def get_parallel_mode_pitches(self, parent_name: str, mode_degree: int, tonic_step: int) -> list:
        scale = self.parent_scales[parent_name]
        num_notes = len(scale)
        mode_offset = scale[mode_degree - 1]
        mode_indices = [(i + mode_degree - 1) % num_notes for i in range(num_notes)]
        return [tonic_step + ((scale[idx] - mode_offset) % self.edo_steps) for idx in mode_indices]

    def generate_diatonic_7th_chords(self, scale_pitches: list, meta: dict, tonic_step: int, perceived_drone_pc: int, octave_offset: int = 0) -> list:
        chords = []
        num_notes = len(scale_pitches)
        progression_order = [0, 3, 4, 5, 2, 1, 6]
        
        for i in progression_order:
            chord_steps = [
                scale_pitches[i % num_notes],
                scale_pitches[(i + 2) % num_notes],
                scale_pitches[(i + 4) % num_notes],
                scale_pitches[(i + 6) % num_notes]
            ]
            root_pc = chord_steps[0] % self.edo_steps
            base_root_step = (5 * self.edo_steps) + (octave_offset * self.edo_steps) + root_pc
            
            formatted_steps = [base_root_step]
            prev_step = base_root_step
            
            for step_val in chord_steps[1:]:
                pc = step_val % self.edo_steps
                interval = (pc - root_pc) % self.edo_steps
                if interval == 0:
                    interval = self.edo_steps
                candidate = base_root_step + interval
                while candidate <= prev_step:
                    candidate += self.edo_steps
                formatted_steps.append(candidate)
                prev_step = candidate

            formatted_notes = [self.get_note_name(s) for s in formatted_steps]
            fixed_solfege = [self.get_solfege(s, drone_pc=perceived_drone_pc) for s in formatted_steps]
            chord_name = self.identify_7th_chord(formatted_steps)

            chords.append({
                "duration": 60,
                "notes": formatted_notes,
                "steps": formatted_steps,
                "solfege": fixed_solfege,
                "fixed_solfege": fixed_solfege,
                "chord_name": chord_name,
                "meta": meta
            })
        return chords

    def generate_drones_for_chord(self, chord_data: dict, perceived_drone_pc: int = 0) -> dict:
        key_tonic_step = chord_data["meta"].get("tonic_step", 5 * self.edo_steps)
        chord_root_step = chord_data["steps"][0]

        key_pc = key_tonic_step % self.edo_steps
        chord_pc = chord_root_step % self.edo_steps

        step_key_oct0 = (self.edo_steps * 1) + key_pc        # Octave 0
        step_key_oct1 = (self.edo_steps * 2) + key_pc        # Octave 1
        step_chord_oct2 = (self.edo_steps * 3) + chord_pc    # Octave 2

        return {
            "key_drone_low": {
                "note": self.get_note_name(step_key_oct0),
                "step": step_key_oct0,
                "frequency": self.edo_to_freq(step_key_oct0),
                "solfege": self.get_solfege(step_key_oct0, perceived_drone_pc)
            },
            "key_drone_high": {
                "note": self.get_note_name(step_key_oct1),
                "step": step_key_oct1,
                "frequency": self.edo_to_freq(step_key_oct1),
                "solfege": self.get_solfege(step_key_oct1, perceived_drone_pc)
            },
            "chord_drone": {
                "note": self.get_note_name(step_chord_oct2),
                "step": step_chord_oct2,
                "frequency": self.edo_to_freq(step_chord_oct2),
                "solfege": self.get_solfege(step_chord_oct2, perceived_drone_pc)
            }
        }
