#!/usr/bin/env python3

import asyncio
import cmath
import http.server
import json
import math
import os
import random
import signal
import socketserver
import sys
import threading
import time
import websockets

# ==========================================
# CONFIGURATION & TUNING
# ==========================================
HTTP_PORT = 5004
WS_PORT = 65432
BPM = 60                       # 1 tick per second
TICK_DURATION = 60.0 / BPM
CHORD_DURATION_TICKS = 60      # Real-time minute (60s) per inner chord change
A4_FREQ = 432.0                # Master Reference Pitch
STATE_FILE = "clock_state.json"

ALL_FAMILIES = [
    "Major", "Harmonic Minor", "Melodic Minor",
    "Harmonic Major", "Double Harmonic Major",
    "Neapolitan Major", "Neapolitan Minor"
]

# Master Drone Pitch Class anchor (0 = C, 7 = G)
SYSTEM_DRONE_PC = 0

CHROMATIC_SOLFEGE_MAP = {
    0: "Do",  1: "Ra",  2: "Re",  3: "Me",
    4: "Mi",  5: "Fa",  6: "Fi",  7: "So",
    8: "Le",  9: "La", 10: "Te", 11: "Ti"
}

NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

PARENT_SCALES = {
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Harmonic Minor": [0, 2, 3, 5, 7, 8, 11],
    "Melodic Minor": [0, 2, 3, 5, 7, 9, 11],
    "Harmonic Major": [0, 2, 4, 5, 7, 8, 11],
    "Double Harmonic Major": [0, 1, 4, 5, 7, 8, 11],
    "Neapolitan Major": [0, 1, 3, 5, 7, 9, 11],
    "Neapolitan Minor": [0, 1, 3, 5, 7, 8, 11],
}

MODE_ORDERING = {
    "Major": [4, 1, 5, 2, 6, 3, 7],
    "Harmonic Minor": [6, 3, 7, 1, 4, 5, 2],
    "Melodic Minor": [3, 4, 1, 5, 2, 6, 7],
    "Harmonic Major": [2, 1, 5, 4, 3, 7, 6],
    "Double Harmonic Major": [4, 1, 5, 2, 6, 3, 7],
    "Neapolitan Major": [4, 7, 1, 5, 2, 6, 3],
    "Neapolitan Minor": [4, 7, 1, 5, 2, 3, 6],
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

# ==========================================
# BJORKLUND EUCLIDEAN RHYTHM ALGORITHM
# ==========================================
def bjorklund(steps: int, pulses: int) -> list[int]:
    if pulses <= 0:
        return [0] * steps
    if pulses >= steps:
        return [1] * steps

    pattern = [[1] for _ in range(pulses)]
    remainder = [[0] for _ in range(steps - pulses)]

    while len(remainder) > 1:
        num_patterns = len(pattern)
        num_remainders = len(remainder)
        count = min(num_patterns, num_remainders)
        for i in range(count):
            pattern[i].extend(remainder.pop(0))

    pattern.extend(remainder)
    return [item for sublist in pattern for item in sublist]

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def is_regular_polygon(pattern: list[int], steps: int) -> bool:
    k = sum(pattern)
    if k < 3 or steps % k != 0:
        return False
    stride = steps // k
    indices = [i for i, b in enumerate(pattern) if b]
    expected = [(indices[0] + j * stride) % steps for j in range(k)]
    return sorted(indices) == sorted(expected)

def is_balanced_cyclotomic(pattern: list[int], steps: int, tol: float = 1e-5) -> bool:
    total_vec = sum(cmath.exp(2j * math.pi * i / steps) for i, b in enumerate(pattern) if b)
    if abs(total_vec) < tol:
        return True

    for k_bin in range(1, steps):
        val = sum(cmath.exp(-2j * math.pi * k_bin * i / steps) for i, b in enumerate(pattern) if b)
        if abs(val) < tol:
            return True
    return False

def get_all_balanced_pulses(steps: int) -> list[int]:
    pulses = []
    for k in range(1, steps):
        pattern = bjorklund(steps, k)
        if gcd(k, steps) == 1:
            pulses.append(k)
        elif is_balanced_cyclotomic(pattern, steps) and not is_regular_polygon(pattern, steps):
            pulses.append(k)
    return pulses

class EuclideanProgression:
    def __init__(self, step_cycles=list(range(3, 33)), repeats_per_rhythm=7, invert_pattern=False):
        self.step_cycles = step_cycles
        self.repeats_per_rhythm = repeats_per_rhythm
        self.invert_pattern = invert_pattern
        self.sequence = []
        for n in self.step_cycles:
            balanced_pulses = get_all_balanced_pulses(n)
            for k in balanced_pulses:
                pattern = bjorklund(n, k)
                if self.invert_pattern:
                    pattern = pattern[::-1]
                self.sequence.append({
                    "steps": n,
                    "pulses": k,
                    "pattern": pattern
                })
        self.seq_idx = 0
        self.current_repeat = 0
        self.step_in_pattern = 0

    def tick(self, offset_ticks: int = 0) -> tuple[bool, int, int, int, list[int]]:
        curr = self.sequence[self.seq_idx]
        pattern = curr["pattern"]

        eval_idx = (self.step_in_pattern + offset_ticks) % len(pattern)
        hit = bool(pattern[eval_idx])
        step_idx = self.step_in_pattern

        self.step_in_pattern += 1
        if self.step_in_pattern >= curr["steps"]:
            self.step_in_pattern = 0
            self.current_repeat += 1
            if self.current_repeat >= self.repeats_per_rhythm:
                self.current_repeat = 0
                self.seq_idx = (self.seq_idx + 1) % len(self.sequence)

        return hit, step_idx, curr["pulses"], curr["steps"], pattern

# ==========================================
# HARMONIC & SOLFEGE ENGINE
# ==========================================
def identify_7th_chord(formatted_midis: list[int]) -> str:
    root_midi = formatted_midis[0]
    root_name = NOTE_NAMES[root_midi % 12]
    intervals = tuple(sorted((m - root_midi) % 12 for m in formatted_midis[1:]))

    quality_map = {
        (4, 7, 11): "Maj7",
        (3, 7, 10): "m7",
        (4, 7, 10): "7",
        (3, 6, 10): "m7b5",
        (3, 6, 9):  "dim7",
        (4, 8, 11): "Maj7#5",
        (4, 8, 10): "7#5",
        (3, 7, 11): "m(Maj7)",
        (4, 6, 10): "7b5",
        (4, 6, 11): "Maj7b5",
        (3, 6, 11): "m(Maj7)b5",
        (5, 7, 10): "7sus4",
        (2, 7, 10): "7sus2",
    }
    
    quality = quality_map.get(intervals, "7th Custom")
    return f"{root_name} {quality}"

def get_fixed_do_syllable(midi_note: int, drone_pc: int = SYSTEM_DRONE_PC) -> str:
    """Returns Fixed-Do solfège syllable relative to the current lowest drone root."""
    semitones_above_drone = (midi_note - drone_pc) % 12
    return CHROMATIC_SOLFEGE_MAP[semitones_above_drone]

def get_parallel_mode_pitches(parent_name: str, mode_degree: int, tonic_midi: int) -> list[int]:
    scale = PARENT_SCALES[parent_name]
    num_notes = len(scale)
    mode_offset = scale[mode_degree - 1]
    mode_indices = [(i + mode_degree - 1) % num_notes for i in range(num_notes)]

    mode_pitches = []
    for idx in mode_indices:
        interval = (scale[idx] - mode_offset) % 12
        mode_pitches.append(tonic_midi + interval)

    return mode_pitches

def get_exact_modal_solfege(parent_name: str, mode_degree: int, tonic_midi: int) -> list[str]:
    """Generates scale solfège anchored strictly to the current drone pitch class."""
    pitches = get_parallel_mode_pitches(parent_name, mode_degree, tonic_midi)
    return [get_fixed_do_syllable(m, drone_pc=tonic_midi % 12) for m in pitches]

def generate_diatonic_7th_chords(scale_pitches: list[int], mode_solfege: list[str], meta: dict, tonic_midi: int, octave_offset: int = 0) -> list[dict]:
    chords = []
    num_notes = len(scale_pitches)
    drone_pc = tonic_midi % 12

    for i in CHORD_PROGRESSION_ORDER:
        chord_midis = [
            scale_pitches[i % num_notes],
            scale_pitches[(i + 2) % num_notes],
            scale_pitches[(i + 4) % num_notes],
            scale_pitches[(i + 6) % num_notes]
        ]

        root_pc = chord_midis[0] % 12
        root_midi = (60 + (octave_offset * 12)) + root_pc

        formatted_midis = [root_midi]
        prev_midi = root_midi

        for midi_val in chord_midis[1:]:
            pc = midi_val % 12
            interval = (pc - root_pc) % 12
            if interval == 0: interval = 12
            candidate = root_midi + interval

            while candidate <= prev_midi:
                candidate += 12

            formatted_midis.append(candidate)
            prev_midi = candidate

        formatted_notes = []
        fixed_solfege = []
        chord_solfege = []

        for m in formatted_midis:
            name = NOTE_NAMES[m % 12]
            octave = (m // 12) - 1
            formatted_notes.append(f"{name}{octave}")

            solf = get_fixed_do_syllable(m, drone_pc=drone_pc)
            fixed_solfege.append(solf)
            chord_solfege.append(solf)

        chord_name = identify_7th_chord(formatted_midis)

        chords.append({
            "duration": CHORD_DURATION_TICKS,
            "notes": formatted_notes,
            "solfege": chord_solfege,
            "fixed_solfege": fixed_solfege,
            "chord_name": chord_name,
            "meta": meta
        })
    return chords

def generate_parallel_family_block(family_name: str, tonic_midi: int, octave_offset: int = 0) -> list[dict]:
    block = []
    tonic_name = NOTE_NAMES[tonic_midi % 12]

    for mode_deg in MODE_ORDERING[family_name]:
        pitches = get_parallel_mode_pitches(family_name, mode_deg, tonic_midi)
        mode_label = MODE_NAMES[family_name][mode_deg - 1]

        mode_solfege = get_exact_modal_solfege(family_name, mode_deg, tonic_midi)
        scale_solfege_str = " - ".join(mode_solfege)

        meta = {
            "key": f"{tonic_name} Parallel {family_name}",
            "mode": f"Mode {mode_deg}: {tonic_name} {mode_label}",
            "tonic_name": tonic_name,
            "scale_solfege": scale_solfege_str
        }

        block.extend(generate_diatonic_7th_chords(
            pitches,
            mode_solfege,
            meta,
            tonic_midi=tonic_midi,
            octave_offset=octave_offset
        ))
    return block

def build_descending_perpetual_progression(families: list[str], octave_offset: int = 0) -> list[dict]:
    progression = []
    current_tonic_midi = 60  # Start at C
    family_idx = 0
    num_families = len(families)

    total_family_passes = 84

    for _ in range(total_family_passes):
        family = families[family_idx % num_families]

        progression.extend(
            generate_parallel_family_block(family, current_tonic_midi, octave_offset=octave_offset)
        )

        current_tonic_midi -= 1
        family_idx += 1

    return progression

# ==========================================
# MASTER CLOCK & STATE MANAGEMENT
# ==========================================
CONNECTED_CLIENTS = set()

class MasterClock:
    def __init__(self, moduli=(3, 4, 5)):
        self.moduli = moduli
        self.inner_family_order = ALL_FAMILIES.copy()
        self.outer_family_order = ALL_FAMILIES.copy()
        self.rebuild_progressions()

        self.inner_euc_engine = EuclideanProgression(step_cycles=list(range(3, 33)), repeats_per_rhythm=7, invert_pattern=False)
        self.outer_euc_engine = EuclideanProgression(step_cycles=list(range(3, 33)), repeats_per_rhythm=7, invert_pattern=True)

    def rebuild_progressions(self):
        self.inner_prog = build_descending_perpetual_progression(self.inner_family_order, octave_offset=0)
        self.outer_prog = build_descending_perpetual_progression(self.outer_family_order, octave_offset=2)

    def get_sub_root_doubler(self, note_str: str) -> str:
        name = note_str[:-1]
        octave = int(note_str[-1])
        return f"{name}{max(1, octave - 1)}"

    def get_tonic_drones(self, root_name: str) -> tuple[str, str]:
        return f"{root_name}0", f"{root_name}1"

    def save_state(self):
        data = {
            "master_tick": getattr(self, "master_tick", int(time.time())),
            "inner_family_order": self.inner_family_order,
            "outer_family_order": self.outer_family_order
        }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[STATE] Saved state at tick {data['master_tick']}.")
        except Exception as e:
            print(f"[STATE] Failed to save state: {e}")

    async def run(self):
        while True:
            now = time.time()

            # Lock tick to absolute UTC second (1 tick / sec = 60 BPM)
            self.master_tick = int(now)
            elapsed_seconds = self.master_tick

            total_inner = len(self.inner_prog)
            total_outer = len(self.outer_prog)

            # Deterministic index lookup based on epoch time
            inner_idx = (elapsed_seconds // CHORD_DURATION_TICKS) % total_inner
            inner_chord_data = self.inner_prog[inner_idx]

            outer_idx = (elapsed_seconds // (CHORD_DURATION_TICKS * total_inner)) % total_outer
            outer_chord_data = self.outer_prog[outer_idx]

            minute_tick = elapsed_seconds % CHORD_DURATION_TICKS

            inner_triad_trigs = [self.master_tick % m == 0 for m in self.moduli]
            outer_triad_trigs = [(self.master_tick + 30) % m == 0 for m in self.moduli]
            positions = [self.master_tick % m for m in self.moduli]

            v4_trig_inner, v4_step_inner, pulses_in, total_steps_in, pattern_in = self.inner_euc_engine.tick(offset_ticks=0)
            v4_trig_outer, v4_step_outer, pulses_out, total_steps_out, pattern_out = self.outer_euc_engine.tick(offset_ticks=30)

            sub_root_note = self.get_sub_root_doubler(inner_chord_data["notes"][0])
            tonic_0, tonic_1 = self.get_tonic_drones(inner_chord_data["meta"]["tonic_name"])

            state = {
                "server_time": now,
                "tick": self.master_tick,
                "minute_tick": minute_tick,

                # Inner Loop
                "chord": inner_chord_data["notes"],
                "chord_solfege": inner_chord_data["solfege"],
                "fixed_solfege": inner_chord_data["fixed_solfege"],
                "chord_name": inner_chord_data["chord_name"],
                "key": inner_chord_data["meta"]["key"],
                "mode": inner_chord_data["meta"]["mode"],
                "scale_solfege": inner_chord_data["meta"]["scale_solfege"],

                # Outer Loop
                "outer_chord": outer_chord_data["notes"],
                "outer_solfege": outer_chord_data["solfege"],
                "outer_fixed_solfege": outer_chord_data["fixed_solfege"],
                "outer_chord_name": outer_chord_data["chord_name"],
                "outer_key": outer_chord_data["meta"]["key"],
                "outer_mode": outer_chord_data["meta"]["mode"],

                # Acoustic / Moduli Triggers
                "sub_root": sub_root_note,
                "drone_tonic_0": tonic_0,
                "drone_tonic_1": tonic_1,
                "inner_triad_trigs": inner_triad_trigs,
                "outer_triad_trigs": outer_triad_trigs,
                "v4_trig": v4_trig_inner,
                "v4_trig_outer": v4_trig_outer,
                "positions": positions,
                "v4_step": v4_step_inner,
                "v4_info": f"E({pulses_in},{total_steps_in})",
                "v4_step_outer": v4_step_outer,
                "v4_outer_info": f"E({pulses_out},{total_steps_out})",
                "inner_pattern": pattern_in,
                "outer_pattern": pattern_out,
                "a4_freq": A4_FREQ
            }

            if CONNECTED_CLIENTS:
                payload = json.dumps(state)
                await asyncio.gather(*[client.send(payload) for client in CONNECTED_CLIENTS], return_exceptions=True)

            next_tick_time = math.floor(now) + 1.0
            sleep_time = max(0.001, next_tick_time - time.time())
            await asyncio.sleep(sleep_time)

async def ws_handler(websocket):
    CONNECTED_CLIENTS.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        CONNECTED_CLIENTS.remove(websocket)

# ==========================================
# HTTP SERVER SETUP
# ==========================================
class HTTPHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            template_path = os.path.join('templates', 'index.html')
            if os.path.exists(template_path):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                with open(template_path, 'rb') as f:
                    self.wfile.write(f.read())
                return
        super().do_GET()

def start_http_server():
    os.makedirs('templates', exist_ok=True)
    with socketserver.TCPServer(("", HTTP_PORT), HTTPHandler) as httpd:
        print(f"[HTTP SERVER] Hosting Web Player on http://0.0.0.0:{HTTP_PORT}")
        httpd.serve_forever()

# ==========================================
# MAIN ENTRY POINT
# ==========================================
async def main():
    clock = MasterClock()

    def handle_exit(signum, frame):
        print("\n[SERVER] Shutting down...")
        clock.save_state()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    threading.Thread(target=start_http_server, daemon=True).start()

    async with websockets.serve(ws_handler, "0.0.0.0", WS_PORT):
        print(f"[WS SERVER] Broadcasting time sync on ws://0.0.0.0:{WS_PORT}")
        await clock.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
