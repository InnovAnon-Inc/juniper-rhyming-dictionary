#!/usr/bin/env python3
import asyncio
import http.server
import json
import math
import os
import signal
import socketserver
import sys
import threading
import time
import websockets

# ==============================================================================
# CONFIGURATION & TUNING
# ==============================================================================
HTTP_PORT = 5024
WS_PORT = 65402
POLYGONS_WS_URL = "ws://127.0.0.1:65403"

BPM = 60                       # 1 tick per second (1.0s metronome pulse)
TICK_DURATION = 60.0 / BPM
CHORD_DURATION_TICKS = 60      # Real-time minute (60s) per chord shift
A4_FREQ = 432.0                # Master Reference Pitch

STATE_FILE = "clock_state.json"

# Master System Fundamental Pitch Class (0 = C)
SYSTEM_DRONE_PC = 0

# ==============================================================================
# 24-EDO MICROTONAL SYSTEM (Quarter-Tone Resolution)
# ==============================================================================
EDO_STEPS = 24

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

def identify_7th_chord_24(formatted_steps: list[int]) -> str:
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

def get_parallel_mode_pitches_24(parent_name: str, mode_degree: int, tonic_step: int) -> list[int]:
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
    scale_pitches: list[int],
    meta: dict,
    tonic_step: int,
    perceived_drone_pc: int,
    octave_offset: int = 0
) -> list[dict]:
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
) -> list[dict]:
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
    families: list[str],
    octave_offset: int = 0
) -> list[dict]:
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

# ==============================================================================
# DRONES & POLYRHYTHMIC CHORD TONE GENERATION
# ==============================================================================

def generate_drones_for_chord(chord_data: dict, perceived_drone_pc: int = SYSTEM_DRONE_PC) -> dict:
    key_tonic_step = chord_data["meta"].get("tonic_step", 120)
    chord_root_step = chord_data["steps"][0]

    key_pc = key_tonic_step % EDO_STEPS
    chord_pc = chord_root_step % EDO_STEPS

    step_key_oct0 = (24 * 1) + key_pc        # Octave 0
    step_key_oct1 = (24 * 2) + key_pc        # Octave 1
    step_chord_oct2 = (24 * 3) + chord_pc    # Octave 2

    return {
        "key_drone_low": {
            "note": f"{NOTE_NAMES_24[key_pc]}0",
            "step": step_key_oct0,
            "frequency": edo24_to_freq_432(step_key_oct0),
            "solfege": get_fixed_do_solfege_24(step_key_oct0, perceived_drone_pc)
        },
        "key_drone_high": {
            "note": f"{NOTE_NAMES_24[key_pc]}1",
            "step": step_key_oct1,
            "frequency": edo24_to_freq_432(step_key_oct1),
            "solfege": get_fixed_do_solfege_24(step_key_oct1, perceived_drone_pc)
        },
        "chord_drone": {
            "note": f"{NOTE_NAMES_24[chord_pc]}2",
            "step": step_chord_oct2,
            "frequency": edo24_to_freq_432(step_chord_oct2),
            "solfege": get_fixed_do_solfege_24(step_chord_oct2, perceived_drone_pc)
        }
    }

def compute_tone_rhythms(minute_tick: int, is_7th_allowed: bool) -> list[dict]:
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

def compute_tone_rhythms_rh(minute_tick: int, is_7th_allowed: bool) -> list[dict]:
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

# ==============================================================================
# MASTER CLOCK & BROADCAST ENGINE
# ==============================================================================

class MasterClock:
    def __init__(self):
        self.inner_family_order = ALL_FAMILIES.copy()
        self.outer_family_order = ALL_FAMILIES.copy()
        
        self.polygon_state = {
            "N": 12,
            "step_index": 0,
            "hits": {
                "left_hand_7th": True,
                "right_hand_7th": True,
                "neg_hit": False
            }
        }
        
        self.rebuild_progressions()

    def rebuild_progressions(self):
        self.inner_prog = build_descending_circle_of_fifths_progression(
            self.inner_family_order,
            octave_offset=0
        )
        self.outer_prog = build_descending_circle_of_fifths_progression(
            self.outer_family_order,
            octave_offset=2
        )

    def update_polygon_state(self, state: dict):
        if isinstance(state, dict):
            self.polygon_state = state

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
            self.master_tick = int(now)
            elapsed_seconds = self.master_tick

            total_inner = len(self.inner_prog)
            total_outer = len(self.outer_prog)

            inner_idx = (elapsed_seconds // CHORD_DURATION_TICKS) % total_inner
            inner_chord_data = self.inner_prog[inner_idx]

            outer_idx = (elapsed_seconds // (CHORD_DURATION_TICKS * total_inner)) % total_outer
            outer_chord_data = self.outer_prog[outer_idx]

            minute_tick = elapsed_seconds % CHORD_DURATION_TICKS

            inner_freqs = [edo24_to_freq_432(s) for s in inner_chord_data["steps"]]
            outer_freqs = [edo24_to_freq_432(s) for s in outer_chord_data["steps"]]

            hits = self.polygon_state.get("hits", {
                "left_hand_7th": True,
                "right_hand_7th": True,
                "neg_hit": False
            })

            lh_is_7th = hits.get("left_hand_7th", True)
            rh_is_7th = hits.get("right_hand_7th", True)

            lh_rhythms = compute_tone_rhythms(minute_tick, lh_is_7th)
            rh_rhythms = compute_tone_rhythms_rh(minute_tick, rh_is_7th)

            # Master Pitch Anchor: Set Do to Left Hand's active tonic pitch class
            lh_key_pc = inner_chord_data["meta"]["tonic_step"] % EDO_STEPS

            # Left Hand Solfège
            lh_scale_pitches = inner_chord_data["meta"].get("scale_pitches", [])
            lh_scale_solfege = " - ".join([get_fixed_do_solfege_24(p, drone_pc=lh_key_pc) for p in lh_scale_pitches]) if lh_scale_pitches else inner_chord_data["meta"]["scale_solfege"]
            lh_chord_solfege = [get_fixed_do_solfege_24(s, drone_pc=lh_key_pc) for s in inner_chord_data["steps"]]
            lh_drones = generate_drones_for_chord(inner_chord_data, perceived_drone_pc=lh_key_pc)

            # Right Hand Solfège (Fixed to Left Hand's active tonic!)
            rh_scale_pitches = outer_chord_data["meta"].get("scale_pitches", [])
            rh_scale_solfege = " - ".join([get_fixed_do_solfege_24(p, drone_pc=lh_key_pc) for p in rh_scale_pitches]) if rh_scale_pitches else outer_chord_data["meta"]["scale_solfege"]
            rh_chord_solfege = [get_fixed_do_solfege_24(s, drone_pc=lh_key_pc) for s in outer_chord_data["steps"]]
            rh_drones = generate_drones_for_chord(outer_chord_data, perceived_drone_pc=lh_key_pc)

            state = {
                "server_time": now,
                "tick": self.master_tick,
                "minute_tick": minute_tick,
                "edo_system": "24-EDO",
                "a4_freq": A4_FREQ,

                "metronome": {
                    "bpm": BPM,
                    "tick_duration_s": TICK_DURATION,
                    "is_second_pulse": True
                },
                "permissible_triggers": {
                    "left_hand_7th_allowed": lh_is_7th,
                    "right_hand_7th_allowed": rh_is_7th,
                    "neg_hit_trigger": hits.get("neg_hit", False)
                },

                "polygon_sync": self.polygon_state,

                # Left Hand / Inner Loop Data
                "left_hand": {
                    "chord_name": inner_chord_data["chord_name"],
                    "notes": inner_chord_data["notes"],
                    "solfege": lh_chord_solfege,
                    "frequencies": inner_freqs,
                    "tone_rhythms": lh_rhythms,
                    "active_tone_mask": [r["active"] for r in lh_rhythms],
                    "drones": lh_drones,
                    "key": inner_chord_data["meta"]["key"],
                    "mode": inner_chord_data["meta"]["mode"],
                    "scale_solfege": lh_scale_solfege,
                    "scale_notes": inner_chord_data["meta"]["scale_notes"]
                },

                # Right Hand / Outer Loop Data
                "right_hand": {
                    "chord_name": outer_chord_data["chord_name"],
                    "notes": outer_chord_data["notes"],
                    "solfege": rh_chord_solfege,
                    "frequencies": outer_freqs,
                    "tone_rhythms": rh_rhythms,
                    "active_tone_mask": [r["active"] for r in rh_rhythms],
                    "drones": rh_drones,
                    "key": outer_chord_data["meta"]["key"],
                    "mode": outer_chord_data["meta"]["mode"],
                    "scale_solfege": rh_scale_solfege,
                    "scale_notes": outer_chord_data["meta"]["scale_notes"]
                }
            }

            if CONNECTED_CLIENTS:
                payload = json.dumps(state)
                await asyncio.gather(*[client.send(payload) for client in CONNECTED_CLIENTS], return_exceptions=True)

            next_tick_time = math.floor(now) + 1.0
            sleep_time = max(0.001, next_tick_time - time.time())
            await asyncio.sleep(sleep_time)

async def listen_to_polygons_v2(clock: MasterClock):
    while True:
        try:
            print(f"[POLYGON CLIENT] Connecting to polygons-v2 at {POLYGONS_WS_URL}...")
            async with websockets.connect(POLYGONS_WS_URL) as ws:
                print("[POLYGON CLIENT] Connected to polygons-v2 server successfully.")
                async for message in ws:
                    try:
                        data = json.loads(message)
                        clock.update_polygon_state(data)
                    except json.JSONDecodeError:
                        pass
        except (websockets.exceptions.ConnectionClosedError, OSError) as e:
            print(f"[POLYGON CLIENT] Connection lost: {e}. Reconnecting in 3s...")
            await asyncio.sleep(3.0)

CONNECTED_CLIENTS = set()

async def ws_handler(websocket):
    CONNECTED_CLIENTS.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        CONNECTED_CLIENTS.remove(websocket)

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
        print(f"[HTTP SERVER] Chimes server web interface running at http://0.0.0.0:{HTTP_PORT}")
        httpd.serve_forever()

async def main():
    clock = MasterClock()

    def handle_exit(signum, frame):
        print("\n[SERVER] Shutting down chimes server...")
        clock.save_state()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    threading.Thread(target=start_http_server, daemon=True).start()
    asyncio.create_task(listen_to_polygons_v2(clock))

    async with websockets.serve(ws_handler, "0.0.0.0", WS_PORT):
        print(f"[WS SERVER] Broadcasting 24-EDO time sync & chords on ws://0.0.0.0:{WS_PORT}")
        await clock.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
