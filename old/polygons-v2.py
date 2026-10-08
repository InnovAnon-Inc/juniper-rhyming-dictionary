import asyncio
import cmath
import math
import re
import time
import json
import threading
import requests
import websockets
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

# ==============================================================================
# RIGOROUS MATHEMATICAL & BALANCED RHYTHM ENGINE
# ==============================================================================

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    return (a * b) // gcd(a, b) if a and b else 0

def bjorklund(steps: int, pulses: int) -> list[int]:
    """Generates standard Bjorklund Euclidean rhythm E(pulses, steps)."""
    if pulses <= 0: return [0] * steps
    if pulses >= steps: return [1] * steps

    pattern = [[1] for _ in range(pulses)]
    remainder = [[0] for _ in range(steps - pulses)]

    while len(remainder) > 1:
        count = min(len(pattern), len(remainder))
        for i in range(count):
            pattern[i].extend(remainder.pop(0))

    pattern.extend(remainder)
    return [bit for group in pattern for bit in group]

def get_centroid(pattern: list[int], N: int) -> tuple[float, float]:
    if not pattern or sum(pattern) == 0:
        return 0.0, 0.0
    total_vector = 0j
    for i, active in enumerate(pattern):
        if active:
            angle = 2 * math.pi * i / N
            total_vector += cmath.exp(1j * angle)
    center = total_vector / sum(pattern)
    return center.real, center.imag

def is_strictly_balanced(pattern: list[int], N: int, tol: float = 1e-5) -> bool:
    cx, cy = get_centroid(pattern, N)
    return math.hypot(cx, cy) < tol

def is_regular_polygon(pattern: list[int], N: int) -> bool:
    k = sum(pattern)
    if k < 2 or N % k != 0:
        return False
    stride = N // k
    active_indices = [i for i, b in enumerate(pattern) if b]
    start = active_indices[0]
    expected = [(start + j * stride) % N for j in range(k)]
    return sorted(active_indices) == sorted(expected)

def analyze_pattern(pattern: list[int], N: int) -> dict:
    cx, cy = get_centroid(pattern, N)
    balanced = is_strictly_balanced(pattern, N)
    regular = is_regular_polygon(pattern, N) if balanced else False
    class_type = "Class 1 (Regular)" if regular else ("Class 2 (Composite)" if balanced else "Unbalanced")
    
    return {
        "pattern": pattern,
        "is_balanced": balanced,
        "class_type": class_type,
        "centroid": [round(cx, 5), round(cy, 5)],
        "dist_from_origin": round(math.hypot(cx, cy), 5)
    }

def get_canonical_rotation(pattern: list[int]) -> tuple[int, ...]:
    """Returns the lexicographically smallest rotational shift of a pattern."""
    n = len(pattern)
    rotations = [tuple(pattern[i:] + pattern[:i]) for i in range(n)]
    return min(rotations)

def circular_distance(pat1: list[int], pat2: list[int]) -> float:
    """Calculates rotational distance and density difference between two patterns."""
    N = len(pat1)
    idx1 = [i for i, x in enumerate(pat1) if x]
    idx2 = [i for i, x in enumerate(pat2) if x]
    
    if not idx1 and not idx2: return 0.0
    if not idx1 or not idx2: return float(N)
    
    dist = 0
    for p1 in idx1:
        dist += min(min(abs(p1 - p2), N - abs(p1 - p2)) for p2 in idx2)
    for p2 in idx2:
        dist += min(min(abs(p2 - p1), N - abs(p2 - p1)) for p1 in idx1)
        
    density_penalty = abs(len(idx1) - len(idx2)) * (N / 4.0)
    return dist + density_penalty

def sequence_smooth_path(rhythm_list: list[dict]) -> list[dict]:
    """Sorts a list of rhythms into a smooth path minimizing circular distance."""
    if not rhythm_list: return []

    unvisited = rhythm_list[:]
    path = [unvisited.pop(0)]

    while unvisited:
        current_pat = path[-1]['pattern']
        closest_idx = 0
        min_dist = float('inf')

        for i, candidate in enumerate(unvisited):
            dist = circular_distance(current_pat, candidate['pattern'])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        path.append(unvisited.pop(closest_idx))

    return path

def generate_rhythm_library(N: int) -> dict:
    """Generates all Euclidean and constructive cyclotomic balanced rhythms for N steps."""
    cyclotomic = []
    bjorklund_rhythms = []
    seen_cyc = set()
    seen_bjork = set()

    # 1. Euclidean / Bjorklund Rhythms
    for k in range(1, N):
        pat = bjorklund(N, k)
        canonical = get_canonical_rotation(pat)
        if canonical not in seen_bjork:
            seen_bjork.add(canonical)
            info = analyze_pattern(list(canonical), N)
            info["label"] = f"Euclidean E({k},{N})"
            info["is_coprime"] = gcd(k, N) == 1
            bjorklund_rhythms.append(info)

    # 2. Constructive Cyclotomic Generation
    divisors = [d for d in range(2, N) if N % d == 0]
    basis_polygons = []

    for d in divisors:
        stride = N // d
        for offset in range(stride):
            pat_set = frozenset(offset + i * stride for i in range(d))
            basis_polygons.append(pat_set)

    def build_balanced_combinations(index: int, current_union: frozenset):
        if len(seen_cyc) > 2000:
            return

        if current_union:
            canonical = get_canonical_rotation([1 if i in current_union else 0 for i in range(N)])
            if canonical not in seen_cyc:
                seen_cyc.add(canonical)
                canonical_pat = list(canonical)
                info = analyze_pattern(canonical_pat, N)
                info["label"] = f"Cyclotomic {info['class_type']} ({sum(canonical_pat)} pulses)"
                cyclotomic.append(info)

        for i in range(index, len(basis_polygons)):
            if not current_union.intersection(basis_polygons[i]):
                build_balanced_combinations(i + 1, current_union.union(basis_polygons[i]))

    build_balanced_combinations(0, frozenset())

    return {
        "cyclotomic": sequence_smooth_path(cyclotomic),
        "bjorklund": sequence_smooth_path(bjorklund_rhythms)
    }

def is_nontrivial(pattern: list[int]) -> bool:
    """Excludes trivial patterns that have fewer than 2 beats or fewer than 2 rests."""
    k = sum(pattern)
    n = len(pattern)
    return 1 < k < (n - 1)

# ==============================================================================
# NESTED POLYGON PROGRESSION ENGINE
# ==============================================================================

class PolygonProgressionEngine:
    """
    Iterates through all non-trivial balanced polygons (positive and negative).
    Repeats each valid combination 7 times before stepping to the next pair.
    """
    #def __init__(self, step_cycles=list(range(3, 33)), repeats_per_combo=7, bpm=60, ws_port=65433):
    def __init__(self, step_cycles=list(range(3, 33)), repeats_per_combo=7, bpm=60, ws_port=65403):
        self.step_cycles = step_cycles
        self.repeats_per_combo = repeats_per_combo
        self.bpm = bpm
        self.ws_port = ws_port
        self.connected_clients = set()

        self.current_n_idx = 0
        self.sequence = []
        self.combo_idx = 0
        self.current_repeat = 0
        self.step_in_pattern = 0

        self.lock = threading.Lock()
        self.current_state = {}

        self._load_combos_for_n(self.step_cycles[self.current_n_idx])

    def _get_balanced_polygons(self, N: int) -> list[dict]:
        lib = generate_rhythm_library(N)
        all_pats = []
        seen = set()
        for group in [lib.get("bjorklund", []), lib.get("cyclotomic", [])]:
            for info in group:
                pat = info["pattern"]
                key = tuple(pat)
                if key not in seen and is_nontrivial(pat):
                    seen.add(key)
                    all_pats.append(info)
        return all_pats

    def _load_combos_for_n(self, N: int):
        balanced = self._get_balanced_polygons(N)
        combos = []
        
        # Nested loop iterating all possible positive and negative non-trivial polygon combinations
        for pos_info in balanced:
            pos_pat = pos_info["pattern"]
            for neg_info in balanced:
                neg_pat = neg_info["pattern"]
                
                # Bitwise subtraction: Positive MINUS Negative
                sub_pat = [1 if (p and not q) else 0 for p, q in zip(pos_pat, neg_pat)]
                
                # Keep only combinations resulting in non-trivial patterns
                if is_nontrivial(sub_pat):
                    combos.append({
                        "N": N,
                        "pos": pos_info,
                        "neg": neg_info,
                        "sub_pattern": sub_pat,
                        "sub_analysis": analyze_pattern(sub_pat, N)
                    })

        self.sequence = combos
        self.combo_idx = 0
        self.current_repeat = 0
        self.step_in_pattern = 0
        
        if not combos:
            self._advance_n()

    def _advance_n(self):
        self.current_n_idx = (self.current_n_idx + 1) % len(self.step_cycles)
        self._load_combos_for_n(self.step_cycles[self.current_n_idx])

    def tick(self) -> dict:
        with self.lock:
            if not self.sequence:
                self._advance_n()
                if not self.sequence:
                    return {}

            curr_combo = self.sequence[self.combo_idx]
            N = curr_combo["N"]
            pos_pat = curr_combo["pos"]["pattern"]
            neg_pat = curr_combo["neg"]["pattern"]
            sub_pat = curr_combo["sub_pattern"]

            eval_idx = self.step_in_pattern % N
            pos_hit = bool(pos_pat[eval_idx])
            neg_hit = bool(neg_pat[eval_idx])
            sub_hit = bool(sub_pat[eval_idx])

            state = {
                "N": N,
                "step_index": eval_idx,
                "repeat_count": self.current_repeat + 1,
                "total_repeats": self.repeats_per_combo,
                "combo_index": self.combo_idx + 1,
                "total_combos": len(self.sequence),
                "pos_polygon": curr_combo["pos"],
                "neg_polygon": curr_combo["neg"],
                "sub_pattern": sub_pat,
                "sub_analysis": curr_combo["sub_analysis"],
                "hits": {
                    "left_hand_7th": pos_hit,          # Positive polygon trigger
                    "right_hand_7th": sub_hit,         # (Positive - Negative) polygon trigger
                    "neg_hit": neg_hit
                },
                "timestamp": time.time()
            }
            self.current_state = state

            # Step pattern index and handle 7-repeat iteration
            self.step_in_pattern += 1
            if self.step_in_pattern >= N:
                self.step_in_pattern = 0
                self.current_repeat += 1
                if self.current_repeat >= self.repeats_per_combo:
                    self.current_repeat = 0
                    self.combo_idx += 1
                    if self.combo_idx >= len(self.sequence):
                        self._advance_n()

            return state

polygon_engine = PolygonProgressionEngine(step_cycles=list(range(3, 33)), repeats_per_combo=7, bpm=60)

# ==============================================================================
# WEBSOCKET BROADCAST & FLASK API
# ==============================================================================

def note_to_freq_432(note_str: str) -> float:
    match = re.match(r"^([A-Ga-g][#b]?)(-?\d+)$", note_str.strip())
    if not match:
        return 432.0
    
    note_name, octave_str = match.groups()
    octave = int(octave_str)
    
    semitone_offsets = {
        'C': -9, 'C#': -8, 'Db': -8, 'D': -7, 'D#': -6, 'Eb': -6,
        'E': -5, 'F': -4, 'F#': -3, 'Gb': -3, 'G': -2, 'G#': -1,
        'Ab': -1, 'A': 0, 'A#': 1, 'Bb': 1, 'B': 2
    }
    
    clean_note = note_name.capitalize()
    semitone = semitone_offsets.get(clean_note, 0)
    midi_num = (octave + 1) * 12 + semitone + 9
    return 432.0 * math.pow(2, (midi_num - 69) / 12.0)

@app.route('/api/polygon_state', methods=['GET'])
def get_polygon_state():
    with polygon_engine.lock:
        return jsonify(polygon_engine.current_state)

@app.route('/api/library', methods=['GET'])
def get_library():
    n_steps = int(request.args.get('n', 12))
    lib = generate_rhythm_library(n_steps)
    return jsonify({"n": n_steps, "library": lib})

@app.route('/api/chimes_synesthesia', methods=['GET'])
def get_chimes_synesthesia():
    try:
        res = requests.get('http://127.0.0.1:5001/chimes_state', timeout=1.0)
        if res.status_code == 200 and res.json():
            data = res.json()
            for hand in ['inner_hand', 'outer_hand', 'left_hand', 'right_hand']:
                if hand in data and 'notes' in data[hand]:
                    for idx, note_str in enumerate(data[hand]['notes']):
                        freq = note_to_freq_432(note_str)
                        if idx < len(data[hand]['colors']):
                            data[hand]['colors'][idx]['exact_freq_hz'] = freq
            return jsonify(data)
    except Exception:
        pass

    fallback_data = {
        "inner_hand": {
            "chord_name": "Cmaj7 (Lower Hand / Octave 2-3)",
            "notes": ["C2", "G2", "B2", "E3"],
            "colors": [
                {"r": 255, "g": 87,  "b": 34,  "exact_freq_hz": note_to_freq_432("C2")},
                {"r": 76,  "g": 175, "b": 80,  "exact_freq_hz": note_to_freq_432("G2")},
                {"r": 33,  "g": 150, "b": 243, "exact_freq_hz": note_to_freq_432("B2")},
                {"r": 255, "g": 193, "b": 7,   "exact_freq_hz": note_to_freq_432("E3")}
            ]
        },
        "outer_hand": {
            "chord_name": "Am9 (Upper Hand / Octave 4-5)",
            "notes": ["C4", "E4", "G4", "B4"],
            "colors": [
                {"r": 255, "g": 87,  "b": 34,  "exact_freq_hz": note_to_freq_432("C4")},
                {"r": 255, "g": 193, "b": 7,   "exact_freq_hz": note_to_freq_432("E4")},
                {"r": 76,  "g": 175, "b": 80,  "exact_freq_hz": note_to_freq_432("G4")},
                {"r": 33,  "g": 150, "b": 243, "exact_freq_hz": note_to_freq_432("B4")}
            ]
        }
    }
    return jsonify(fallback_data)

@app.route('/api/evaluate_voice_bitwise', methods=['POST'])
def evaluate_voice_bitwise():
    data = request.json
    polygons = data.get('polygons', [])
    N = data.get('N', 12)
    
    voices = {}
    for poly in polygons:
        tone_id = poly.get('tone_id')
        pat = poly.get('pattern', [0] * N)
        is_pos = poly.get('type') == 'positive'
        
        if tone_id not in voices:
            voices[tone_id] = {'pos': [0] * N, 'neg': [0] * N, 'tone_name': poly.get('tone_name')}
        
        for i in range(len(pat)):
            idx = i % N
            if pat[i]:
                if is_pos:
                    voices[tone_id]['pos'][idx] = 1
                else:
                    voices[tone_id]['neg'][idx] = 1

    global_combined_pattern = [0] * N
    voice_results = {}
    
    for tone_id, vdata in voices.items():
        res_pat = [1 if (pos and not neg) else 0 for pos, neg in zip(vdata['pos'], vdata['neg'])]
        voice_results[tone_id] = {
            "tone_name": vdata['tone_name'],
            "pattern": res_pat,
            "analysis": analyze_pattern(res_pat, N)
        }
        for i in range(N):
            if res_pat[i]:
                global_combined_pattern[i] = 1

    global_analysis = analyze_pattern(global_combined_pattern, N)

    return jsonify({
        "voices": voice_results,
        "global_analysis": global_analysis
    })

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# ==============================================================================
# FRONTEND INTERFACE
# ==============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Harmonic Polygon Server - Live Nested Loop</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121214;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 { margin-bottom: 5px; color: #4db6ac; }
        p.subtitle { color: #888; margin-top: 0; margin-bottom: 20px; text-align: center; }
        
        .top-bar {
            background: #1e1e24;
            padding: 15px 25px;
            border-radius: 8px;
            display: flex;
            gap: 20px;
            align-items: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        label { font-weight: bold; font-size: 14px; }
        
        .workspace {
            display: flex;
            gap: 25px;
            flex-wrap: wrap;
            justify-content: center;
            max-width: 1450px;
            width: 100%;
        }
        
        .canvas-card {
            background: #1e1e24;
            padding: 20px;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        canvas { background: #18181c; border-radius: 50%; border: 1px solid #333; }
        
        .panel {
            background: #1e1e24;
            padding: 15px;
            border-radius: 8px;
            width: 540px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .status-box {
            background: #25252e;
            padding: 12px;
            border-radius: 6px;
            font-size: 14px;
            border-left: 4px solid #00e676;
            line-height: 1.6;
        }
        
        .badge { font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
        .badge-pos { background: #00897b; color: #fff; }
        .badge-neg { background: #d32f2f; color: #fff; }
        .badge-sub { background: #7b1fa2; color: #fff; }
    </style>
</head>
<body>

    <h1>Harmonic Polygon Progression Server</h1>
    <p class="subtitle">Live Bitwise Subtraction Broadcast & Dual-Hand Chimes Sync Engine</p>

    <div class="top-bar">
        <span id="n-display" style="font-size: 16px; font-weight: bold; color: #00e676;">N = --</span>
        <span id="combo-display" style="font-size: 14px; color: #bbb;">Combo: -- / --</span>
        <span id="repeat-display" style="font-size: 14px; color: #ffeb3b;">Repeat: -- / 7</span>
        <span id="step-display" style="font-size: 14px; color: #4db6ac;">Step: --</span>
    </div>

    <div class="workspace">
        <div class="canvas-card">
            <canvas id="polyCanvas" width="500" height="500"></canvas>
        </div>

        <div class="panel">
            <h3>Live Nested Loop Status</h3>
            <div class="status-box" id="loop-status">
                Connecting to Polygon Engine WebSocket...
            </div>
        </div>
    </div>

    <script>
        let ws = null;

        function connectWS() {
            //ws = new WebSocket(`ws://${window.location.hostname}:65433`);
            ws = new WebSocket(`ws://${window.location.hostname}:65403`);

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateUI(data);
            };

            ws.onclose = () => {
                setTimeout(connectWS, 1000);
            };
        }

        function updateUI(data) {
            if (!data || !data.pos_polygon) return;

            document.getElementById('n-display').innerText = `N = ${data.N}`;
            document.getElementById('combo-display').innerText = `Combo: ${data.combo_index} / ${data.total_combos}`;
            document.getElementById('repeat-display').innerText = `Repeat: ${data.repeat_count} / ${data.total_repeats}`;
            document.getElementById('step-display').innerText = `Step: ${data.step_index + 1} / ${data.N}`;

            const posLabel = data.pos_polygon.label || 'Positive Polygon';
            const negLabel = data.neg_polygon.label || 'Negative Polygon';

            document.getElementById('loop-status').innerHTML = `
                <div><span class="badge badge-pos">POSITIVE (Left Hand 7ths)</span><br>
                <strong>${posLabel}</strong>: [${data.pos_polygon.pattern.join('')}]</div><br>
                
                <div><span class="badge badge-neg">NEGATIVE (Bitwise Subtracted)</span><br>
                <strong>${negLabel}</strong>: [${data.neg_polygon.pattern.join('')}]</div><br>
                
                <div><span class="badge badge-sub">RESULT (Right Hand 7ths)</span><br>
                <strong>Pattern</strong>: [${data.sub_pattern.join('')}]</div>
            `;

            drawCanvas(data);
        }

        function drawCanvas(data) {
            const canvas = document.getElementById('polyCanvas');
            const ctx = canvas.getContext('2d');
            const N = data.N;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = 180;

            // Draw radial grid
            for (let i = 0; i < N; i++) {
                const angle = (2 * Math.PI * i / N) - (Math.PI / 2);
                const x1 = centerX + (radius + 10) * Math.cos(angle);
                const y1 = centerY + (radius + 10) * Math.sin(angle);
                
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(x1, y1);
                ctx.strokeStyle = (i === data.step_index) ? '#ffeb3b' : '#333';
                ctx.lineWidth = (i === data.step_index) ? 2.5 : 0.8;
                ctx.stroke();
            }

            // Draw Positive Polygon (Left Hand)
            drawPolygonShape(ctx, centerX, centerY, radius, data.pos_polygon.pattern, '#00e676', false, 2.5);
            // Draw Negative Polygon
            drawPolygonShape(ctx, centerX, centerY, radius - 15, data.neg_polygon.pattern, '#ff5252', true, 1.5);
            // Draw Subtracted Result Polygon (Right Hand)
            drawPolygonShape(ctx, centerX, centerY, radius - 30, data.sub_pattern, '#ab47bc', false, 2.5);
        }

        function drawPolygonShape(ctx, centerX, centerY, radius, pattern, color, isDashed, lineWidth) {
            const N = pattern.length;
            const vertices = [];

            for (let i = 0; i < N; i++) {
                if (pattern[i]) {
                    const angle = (2 * Math.PI * (i / N)) - (Math.PI / 2);
                    vertices.push({
                        x: centerX + radius * Math.cos(angle),
                        y: centerY + radius * Math.sin(angle)
                    });
                }
            }

            if (vertices.length > 1) {
                ctx.beginPath();
                ctx.moveTo(vertices[0].x, vertices[0].y);
                vertices.forEach(v => ctx.lineTo(v.x, v.y));
                ctx.closePath();

                ctx.strokeStyle = color;
                ctx.setLineDash(isDashed ? [5, 5] : []);
                ctx.lineWidth = lineWidth;
                ctx.stroke();
                ctx.setLineDash([]);
            }

            vertices.forEach(v => {
                ctx.beginPath();
                ctx.arc(v.x, v.y, 4, 0, 2 * Math.PI);
                ctx.fillStyle = color;
                ctx.fill();
            });
        }

        connectWS();
    </script>
</body>
</html>
"""

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def start_ws_broadcast():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def ws_handler(websocket):
        polygon_engine.connected_clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            polygon_engine.connected_clients.remove(websocket)

    async def broadcast_loop():
        while True:
            state = polygon_engine.tick()
            if polygon_engine.connected_clients and state:
                payload = json.dumps(state)
                await asyncio.gather(
                    *[client.send(payload) for client in polygon_engine.connected_clients],
                    return_exceptions=True
                )
            now = time.time()
            next_tick = math.floor(now) + (60.0 / polygon_engine.bpm)
            await asyncio.sleep(max(0.01, next_tick - time.time()))

    async def main_ws():
        #async with websockets.serve(ws_handler, "0.0.0.0", 65433):
        async with websockets.serve(ws_handler, "0.0.0.0", 65403):
            #print("[POLYGON WS] Broadcasting polygon engine on ws://0.0.0.0:65433")
            print("[POLYGON WS] Broadcasting polygon engine on ws://0.0.0.0:65403")
            await broadcast_loop()

    loop.run_until_complete(main_ws())

if __name__ == '__main__':
    threading.Thread(target=start_ws_broadcast, daemon=True).start()
    print("Running Harmonic Polygon Server on http://0.0.0.0:5007")
    app.run(host='0.0.0.0', port=5017, debug=True)
