#! /usr/bin/env python3
"""
Synesthesia V2 Server (app-v2.py)
---------------------------------
Downstream pipeline listener:
  polygons-v2 (ws://127.0.0.1:65403) 
    --> chimes-v2 (ws://127.0.0.1:65402) 
      --> synesthesia-v2 (ws://127.0.0.1:65402 listener -> ws://0.0.0.0:65401 broadcast & HTTP 5021)

Calculates visible light spectrum wavelengths ($\lambda \in [380, 780]$ nm),
light frequencies ($\text{THz}$), and 24-EDO pitch mappings to produce a synchronized
8-color palette (4 Left-Hand tones + 4 Right-Hand tones) for downstream musical visualizers.
"""

import asyncio
import json
import math
import threading
import time
from flask import Flask, render_template_string, request, jsonify
import webcolors
import websockets

app = Flask(__name__)

# ==========================================
# CONSTANTS & PHYSICAL LOGIC
# ==========================================
HTTP_PORT = 5021
UPSTREAM_CHIMES_WS = "ws://127.0.0.1:65402"
DOWNSTREAM_WS_PORT = 65401

A4_FREQ = 432.0         # Reference tuning standard
SPEED_OF_LIGHT = 3e8    # m/s
LAMBDA_MIN = 380.0      # Visible violet limit (nm)
LAMBDA_MAX = 780.0      # Visible red limit (nm)

QUARTER_NOTE_NAMES = [
    "A", "A‡", "A#", "B♭‡", "B", "C", "C‡", "C#", "D♭‡", "D", "D‡", "D#",
    "E♭‡", "E", "F", "F‡", "F#", "G♭‡", "G", "G‡", "G#", "A♭‡", "A‡ (High)", "A# (High)"
]

# 24-EDO Quartertone lookup table mapping standard & microtonal accidental symbols to quartertone indices (0-23, where A=0)
NOTE_TO_QUARTERTONE = {
    'A': 0,
    'A𝄳': 1, 'A‡': 1,
    'A#': 2, 'Bb': 2, 'A♭': 2,
    'B𝄲': 3, 'B♭‡': 3,
    'B': 4,
    'C': 5,
    'C𝄳': 6, 'C‡': 6,
    'C#': 7, 'Db': 7,
    'D𝄲': 8, 'D♭‡': 8,
    'D': 9,
    'D𝄳': 10, 'D‡': 10,
    'D#': 11, 'Eb': 11,
    'E𝄲': 12, 'E♭‡': 12,
    'E': 13,
    'F': 14,
    'F𝄳': 15, 'F‡': 15,
    'F#': 16, 'Gb': 16,
    'G𝄲': 17, 'G♭‡': 17,
    'G': 18,
    'G𝄳': 19, 'G‡': 19,
    'G#': 20, 'Ab': 20,
    'A𝄲': 21, 'A♭‡': 21,
    'B𝄳': 22, 'C♭': 22,
    'B#': 5
}

latest_chimes_data = {}
connected_downstream_clients = set()


def get_color_name(requested_rgb):
    try:
        if hasattr(webcolors, 'rgb_to_name'):
            return webcolors.rgb_to_name(requested_rgb)
    except (ValueError, AttributeError):
        pass

    rgb_map = {}
    if hasattr(webcolors, 'CSS3_HEX_TO_NAMES'):
        for hex_code, name in webcolors.CSS3_HEX_TO_NAMES.items():
            h = hex_code.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            rgb_map[rgb] = name
    elif hasattr(webcolors, 'names'):
        for name in webcolors.names("css3"):
            try:
                rgb = webcolors.name_to_rgb(name)
                rgb_map[rgb] = name
            except Exception:
                continue

    if not rgb_map:
        return "Unknown"

    min_distance = float("inf")
    closest_name = "Unknown"

    for rgb, name in rgb_map.items():
        d = math.sqrt((rgb[0] - requested_rgb[0])**2 + (rgb[1] - requested_rgb[1])**2 + (rgb[2] - requested_rgb[2])**2)
        if d < min_distance:
            min_distance = d
            closest_name = name

    return closest_name.title() if min_distance == 0 else f"{closest_name.title()} (approx)"


def rgb_to_hsv(r, g, b):
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    mx = max(r_norm, g_norm, b_norm)
    mn = min(r_norm, g_norm, b_norm)
    df = mx - mn

    if mx == mn:
        h = 0
    elif mx == r_norm:
        h = (60 * ((g_norm - b_norm) / df) + 360) % 360
    elif mx == g_norm:
        h = (60 * ((b_norm - r_norm) / df) + 120) % 360
    elif mx == b_norm:
        h = (60 * ((r_norm - g_norm) / df) + 240) % 360

    return round(h, 2)


def hsv_to_rgb(h, s=1.0, v=1.0):
    c = v * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r_p, g_p, b_p = c, x, 0
    elif 60 <= h < 120:
        r_p, g_p, b_p = x, c, 0
    elif 120 <= h < 180:
        r_p, g_p, b_p = 0, c, x
    elif 180 <= h < 240:
        r_p, g_p, b_p = 0, x, c
    elif 240 <= h < 300:
        r_p, g_p, b_p = x, 0, c
    else:
        r_p, g_p, b_p = c, 0, x

    return int(round((r_p + m) * 255)), int(round((g_p + m) * 255)), int(round((b_p + m) * 255))


def calculate_single_voice(r, g, b):
    color_name = get_color_name((r, g, b))
    hue = rgb_to_hsv(r, g, b)

    quarter_index = int(round((hue / 360.0) * 23)) % 24
    closest_24_note = QUARTER_NOTE_NAMES[quarter_index]

    wavelength_nm = LAMBDA_MIN + (hue / 360.0) * (LAMBDA_MAX - LAMBDA_MIN)
    light_freq_hz = SPEED_OF_LIGHT / (wavelength_nm * 1e-9)

    audible_freq = A4_FREQ * (2 ** (quarter_index / 24.0))

    n_semitones = 12 * math.log2(audible_freq / A4_FREQ)
    semitone_ratio = 2 ** ((n_semitones % 12) / 12)
    chromatic_names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    closest_12_note = chromatic_names[int(round(n_semitones % 12)) % 12]

    return {
        "color_name": color_name,
        "hsv_hue": hue,
        "wavelength_nm": round(wavelength_nm, 2),
        "light_freq_thz": round(light_freq_hz / 1e12, 2),
        "audible_freq_hz": round(audible_freq, 2),
        "closest_note": f"{closest_12_note} ({round(semitone_ratio, 3)}:1 ratio)",
        "quartertone_note": closest_24_note,
        "quartertone_index": quarter_index,
        "r": r, "g": g, "b": b
    }


def calculate_note_to_color(quartertone_index):
    quartertone_index = quartertone_index % 24
    hue = (quartertone_index / 23.0) * 360.0
    r, g, b = hsv_to_rgb(hue)
    res = calculate_single_voice(r, g, b)
    res["quartertone_index"] = quartertone_index
    res["quartertone_note"] = QUARTER_NOTE_NAMES[quartertone_index]
    return res


def convert_note_list_to_colors(note_list):
    colors = []
    for note in note_list:
        clean_note = ''.join([c for c in str(note) if not c.isdigit() and c != '-'])
        q_idx = NOTE_TO_QUARTERTONE.get(clean_note, 0)
        colors.append(calculate_note_to_color(q_idx))
    return colors

# ==========================================
# CHIMES SERVER V2 WEBSOCKET LISTENER & BROADCASTER
# ==========================================
async def listen_and_broadcast_chimes():
    global latest_chimes_data
    print(f"[SYNESTHESIA V2] Listening upstream to chimes-v2 on {UPSTREAM_CHIMES_WS}...")
    
    while True:
        try:
            async with websockets.connect(UPSTREAM_CHIMES_WS) as websocket:
                print("[SYNESTHESIA V2] Connected to chimes-v2 stream.")
                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)

                    left_hand_notes = data.get("left_hand", {}).get("notes", [])
                    right_hand_notes = data.get("right_hand", {}).get("notes", [])

                    lh_colors = convert_note_list_to_colors(left_hand_notes)
                    rh_colors = convert_note_list_to_colors(right_hand_notes)

                    # Build active 8-color palette for current minute
                    combined_palette = lh_colors + rh_colors

                    augmented_state = {
                        "server_time": time.time(),
                        "tick": data.get("tick"),
                        "minute_tick": data.get("minute_tick"),
                        "metronome": data.get("metronome"),
                        "permissible_triggers": data.get("permissible_triggers"),
                        "polygon_sync": data.get("polygon_sync"),
                        
                        "left_hand": {
                            **data.get("left_hand", {}),
                            "colors": lh_colors
                        },
                        "right_hand": {
                            **data.get("right_hand", {}),
                            "colors": rh_colors
                        },
                        "palette_8_color": combined_palette
                    }

                    latest_chimes_data = augmented_state

                    # Broadcast augmented synesthesia state to downstream listeners
                    if connected_downstream_clients:
                        payload = json.dumps(augmented_state)
                        await asyncio.gather(
                            *[client.send(payload) for client in connected_downstream_clients],
                            return_exceptions=True
                        )

        except (websockets.exceptions.ConnectionClosedError, OSError) as e:
            print(f"[SYNESTHESIA V2] Upstream chimes connection lost: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2)

async def downstream_ws_handler(websocket):
    connected_downstream_clients.add(websocket)
    try:
        if latest_chimes_data:
            await websocket.send(json.dumps(latest_chimes_data))
        await websocket.wait_closed()
    finally:
        connected_downstream_clients.remove(websocket)

def start_asyncio_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def main_ws():
        async with websockets.serve(downstream_ws_handler, "0.0.0.0", DOWNSTREAM_WS_PORT):
            print(f"[SYNESTHESIA WS] Broadcasting color-augmented state on ws://0.0.0.0:{DOWNSTREAM_WS_PORT}")
            await listen_and_broadcast_chimes()

    loop.run_until_complete(main_ws())

# ==========================================
# FLASK WEB INTERFACE & API ENDPOINTS
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Synesthesia V2 Engine (Port 5021)</title>
    <style>
        body { background: #111; color: #eee; font-family: sans-serif; padding: 20px; }
        h1 { color: #00e676; }
        .palette { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
        .color-card {
            width: 130px; padding: 12px; border-radius: 8px; text-align: center;
            color: #000; font-weight: bold; font-size: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        }
        .section-title { margin-top: 25px; border-bottom: 1px solid #333; padding-bottom: 5px; }
        pre { background: #1a1a1a; padding: 12px; border-radius: 6px; overflow-x: auto; color: #4db6ac; }
    </style>
</head>
<body>
    <h1>Synesthesia V2 Color & Harmonic Visualizer</h1>
    <p>Connected to <code>chimes-v2</code> (ws://127.0.0.1:65402). Broadcasting on port <code>65401</code>.</p>
    
    <div id="status">Connecting to Synesthesia V2 WebSocket...</div>

    <h2 class="section-title">Active 8-Color Palette</h2>
    <div class="palette" id="palette-container"></div>

    <h2 class="section-title">Live State JSON</h2>
    <pre id="json-debug">Waiting for tick data...</pre>

    <script>
        const ws = new WebSocket(`ws://${window.location.hostname}:65401`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            document.getElementById('status').innerText = `Tick: ${data.tick} | Minute Tick: ${data.minute_tick}s / 60s`;
            document.getElementById('json-debug').innerText = JSON.stringify(data, null, 2);

            const container = document.getElementById('palette-container');
            container.innerHTML = '';

            if (data.palette_8_color) {
                data.palette_8_color.forEach((item, idx) => {
                    const card = document.createElement('div');
                    card.className = 'color-card';
                    card.style.backgroundColor = `rgb(${item.r}, ${item.g}, ${item.b})`;
                    // Text contrast
                    const luminance = (0.299 * item.r + 0.587 * item.g + 0.114 * item.b);
                    card.style.color = luminance > 128 ? '#000' : '#fff';
                    card.innerHTML = `
                        <div>#${idx + 1} ${item.quartertone_note}</div>
                        <div>${item.color_name}</div>
                        <div>${item.light_freq_thz} THz</div>
                        <div>${item.wavelength_nm} nm</div>
                    `;
                    container.appendChild(card);
                });
            }
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/calculate', methods=['POST'])
def calculate():
    req = request.json or {}
    r, g, b = int(req.get('r', 0)), int(req.get('g', 0)), int(req.get('b', 0))
    return jsonify(calculate_single_voice(r, g, b))

@app.route('/calculate_note', methods=['POST'])
def calculate_note():
    req = request.json or {}
    note_index = int(req.get('note_index', 0))
    return jsonify(calculate_note_to_color(note_index))

@app.route('/sync_chimes', methods=['POST'])
def sync_chimes():
    data = request.json or {}
    left_colors = convert_note_list_to_colors(data.get("left_hand", []))
    right_colors = convert_note_list_to_colors(data.get("right_hand", []))
    return jsonify({
        "left_colors": left_colors,
        "right_colors": right_colors,
        "palette_8_color": left_colors + right_colors
    })

@app.route('/chimes_state', methods=['GET'])
def chimes_state():
    """Exposes current 8-color palette, scales, and frequencies for downstream apps."""
    return jsonify(latest_chimes_data)

if __name__ == '__main__':
    threading.Thread(target=start_asyncio_loop, daemon=True).start()
    print(f"Running Synesthesia V2 Server on http://0.0.0.0:{HTTP_PORT}")
    app.run(host='0.0.0.0', port=HTTP_PORT)
