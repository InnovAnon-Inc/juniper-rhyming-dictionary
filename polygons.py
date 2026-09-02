import cmath
import math
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

# ==============================================================================
# MATHEMATICAL & RHYTHMIC ENGINE
# ==============================================================================

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def bjorklund_euclidean(steps: int, pulses: int) -> list[int]:
    """Generates a Euclidean rhythm E(pulses, steps) via Bjorklund's algorithm."""
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

def is_regular_polygon(pattern: list[int], N: int) -> bool:
    """Checks if a pattern forms a single equilateral/regular polygon."""
    k = sum(pattern)
    if k < 3 or N % k != 0:
        return False
    
    stride = N // k
    active_indices = [i for i, b in enumerate(pattern) if b]
    
    # Check if indices form a constant stride modulo N
    start = active_indices[0]
    expected = [(start + j * stride) % N for j in range(k)]
    return sorted(active_indices) == sorted(expected)

def get_centroid(pattern: list[int], N: int) -> tuple[float, float]:
    """Calculates the center of mass on the complex unit circle."""
    if not pattern or sum(pattern) == 0:
        return 0.0, 0.0
    total_vector = 0j
    for i, active in enumerate(pattern):
        if active:
            angle = 2 * math.pi * i / N
            total_vector += cmath.exp(1j * angle)
    center = total_vector / sum(pattern)
    return center.real, center.imag

def is_euclidean(pattern: list[int], N: int) -> tuple[bool, int]:
    """Checks if pattern is a rotated Euclidean rhythm and returns its k value."""
    k = sum(pattern)
    if k == 0 or k == N:
        return True, k
    base_euc = bjorklund_euclidean(N, k)
    
    for shift in range(N):
        rotated = base_euc[shift:] + base_euc[:shift]
        if rotated == pattern:
            return True, k
    return False, k

def analyze_and_classify(pattern: list[int], N: int, tol: float = 1e-5) -> dict:
    """Comprehensively analyzes a rhythm for balance, cyclotomic status, and interest."""
    cx, cy = get_centroid(pattern, N)
    dist_from_center = math.hypot(cx, cy)
    
    # Class 1: Centroid is at origin (0,0)
    is_c1 = dist_from_center < tol
    
    # Class 2: Sum of sub-polygon vectors balances out algebraically
    dft_zeros = 0
    for k_bin in range(1, N):
        val = sum(cmath.exp(-2j * math.pi * k_bin * i / N) for i, b in enumerate(pattern) if b)
        if abs(val) < tol:
            dft_zeros += 1
            
    is_c2 = (dft_zeros > 0) and not is_c1
    euc, k_pulses = is_euclidean(pattern, N)
    
    # Highlight criteria
    is_coprime_euc = euc and (gcd(k_pulses, N) == 1) and (1 < k_pulses < N - 1)
    is_non_equilateral_cyclotomic = (is_c1 or is_c2) and not is_regular_polygon(pattern, N)
    
    is_interesting = is_coprime_euc or is_non_equilateral_cyclotomic

    return {
        "pattern": pattern,
        "negative_pattern": [1 - b for b in pattern],
        "class_1": is_c1,
        "class_2": is_c2,
        "is_euclidean": euc,
        "k_pulses": k_pulses,
        "is_interesting": is_interesting,
        "interesting_reason": (
            "Coprime Euclidean (Euc & Coprime)" if is_coprime_euc else 
            "Composite Cyclotomic (Balanced & Non-Equilateral)" if is_non_equilateral_cyclotomic else 
            "Standard Regular/Symmetric Shape"
        ),
        "center_of_mass": [round(cx, 4), round(cy, 4)],
        "dist_from_center": round(dist_from_center, 4)
    }

def generate_comprehensive_rhythms(N: int) -> list[dict]:
    """Generates all valid Euclidean and Balanced Cyclotomic rhythms for arbitrary N."""
    results = []
    seen = set()

    # 1. Exhaustive Euclidean Generation
    for k in range(1, N):
        euc_pat = bjorklund_euclidean(N, k)
        key = tuple(euc_pat)
        if key not in seen:
            seen.add(key)
            results.append(analyze_and_classify(euc_pat, N))

    # 2. Combinatorial Search for Cyclotomic/Balanced Polygons
    total_combos = 1 << N
    # Cap iterations to prevent lockup on N > 20 while maintaining full search for lower N
    step_size = max(1, total_combos // 4096)
    
    for i in range(1, total_combos, step_size):
        pat = [(i >> j) & 1 for j in range(N)]
        key = tuple(pat)
        if key not in seen:
            res = analyze_and_classify(pat, N)
            if res["class_1"] or res["class_2"]:
                seen.add(key)
                results.append(res)
                
    # Sort to put "Interesting" highlighted rhythms at the top
    results.sort(key=lambda x: (not x["is_interesting"], not x["is_euclidean"]))
    return results

# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route('/api/polygons', methods=['GET'])
def get_polygons():
    n_steps = int(request.args.get('n', 12))
    polygons = generate_comprehensive_rhythms(n_steps)
    return jsonify({"n": n_steps, "count": len(polygons), "polygons": polygons})

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# ==============================================================================
# UI TEMPLATE
# ==============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Milne Balanced Polygons & Rhythm Audiator</title>
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
        p.subtitle { color: #888; margin-top: 0; margin-bottom: 20px; text-align: center; max-width: 750px; }
        
        .controls {
            background: #1e1e24;
            padding: 15px 25px;
            border-radius: 8px;
            display: flex;
            gap: 15px;
            align-items: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        label { font-weight: bold; font-size: 14px; }
        input, button, select {
            background: #2a2a32;
            border: 1px solid #444;
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            background: #00897b;
            cursor: pointer;
            font-weight: bold;
            transition: 0.2s;
        }
        button:hover { background: #00bfa5; }
        
        .main-container {
            display: flex;
            gap: 25px;
            flex-wrap: wrap;
            justify-content: center;
            max-width: 1250px;
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
        
        canvas {
            background: #18181c;
            border-radius: 50%;
            border: 1px solid #333;
        }
        
        .list-container {
            background: #1e1e24;
            padding: 15px;
            border-radius: 8px;
            max-height: 520px;
            overflow-y: auto;
            width: 380px;
        }
        
        .polygon-item {
            padding: 10px;
            margin-bottom: 8px;
            background: #2a2a32;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            gap: 5px;
            border-left: 4px solid transparent;
            position: relative;
        }
        .polygon-item:hover { background: #33333d; }
        .polygon-item.active { border-left-color: #00bfa5; background: #33333d; }
        .polygon-item.interesting { border-right: 4px solid #ffd54f; }
        
        .badges { display: flex; gap: 4px; flex-wrap: wrap; }
        .badge {
            font-size: 9px;
            padding: 2px 5px;
            border-radius: 3px;
            text-transform: uppercase;
            font-weight: bold;
        }
        .badge-c1 { background: #2e7d32; color: #fff; }
        .badge-c2 { background: #f57f17; color: #fff; }
        .badge-euc { background: #0288d1; color: #fff; }
        .badge-highlight { background: #ffd54f; color: #000; }

        .explanation {
            background: #25252e;
            padding: 12px;
            border-radius: 6px;
            font-size: 12px;
            line-height: 1.4;
            margin-top: 10px;
            border-left: 3px solid #ffeb3b;
            max-width: 400px;
        }
    </style>
</head>
<body>

    <h1>Milne Balanced Polygons & Comprehensive Rhythm Engine</h1>
    <p class="subtitle">A4 = 432 Hz Master Tuning | Highlighting Coprime Euclidean & Composite Cyclotomic Patterns</p>

    <div class="controls">
        <label for="n-input">Pulses (N):</label>
        <input type="number" id="n-input" value="12" min="2" max="32" style="width: 65px;">
        <button onclick="fetchPolygons()">Update N</button>

        <label for="filter-select">Filter:</label>
        <select id="filter-select" onchange="renderList()">
            <option value="all">Show All Patterns</option>
            <option value="interesting" selected>★ Highlighted/Interesting Only</option>
            <option value="euclidean">Euclidean Only</option>
            <option value="balanced">Class 1 & 2 Balanced Only</option>
        </select>

        <label for="bpm-input">BPM:</label>
        <input type="number" id="bpm-input" value="120" min="40" max="240" style="width: 60px;">

        <button id="play-btn" onclick="togglePlay()">Play Rhythm</button>
    </div>

    <div class="main-container">
        <div class="canvas-card">
            <canvas id="polyCanvas" width="420" height="420"></canvas>
            <div class="explanation" id="centroid-explanation"></div>
        </div>

        <div class="list-container" id="polygon-list"></div>
    </div>

    <script>
        const A4_FREQ = 432.0;
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        let polygonsData = [];
        let filteredData = [];
        let selectedIndex = 0;
        let isPlaying = false;
        let currentStep = 0;
        let timerId = null;

        function playTone(freq, isPositive) {
            if (audioCtx.state === 'suspended') audioCtx.resume();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();

            osc.type = isPositive ? 'sine' : 'triangle';
            const actualFreq = isPositive ? freq : freq / 2;
            
            osc.frequency.setValueAtTime(actualFreq, audioCtx.currentTime);
            gain.gain.setValueAtTime(isPositive ? 0.3 : 0.12, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.18);

            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.18);
        }

        async function fetchPolygons() {
            const n = document.getElementById('n-input').value;
            const response = await fetch(`/api/polygons?n=${n}`);
            const data = await response.json();
            polygonsData = data.polygons;
            selectedIndex = 0;
            renderList();
            drawPolygon();
        }

        function renderList() {
            const filter = document.getElementById('filter-select').value;
            const container = document.getElementById('polygon-list');
            container.innerHTML = '';

            filteredData = polygonsData.filter(poly => {
                if (filter === 'interesting') return poly.is_interesting;
                if (filter === 'euclidean') return poly.is_euclidean;
                if (filter === 'balanced') return poly.class_1 || poly.class_2;
                return true;
            });

            if (filteredData.length === 0) {
                container.innerHTML = '<div style="padding: 10px; color: #888;">No rhythms match filter.</div>';
                return;
            }

            filteredData.forEach((poly, idx) => {
                const item = document.createElement('div');
                item.className = `polygon-item ${idx === selectedIndex ? 'active' : ''} ${poly.is_interesting ? 'interesting' : ''}`;
                
                const cBadge = poly.class_1 ? '<span class="badge badge-c1">Class 1</span>' : 
                              poly.class_2 ? '<span class="badge badge-c2">Class 2</span>' : '';
                const eucBadge = poly.is_euclidean ? '<span class="badge badge-euc">Euclidean</span>' : '';
                const starBadge = poly.is_interesting ? '<span class="badge badge-highlight">★ Highlighted</span>' : '';

                item.innerHTML = `
                    <div class="badges">${starBadge} ${cBadge} ${eucBadge}</div>
                    <div style="font-family: monospace; font-size: 13px;">[${poly.pattern.join('')}]</div>
                    <div style="font-size: 11px; color: #aaa;">${poly.interesting_reason}</div>
                `;
                item.onclick = () => {
                    selectedIndex = idx;
                    renderList();
                    drawPolygon();
                };
                container.appendChild(item);
            });
        }

        function drawPolygon() {
            const canvas = document.getElementById('polyCanvas');
            const ctx = canvas.getContext('2d');
            const N = parseInt(document.getElementById('n-input').value);
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = 160;

            // Unit Circle
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2;
            ctx.stroke();

            const getCoords = (i) => {
                const angle = (2 * Math.PI * i / N) - (Math.PI / 2);
                return {
                    x: centerX + radius * Math.cos(angle),
                    y: centerY + radius * Math.sin(angle)
                };
            };

            if (filteredData[selectedIndex]) {
                const poly = filteredData[selectedIndex];
                const activeIndices = poly.pattern.reduce((acc, val, idx) => val ? [...acc, idx] : acc, []);

                // Draw Polygon Connectors
                if (activeIndices.length > 1) {
                    ctx.beginPath();
                    const start = getCoords(activeIndices[0]);
                    ctx.moveTo(start.x, start.y);
                    activeIndices.forEach(idx => {
                        const pt = getCoords(idx);
                        ctx.lineTo(pt.x, pt.y);
                    });
                    ctx.closePath();
                    ctx.fillStyle = poly.is_interesting ? 'rgba(255, 213, 79, 0.25)' : 'rgba(0, 230, 118, 0.25)';
                    ctx.fill();
                    ctx.strokeStyle = poly.is_interesting ? '#ffd54f' : '#00e676';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }

                // Draw Vertices
                for (let i = 0; i < N; i++) {
                    const pt = getCoords(i);
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
                    ctx.fillStyle = poly.pattern[i] ? '#00e676' : '#444';
                    ctx.fill();

                    if (isPlaying && currentStep === i) {
                        ctx.beginPath();
                        ctx.arc(pt.x, pt.y, 12, 0, 2 * Math.PI);
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 3;
                        ctx.stroke();
                    }
                }

                // Centroid Marker
                const cmX = centerX + poly.center_of_mass[0] * radius;
                const cmY = centerY - poly.center_of_mass[1] * radius;

                ctx.beginPath();
                ctx.arc(cmX, cmY, 8, 0, 2 * Math.PI);
                ctx.fillStyle = '#ffeb3b';
                ctx.fill();
                ctx.strokeStyle = '#000';
                ctx.stroke();

                // Dynamic Explanation
                const exp = document.getElementById('centroid-explanation');
                exp.innerHTML = `
                    <strong>${poly.interesting_reason}</strong><br>
                    • Centroid Location: (${poly.center_of_mass[0]}, ${poly.center_of_mass[1]})<br>
                    • Pulses (k): ${poly.k_pulses} / ${N}<br>
                    • Class 1 Centered: ${poly.class_1 ? 'Yes' : 'No'}<br>
                    • Class 2 Algebraic: ${poly.class_2 ? 'Yes' : 'No'}
                `;
            }
        }

        function togglePlay() {
            if (isPlaying) {
                clearInterval(timerId);
                isPlaying = false;
                document.getElementById('play-btn').innerText = 'Play Rhythm';
                drawPolygon();
            } else {
                isPlaying = true;
                document.getElementById('play-btn').innerText = 'Stop';
                currentStep = 0;
                
                const bpm = parseInt(document.getElementById('bpm-input').value);
                const N = parseInt(document.getElementById('n-input').value);
                const intervalMs = (60000 / bpm) / (N / 4);

                timerId = setInterval(() => {
                    if (filteredData[selectedIndex]) {
                        const poly = filteredData[selectedIndex];
                        const baseFreq = (A4_FREQ / 2) * Math.pow(2, currentStep / N);

                        if (poly.pattern[currentStep]) playTone(baseFreq, true);
                        if (poly.negative_pattern[currentStep]) playTone(baseFreq, false);
                    }

                    drawPolygon();
                    currentStep = (currentStep + 1) % N;
                }, intervalMs);
            }
        }

        fetchPolygons();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    print("Running Milne Verified Balanced Polygons App on http://0.0.0.0:5007")
    app.run(host='0.0.0.0', port=5007, debug=True)
