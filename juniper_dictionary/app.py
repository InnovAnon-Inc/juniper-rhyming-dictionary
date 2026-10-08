#!/usr/bin/env python3
import time
import json
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from juniper_dictionary.dictionary import UnifiedPhonicsEngine, SyncedNarratorState, narration_worker, CACHE_DIR

app = Flask(__name__)

phonics_engine = UnifiedPhonicsEngine(max_word_length=20)
narrator_state = SyncedNarratorState(engine_ref=phonics_engine, speech_rate=120)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(CACHE_DIR, filename)

@app.route("/api/state")
def get_state():
    with narrator_state.lock:
        state_copy = dict(narrator_state.current_state)
        state_copy["chimes"] = narrator_state.current_chimes_state
        return jsonify(state_copy)

@app.route("/api/search", methods=["POST"])
def search_word():
    data = request.get_json() or {}
    word = data.get("word", "").strip().lower()
    if not word: return jsonify({"status": "error", "message": "No word provided"}), 400

    success, message = narrator_state.enqueue_priority_word(word)
    if success:
        return jsonify({"status": "success", "message": f"Queued group for word '{word}'"})
    return jsonify({"status": "error", "message": message}), 404

@app.route("/api/chimes_update", methods=["POST"])
def receive_chimes_update():
    payload = request.get_json(silent=True)
    if payload:
        narrator_state.update_chimes_state(payload)
        return jsonify({"status": "updated"}), 200
    return jsonify({"error": "invalid payload"}), 400

@app.route("/api/stream")
def stream_events():
    def event_generator():
        last_id = -1
        while True:
            with narrator_state.lock:
                state_copy = dict(narrator_state.current_state)
                state_copy["chimes"] = narrator_state.current_chimes_state
            
            if state_copy.get("audio_id") != last_id:
                last_id = state_copy.get("audio_id")
                yield f"data: {json.dumps(state_copy)}\n\n"
            time.sleep(0.125)

    return Response(event_generator(), mimetype="text/event-stream")

import json
import threading
import websocket  # pip install websocket-client

def chimes_ws_bridge():
    def on_message(ws, message):
        try:
            payload = json.loads(message)
            narrator_state.update_chimes_state(payload)
        except Exception as e:
            pass

    while True:
        try:
            ws = websocket.WebSocketApp("ws://127.0.0.1:65432", on_message=on_message)
            ws.run_forever()
        except Exception:
            pass
        time.sleep(2)

if __name__ == "__main__":
    # In app_14.py under if __name__ == "__main__":
    threading.Thread(target=chimes_ws_bridge, daemon=True).start()

    t_narrator = threading.Thread(
        target=narration_worker,
        args=(narrator_state, phonics_engine),
        daemon=True
    )
    t_narrator.start()
    app.run(host="0.0.0.0", port=5003, debug=False)
