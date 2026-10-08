#! /usr/bin/env python3

import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from juniper_dictionary.dictionary import (
    UnifiedPhonicsEngine,
    SyncedNarratorState,
    narration_worker,
    CACHE_DIR
)

app = Flask(__name__)

phonics_engine = UnifiedPhonicsEngine(max_word_length=20)
narrator_state = SyncedNarratorState(
    engine_ref=phonics_engine,
    speech_rate=120
)

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
    t1 = threading.Thread(
        target=narration_worker,
        args=(narrator_state, phonics_engine),
        daemon=True
    )
    t1.start()

    app.run(host="0.0.0.0", port=5003, debug=False)
