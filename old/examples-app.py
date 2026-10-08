#! /usr/bin/env python3
import threading
from flask import Flask, jsonify, request
from juniper_examples.examples import (
    get_cached_data,
    process_word_example,
    llm_example_worker,
    worker_state,
    LLM_CACHE_DIR
)

app = Flask(__name__)

OLLAMA_URL = "http://127.0.0.1:11435"
MODEL_NAME = "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "juniper_examples",
        "port": 5016
    })

@app.route("/example/<word>", methods=["GET"])
def get_example(word):
    """
    Retrieves the LLM example for a word. Returns cached data immediately or
    generates it on-demand if missing.
    """
    word_clean = word.strip().lower()
    cached = get_cached_data(word_clean)

    if cached is not None:
        if cached == "NONE":
            return jsonify({"word": word_clean, "example": None, "cached": True})
        return jsonify({"word": word_clean, "example": cached, "cached": True})

    # On-demand fallback generation
    example = process_word_example(word_clean, ollama_url=OLLAMA_URL, model_name=MODEL_NAME)
    return jsonify({"word": word_clean, "example": example, "cached": False})

@app.route("/api/state", methods=["GET"])
def get_state():
    """Returns worker status and cache stats."""
    with worker_state.lock:
        return jsonify({
            "current_word": worker_state.current_word,
            "total_processed": worker_state.total_processed,
            "cache_dir": LLM_CACHE_DIR
        })

if __name__ == "__main__":
    # Start the LLM background worker thread
    t = threading.Thread(
        target=llm_example_worker,
        kwargs={"ollama_url": OLLAMA_URL, "model_name": MODEL_NAME},
        daemon=True
    )
    t.start()

    app.run(host="0.0.0.0", port=5018, debug=False)
