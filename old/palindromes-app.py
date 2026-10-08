#! /usr/bin/env python3

import threading
from flask import Flask, jsonify, request
from .palindromes import palindrome_worker, get_cached_data, PALINDROME_CACHE_DIR, VOCAB_SET

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "juniper_palindromes"}), 200

@app.route("/api/palindrome/<word>", methods=["GET"])
def get_palindrome(word):
    clean_word = word.strip().lower()
    if clean_word not in VOCAB_SET:
        return jsonify({"status": "error", "message": "Word not in vocabulary"}), 404

    cached = get_cached_data(clean_word, PALINDROME_CACHE_DIR)
    if cached is None:
        return jsonify({"status": "pending", "message": "Palindrome calculation in progress"}), 202

    return jsonify({"status": "success", "word": clean_word, "palindrome": cached if cached != "NONE" else None})

if __name__ == "__main__":
    t_worker = threading.Thread(target=palindrome_worker, daemon=True)
    t_worker.start()

    app.run(host="0.0.0.0", port=5010, debug=False)
