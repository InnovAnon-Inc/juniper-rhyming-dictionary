import os
import time
import json
import threading
import urllib.request
import nltk
from nltk.corpus import cmudict, wordnet

# Ensure NLTK datasets are downloaded
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)

nltk_lock = threading.Lock()

def safe_wordnet_synsets(word):
    with nltk_lock:
        return wordnet.synsets(word)

def safe_wordnet_all_lemma_names():
    with nltk_lock:
        return list(wordnet.all_lemma_names())

CMU_DICT = cmudict.dict()
VALID_SHORT_WORDS = {"a", "i", "in", "on", "no", "is", "it", "or", "to", "at", "am", "an", "so", "do", "go", "me", "my", "we", "he", "be", "us", "up", "if"}

def is_valid_real_word(w):
    w_clean = w.lower()
    if not w_clean.isalpha():
        return False
    if len(w_clean) < 3 and w_clean not in VALID_SHORT_WORDS:
        return False
    if w_clean != w and w.isupper():
        return False
    return w_clean in CMU_DICT

def load_vocab():
    return [
        w.lower() for w in set(safe_wordnet_all_lemma_names())
        if is_valid_real_word(w)
    ]

# Shared Disk Cache Directory
LLM_CACHE_DIR = os.path.join(os.path.dirname(__file__), "llm_cache")
os.makedirs(LLM_CACHE_DIR, exist_ok=True)

def get_cached_data(word, cache_dir=LLM_CACHE_DIR):
    """Safely read a cached example from disk."""
    filepath = os.path.join(cache_dir, f"{word.lower()}.txt")
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content if content else None
    return None

def write_cached_data(word, data, cache_dir=LLM_CACHE_DIR):
    """Write an example result to disk."""
    filepath = os.path.join(cache_dir, f"{word.lower()}.txt")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(str(data))

def generate_ollama_example(word, definition=None, ollama_url="http://127.0.0.1:11435", model_name="Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"):
    """
    Sends a completion request to local llama-server / Ollama instance.
    """
    if not definition:
        synsets = safe_wordnet_synsets(word)
        definition = synsets[0].definition() if synsets else f"the word {word}"

    prompt = (
        f"The word is '{word}', "
        f"with definition '{definition}'. "
        f"Use the word '{word}' in a sentence. "
        f"Output only the sentence."
    )

    payload = json.dumps({
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful dictionary assistant who provides usage examples for words whose examples are missing from the NLTK wordnet. Output only the requested sentence."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 60
    }).encode('utf-8')

    api_url = f"{ollama_url.rstrip('/')}/v1/chat/completions"
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            content = res_data["choices"][0]["message"]["content"].strip()
            return content.strip('"\'')
    except Exception as e:
        print(f"[juniper_examples] LLM API error for '{word}': {e}")
        return None

def process_word_example(word, ollama_url="http://127.0.0.1:11435", model_name="Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"):
    """
    Retrieves cached result or requests a new LLM completion, saving to disk.
    """
    cached = get_cached_data(word)
    if cached is not None:
        return cached if cached != "NONE" else None

    synsets = safe_wordnet_synsets(word)
    definition = synsets[0].definition() if synsets else None

    example = generate_ollama_example(word, definition=definition, ollama_url=ollama_url, model_name=model_name)
    if example:
        write_cached_data(word, example)
        return example
    else:
        #write_cached_data(word, "NONE") # FIXME fills up the disk with files with just NONE. not useful. harmful.
        return None

class WorkerState:
    def __init__(self):
        self.current_word = None
        self.total_processed = 0
        self.lock = threading.Lock()

worker_state = WorkerState()

def llm_example_worker(ollama_url="http://127.0.0.1:11435", model_name="Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"):
    """
    Background worker loop iterating across VOCAB to pre-cache usage examples.
    """
    print("[juniper_examples] LLM usage-example cache worker started.")
    vocab = load_vocab()

    while True:
        for word in vocab:
            cached = get_cached_data(word)

            if cached is None:
                with worker_state.lock:
                    worker_state.current_word = word

                try:
                    process_word_example(word, ollama_url=ollama_url, model_name=model_name)
                    with worker_state.lock:
                        worker_state.total_processed += 1
                except Exception as e:
                    print(f"[juniper_examples Worker] Error generating example for {word}: {e}")

                time.sleep(1.0)
        time.sleep(10.0)
