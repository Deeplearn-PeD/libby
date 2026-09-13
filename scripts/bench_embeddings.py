"""
Standalone benchmark: Ollama vs llama.cpp embedding throughput.

Compares embedding generation speed for a single model (default: embeddinggemma)
across runtimes and calling patterns:

  - Ollama /api/embeddings, one chunk per request, sequential  (Libby production path)
  - Ollama /api/embeddings, 8 concurrent request threads
  - Ollama /api/embed  (native batch endpoint), batches of 32
  - llama-server /v1/embeddings, one chunk per request, sequential
  - llama-server /v1/embeddings, array input, batches of 32
  - BF16 (unquantized) and Q8_0 GGUF weights for llama.cpp

Also checks cross-backend embedding agreement (pairwise cosine similarity) so
quality parity can be verified, not just speed.

Usage:
    python scripts/bench_embeddings.py --pdf /path/to/corpus.pdf \
        --llama-server /path/to/llama-server \
        --bf16 /path/to/embeddinggemma-BF16.gguf \
        --q8 /path/to/embeddinggemma-Q8_0.gguf

Requires: stdlib + numpy (+ pymupdf when --pdf is used). Makes no project
imports; safe to run standalone.
"""

import argparse
import concurrent.futures as cf
import json
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

OLLAMA_HOST = "http://localhost:11434"
LLAMA_HOST = "http://localhost:11435"
BATCH = 32
N_CHUNKS = 100
CHUNK_CHARS = 500
CHUNK_OVERLAP = 80
REPS = 2
WARMUP = 3
AGREEMENT_N = 20


def post_json(url: str, payload: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def ollama_embed_one(text: str, model: str) -> list[float]:
    """Mirrors libbydbot.brain.embed.DocEmbedder._generate_embedding (embed.py:795)."""
    return post_json(f"{OLLAMA_HOST}/api/embeddings", {"model": model, "prompt": text})["embedding"]


def ollama_embed_many(texts: list[str], model: str) -> list[list[float]]:
    out = post_json(f"{OLLAMA_HOST}/api/embed", {"model": model, "input": texts})
    return out["embeddings"]


def llama_embed(texts: list[str], model: str) -> list[list[float]]:
    out = post_json(f"{LLAMA_HOST}/v1/embeddings", {"model": model, "input": texts})
    return [d["embedding"] for d in out["data"]]


def load_chunks(pdf_path: str, n: int) -> list[str]:
    import fitz

    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    words = text.split()
    chunks, i = [], 0
    while i < len(words) and len(chunks) < n + 10:
        buf, j = [], i
        while j < len(words) and sum(len(w) + 1 for w in buf) < CHUNK_CHARS:
            buf.append(words[j])
            j += 1
        chunks.append(" ".join(buf))
        i += max(1, len(buf) - CHUNK_OVERLAP // 5)
    rng = random.Random(42)
    rng.shuffle(chunks)
    return chunks[:n]


class LlamaServer:
    def __init__(self, binary: str, gguf: str, threads: int):
        self.binary, self.gguf, self.threads = binary, gguf, threads
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [
                self.binary,
                "-m", self.gguf,
                "--embedding",
                "--pooling", "mean",
                "--port", "11435",
                "--threads", str(self.threads),
                "-b", "8192",
                "-ub", "2048",
                "--no-webui",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(240):
                if get_ok(f"{LLAMA_HOST}/health"):
                    return self
                time.sleep(0.5)
            raise RuntimeError("llama-server failed to become healthy")
        except BaseException:
            self.proc.terminate()
            raise

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        time.sleep(1)


def timed(texts: list[str], run_group, group_size: int, per_item: bool) -> dict:
    """Time run_group over texts in groups of group_size.

    per_item=True measures latency per single-item group (1-by-1 variants);
    otherwise latency is per group request.
    """
    run_group(texts[:WARMUP])
    latencies = []
    start = time.perf_counter()
    if per_item:
        for x in texts:
            t0 = time.perf_counter()
            run_group([x])
            latencies.append(time.perf_counter() - t0)
    else:
        for i in range(0, len(texts), group_size):
            t0 = time.perf_counter()
            run_group(texts[i : i + group_size])
            latencies.append(time.perf_counter() - t0)
    total = time.perf_counter() - start
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0]
    return {
        "total_s": round(total, 3),
        "chunks_per_s": round(len(texts) / total, 2),
        "mean_req_latency_s": round(statistics.mean(latencies), 4),
        "p95_req_latency_s": round(p95, 4),
        "n_requests": len(latencies),
    }


def run_variant(name, texts, run_group, group_size, per_item, results):
    print(f"  {name:34s} running...", flush=True)
    reps = [timed(texts, run_group, group_size, per_item) for _ in range(REPS)]
    best = max(reps, key=lambda r: r["chunks_per_s"])
    results[name] = {"reps": reps, "best": best}
    print(f"  {name:34s} {best['chunks_per_s']:8.2f} chunks/s   "
          f"total {best['total_s']:7.2f}s   p95 {best['p95_req_latency_s']:6.3f}s", flush=True)


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--llama-server", required=True)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--q8", default=None)
    ap.add_argument("--model", default="embeddinggemma")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    print(f"Loading {N_CHUNKS} x ~{CHUNK_CHARS}-char chunks from {args.pdf}", flush=True)
    chunks = load_chunks(args.pdf, N_CHUNKS)
    lens = [len(c) for c in chunks]
    print(f"  chars: mean {statistics.mean(lens):.0f}, min {min(lens)}, max {max(lens)}\n", flush=True)

    results = {"meta": {"chunks": len(chunks), "chunk_chars_mean": round(statistics.mean(lens), 1),
                        "model": args.model, "reps": REPS}}

    print("== Ollama ==", flush=True)
    run_variant(
        "ollama_1by1 (production)", chunks,
        lambda g: [ollama_embed_one(t, args.model) for t in g],
        group_size=1, per_item=True, results=results)

    def conc(group):
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda t: ollama_embed_one(t, args.model), group))

    run_variant("ollama_concurrent_8", chunks, conc,
                group_size=len(chunks), per_item=False, results=results)
    run_variant("ollama_embed_batch32", chunks, lambda g: ollama_embed_many(g, args.model),
                group_size=BATCH, per_item=False, results=results)

    print("\n== llama.cpp ==", flush=True)
    for threads in (8, 16):
        with LlamaServer(args.llama_server, args.bf16, threads):
            tag = f"t{threads}"
            run_variant(f"llamacpp_f16_1by1_{tag}", chunks,
                        lambda g: llama_embed(g, "f16"),
                        group_size=1, per_item=True, results=results)
            run_variant(f"llamacpp_f16_batch32_{tag}", chunks,
                        lambda g: llama_embed(g, "f16"),
                        group_size=BATCH, per_item=False, results=results)
        if args.q8:
            with LlamaServer(args.llama_server, args.q8, threads):
                run_variant(f"llamacpp_q8_1by1_t{threads}", chunks,
                            lambda g: llama_embed(g, "q8"),
                            group_size=1, per_item=True, results=results)
                run_variant(f"llamacpp_q8_batch32_t{threads}", chunks,
                            lambda g: llama_embed(g, "q8"),
                            group_size=BATCH, per_item=False, results=results)

    print(f"\n== Cross-backend agreement (cosine, first {AGREEMENT_N} chunks) ==", flush=True)
    sub = chunks[:AGREEMENT_N]
    embs = {"ollama": [ollama_embed_one(c, args.model) for c in sub]}
    with LlamaServer(args.llama_server, args.bf16, 8):
        embs["llamacpp_f16"] = llama_embed(sub, "f16")
    if args.q8:
        with LlamaServer(args.llama_server, args.q8, 8):
            embs["llamacpp_q8"] = llama_embed(sub, "q8")
    agreement = {}
    keys = list(embs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sims = [cosine(embs[keys[i]][k], embs[keys[j]][k]) for k in range(len(sub))]
            pair = f"{keys[i]} vs {keys[j]}"
            agreement[pair] = {"mean": round(statistics.mean(sims), 5), "min": round(min(sims), 5)}
            print(f"  {pair:38s} mean {statistics.mean(sims):.5f}  min {min(sims):.5f}", flush=True)
    results["agreement"] = agreement

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
