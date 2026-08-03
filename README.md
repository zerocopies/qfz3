# qfz3 — Zero-Copy Local LLM Inference Engine

[![Rust](https://img.shields.io/badge/rust-1.75+-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![CI](https://github.com/zerocopies/qfz3/actions/workflows/build.yml/badge.svg)](https://github.com/zerocopies/qfz3/actions/workflows/build.yml)

**A literally zero-copy local LLM inference engine — built from scratch in Rust.**

qfz3 is the project; **`qfz3-engine`** is the library/binary crate it ships (renamed from `qfz3` — see [Project structure](#project-structure)). It's a custom inference engine for running quantized large language models locally, with **no dependency on llama.cpp's high-level API**. It implements its own compute graph, KV cache management, batched prefill, and autoregressive decode loop directly on top of ggml primitives.

Built and maintained by [Zero Copies](https://github.com/zerocopies) — engineered for resource-constrained hardware without compromising on correctness.

> **Perfect for:** Laptops, servers, edge devices, or any system where memory bandwidth matters, and where running inference on-device (not in someone else's cloud) is the point.

---

## What makes it different

Most local inference tools are **wrappers around llama.cpp**. qfz3 is **not**. It owns its entire forward pass — from GGUF weight loading through memory-mapped tensors, to the compute graph, KV cache, and token sampling.

### Key design decisions

- **Zero-copy weight loading** — model weights are memory-mapped directly from disk. No heap allocation for weights, ever.
- **Contiguous KV cache** — a single backend-allocated buffer for all layers; K/V tensors are views into fixed offsets. Zero-copy writes via `ggml_cpy`.
- **Batched prefill, streaming decode** — all prompt tokens processed in a single graph pass with a causal mask; `Engine::generate_streaming` yields tokens as they're produced instead of returning one complete string.
- **Arch-aware chat templating** — instruct format (ChatML vs. Llama-3 headers) is detected from what special tokens actually exist in a model's vocab, not hardcoded per-family. Same principle applies throughout: vocab size, rope dimensions, and EOS tokens are all resolved from the model's own metadata rather than assumed. This is also what let Qwen2 support land without per-family special-casing (see [Models supported](#models-supported)).
- **Real multi-turn with sliding-window eviction** — conversation state (KV cache + turn count) persists across turns; follow-up messages append rather than reprocessing history from scratch. When a conversation grows past the context window, the oldest turns are evicted and the sequence is rebuilt from what remains, instead of hard-failing once the window fills up.
- **Thread count pinned to physical cores** — the ggml CPU backend is pinned to the machine's physical core count rather than left at ggml's default (which can pick logical/hyperthreaded count). Benchmarked ~6-9% faster decode in back-to-back A/B runs; see `graph.rs` for the measurement.

---

## Models supported

| Model | Status |
|-------|--------|
| **Llama 3.1 / 3.2 (Q4_K_M)** | ✅ Working — single-shot and multi-turn |
| **Qwen2.5-Coder (1.5B / 3B)** | ✅ Working — single-shot and multi-turn |
| Phi-3-mini | 🔧 Not yet supported (fused QKV split needed) |

Qwen2 support required fixing a chain of llama-only assumptions that don't hold across architectures: vocab size, rope dimension count and mode (NEOX vs. normal), QKV attention biases, and the prefill causal mask. All of these now resolve from GGUF metadata / vocab contents rather than being hardcoded, so adding the next architecture should be materially easier than this one was.

---

## Quick start

### Requirements

- Rust 1.75+
- Linux (tested on Linux Mint)
- A GGUF model file (e.g., from [Hugging Face](https://huggingface.co/models?other=gguf))

### Build

```bash
git clone https://github.com/zerocopies/qfz3
cd qfz3
cargo build --release --manifest-path qfz3-engine/Cargo.toml
```

The binary is named after the crate: `target/release/qfz3-engine` (not `qfz3` — see the package rename note above).

### Run

```bash
# Single-shot
target/release/qfz3-engine -m /path/to/model.gguf -p "Your prompt here"

# Interactive multi-turn chat
target/release/qfz3-engine -m /path/to/model.gguf --chat
```

### Chat commands

```
/reset   — clear conversation memory and KV cache
/quit    — exit
```

### CLI flags

```
-m, --model <path>       Path to GGUF model file
-p, --prompt <text>      Single-shot prompt (non-interactive)
-c, --chat               Interactive multi-turn chat REPL
-b, --bench               Run benchmark harness
-n, --max-tokens <N>     Max tokens to generate [default: 512]
-t, --temperature <f>    Sampling temperature [default: 0.7]
    --top-p <f>          Nucleus sampling threshold [default: 0.9]
    --context-len <N>    KV cache context length [default: 4096]
```

---

## Project structure

```
qfz3/
├── qfz3-engine/
│   ├── src/
│   │   ├── graph.rs       — compute graph (ForwardPass): prefill, decode, attention
│   │   ├── generate.rs    — sampling loop, chat templating, multi-turn session logic
│   │   ├── engine.rs       — public Engine API (load, generate_streaming, generate_rich, reset), sliding-window turn eviction
│   │   ├── loader.rs      — GGUF loader + zero-copy mmap
│   │   ├── tokenizer.rs   — BPE tokenizer, vocab-derived special tokens
│   │   ├── logits.rs      — sampling (temperature, top-p, repetition penalty)
│   │   ├── gguf.rs        — GGUF format parser
│   │   ├── ggml_ffi.rs    — raw ggml bindings
│   │   └── main.rs        — CLI entrypoint
│   └── Cargo.toml
├── vendor/llama.cpp/      — vendored ggml (compute backend, not llama.cpp's model code)
├── LICENSE
└── README.md
```

qfz3 is one component of the **Zero Copies** stack. [buzz-cli](https://github.com/zerocopies/buzz-cli) is the router/TUI product built on top of it, handling local-vs-cloud dispatch, sensitivity-based routing policy, and the interactive chat interface end users actually see. buzz-cli pulls in `qfz3-engine` as a git dependency for local inference — it doesn't vendor or reimplement any of this.

---

## Testing & CI

152 tests currently pass across the workspace (unit + stress tests for the tokenizer, mapper, and logits/sampling code). Every push and PR runs, via GitHub Actions:

```bash
cargo build --release   # all platforms: ubuntu, macos, windows
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

Clippy and the full test suite are enforced on every push — previously CI only built the workspace, so regressions in test coverage or lint cleanliness could land silently.

To run the same checks locally:

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

### Benchmarking

`thread_bench.py` is the A/B harness that produced the thread-pinning number above: it toggles the pinned block in `graph.rs` in and out, rebuilds, runs N times per configuration, and reports mean decode tok/s.

```bash
python3 thread_bench.py [n_runs]   # defaults to 5 runs per configuration
```

Requires a local GGUF model — defaults to `~/qfz3/ai_playground/qwen2.5-coder-1.5b-q4_K_M.gguf`; override with `QFZ3_BENCH_MODEL=/path/to/model.gguf` if that file isn't present on your machine.

---

## Roadmap

- [ ] Phi-3 fused QKV support
- [ ] Persistent decode graph (currently rebuilds per token as an interim correctness measure; `ggml_concat`-based fix scoped)
- [ ] Batched decode (multiple sequences)
- [ ] macOS & Windows support (CI builds all three; only Linux is actively used/tested day-to-day)

---

## Contributing

Contributions are welcome — model architecture support, performance work, documentation, and tests are all valuable. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, and [GOVERNANCE.md](GOVERNANCE.md) for how decisions get made.

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

*Part of the [Zero Copies](https://github.com/zerocopies) product family.
