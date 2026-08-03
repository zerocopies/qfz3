#!/usr/bin/env python3
"""
Triple Mode Test: Normal vs Super Compressed vs Flush
Fixed: Proper lambda signatures, corrected API URLs

Standalone exploratory tool comparing prompting strategies against external
cloud LLM APIs — not part of the shipped qfz3-engine, and not covered by
`cargo test --workspace`.
"""

import os, sys, json, time, urllib.request, gzip

# ── Load keys ──
KEY_FILE = os.path.expanduser("~/.config/qfz3/keys.env")
if os.path.exists(KEY_FILE):
    with open(KEY_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                val = val.strip().strip('"').strip("'")
                if val:
                    os.environ.setdefault(key.strip(), val)

# ── CORRECTED PROVIDERS (tested endpoints) ──
PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.1-8b-instant",
        "key_env": "GROQ_API_KEY",
        "auth": ("Authorization", "Bearer {key}"),
        "format": "openai",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "google/gemma-2-9b-it:free",
        "key_env": "OPENROUTER_API_KEY",
        "auth": ("Authorization", "Bearer {key}"),
        "format": "openai",
        "extra_headers": [
            ("HTTP-Referer", "https://qfz3.local"),
            ("X-Title", "qfz3-triple-test"),
        ],
    },
    "together": {
        "url": "https://api.together.xyz/v1/chat/completions",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "key_env": "TOGETHER_API_KEY",
        "auth": ("Authorization", "Bearer {key}"),
        "format": "openai",
    },
    "deepinfra": {
        "url": "https://api.deepinfra.com/v1/openai/chat/completions",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "key_env": "DEEPINFRA_API_KEY",
        "auth": ("Authorization", "Bearer {key}"),
        "format": "openai",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
        "model": "gemini-2.0-flash",
        "key_env": "GEMINI_API_KEY",
        "format": "gemini",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-haiku-latest",
        "key_env": "ANTHROPIC_API_KEY",
        "auth": ("x-api-key", "{key}"),
        "extra_headers": [("anthropic-version", "2023-06-01")],
        "format": "anthropic",
    },
    "replicate": {
        "url": "https://api.replicate.com/v1/predictions",
        "model": "meta/meta-llama-3.1-8b-instruct",
        "key_env": "REPLICATE_API_KEY",
        "auth": ("Authorization", "Bearer {key}"),
        "format": "replicate",
    },
}

PRIORITY = ["gemini", "openrouter", "together", "deepinfra", "anthropic", "openai", "groq", "replicate"]

def get_available_providers():
    available = []
    for name in PRIORITY:
        cfg = PROVIDERS.get(name, {})
        key = os.environ.get(cfg.get("key_env", ""), "")
        if key:
            available.append(name)
    return available

def select_provider(preferred=None):
    if preferred and preferred in PROVIDERS:
        key = os.environ.get(PROVIDERS[preferred].get("key_env", ""), "")
        if key:
            return preferred, PROVIDERS[preferred]
        print(f"[WARN] No key for {preferred}, auto-selecting")
    for name in PRIORITY:
        cfg = PROVIDERS[name]
        key = os.environ.get(cfg["key_env"], "")
        if key:
            return name, cfg
    print("[ERROR] No API keys found")
    sys.exit(1)

# FIXED: Accept **kwargs
def call_provider(name, cfg, prompt, max_tokens=2048, temperature=0.7, system=None):
    key = os.environ.get(cfg["key_env"], "")
    fmt = cfg.get("format", "openai")

    if fmt == "anthropic":
        payload = json.dumps({
            "model": cfg["model"], "max_tokens": max_tokens,
            "temperature": temperature, "system": system or "",
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        url = cfg["url"]

    elif fmt == "gemini":
        text = (system or "") + "\n" + prompt if system else prompt
        payload = json.dumps({
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }).encode()
        url = cfg["url"].format(key=key)

    elif fmt == "replicate":
        # Replicate requires polling
        create_url = cfg["url"]
        predict_payload = json.dumps({
            "input": {"prompt": (system or "") + "\n" + prompt if system else prompt,
                      "max_tokens": max_tokens, "temperature": temperature},
        }).encode()
        req = urllib.request.Request(create_url, data=predict_payload, method="POST")
        req.add_header("Content-Type", "application/json")
        h, v = cfg["auth"]
        req.add_header(h, v.format(key=key))
        req.add_header("Prefer", "wait")
        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                latency = (time.time() - start) * 1000
                content = data.get("output", "")
                if isinstance(content, list):
                    content = "".join(content)
                return content, latency, 0, 0
        except Exception as e:
            return f"ERROR: {e}", 0, 0, 0

    else:  # openai-compatible
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = json.dumps({
            "model": cfg["model"], "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        }).encode()
        url = cfg["url"]

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if "auth" in cfg:
        h, v = cfg["auth"]
        req.add_header(h, v.format(key=key))
    if "extra_headers" in cfg:
        for h, v in cfg["extra_headers"]:
            req.add_header(h, v)

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            latency = (time.time() - start) * 1000

            if fmt == "anthropic":
                content = data["content"][0]["text"]
                tokens_in = data["usage"]["input_tokens"]
                tokens_out = data["usage"]["output_tokens"]
            elif fmt == "gemini":
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                tokens_in = data.get("usageMetadata", {}).get("promptTokenCount", 0)
                tokens_out = data.get("usageMetadata", {}).get("candidatesTokenCount", 0)
            else:
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens_in = usage.get("prompt_tokens", 0)
                tokens_out = usage.get("completion_tokens", 0)
            return content, latency, tokens_in, tokens_out
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        return f"HTTP {e.code}: {err_body[:200]}", 0, 0, 0
    except Exception as e:
        return f"ERROR: {e}", 0, 0, 0

# TEST PROMPT
TEST_PROMPT = """Design a concurrent inference router in Rust:

1. Accept HTTP requests on port 7474
2. Route between local GGUF and cloud APIs
3. Track token usage and cost
4. Add compression layer
5. Handle backpressure
6. Stream responses

Use tokio, tiny_http/axum, <50MB RAM, <100ms routing, 3+ concurrent."""

def flush_mode(call_fn, query, context, max_tokens=2048):
    chunk_size = 2000
    chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
    print(f"  [FLUSH] Split into {len(chunks)} chunks")
    summaries, total_tok, total_time = [], 0, 0
    for i, chunk in enumerate(chunks):
        sp = f"Summarize in under 100 words, keeping key details:\n\n{chunk}"
        # FIXED: use positional args
        s, l, ti, to = call_fn(sp, 150, 0.3)
        total_tok += ti + to; total_time += l; summaries.append(s)
        print(f"  [FLUSH] Chunk {i+1}/{len(chunks)}: {ti}+{to} tok, {l:.0f}ms")
    combined = "\n".join(summaries)
    fp = f"Based on this:\n\n{combined}\n\nAnswer:\n{query}"
    # FIXED: use positional args
    a, l, ti, to = call_fn(fp, max_tokens, 0.7)
    total_tok += ti + to; total_time += l
    return a, total_time, total_tok, ti, to

def run_single_test(call_fn, provider_name):
    results = {}

    # MODE 1: Normal
    print("\n" + "━" * 50)
    print(f"MODE 1: NORMAL ({provider_name})")
    print("━" * 50)
    # FIXED: use positional args
    a, l, ti, to = call_fn(TEST_PROMPT, 2048, 0.7)
    results["normal"] = {"answer": a, "latency": l, "in": ti, "out": to, "total": ti+to}
    print(f"  Latency: {l:.0f}ms")
    print(f"  Tokens: {ti} in + {to} out = {ti+to}")
    print(f"  Response: {len(a)} chars")
    print(f"  Preview: {a[:150]}...")

    # MODE 2: Compressed
    print("\n" + "━" * 50)
    print(f"MODE 2: COMPRESSED ({provider_name})")
    print("━" * 50)
    cs = time.time()
    compressed = gzip.compress(TEST_PROMPT.encode(), compresslevel=9)
    decompressed = gzip.decompress(compressed).decode()
    ct = (time.time() - cs) * 1000
    ratio = len(TEST_PROMPT) / len(compressed)
    verify = "PASS" if decompressed == TEST_PROMPT else "FAIL"
    print(f"  Original: {len(TEST_PROMPT)}B → {len(compressed)}B")
    print(f"  Ratio: {ratio:.2f}x | Time: {ct:.2f}ms | Verify: {verify}")
    a2, l2, ti2, to2 = call_fn(decompressed, 2048, 0.7)
    results["compressed"] = {"answer": a2, "latency": l2+ct, "in": ti2, "out": to2,
        "total": ti2+to2, "size": len(compressed), "ratio": ratio}
    print(f"  API: {l2:.0f}ms (+{ct:.2f}ms)")
    print(f"  Tokens: {ti2} in + {to2} out")
    print(f"  Response: {len(a2)} chars")
    print(f"  Preview: {a2[:150]}...")

    # MODE 3: Flush
    print("\n" + "━" * 50)
    print(f"MODE 3: FLUSH ({provider_name})")
    print("━" * 50)
    large_context = "\n\n".join([TEST_PROMPT] * 5)
    print(f"  Context: {len(large_context)} chars")
    a3, l3, total_tok, ti3, to3 = flush_mode(call_fn, TEST_PROMPT, large_context)
    results["flush"] = {"answer": a3, "latency": l3, "in": ti3, "out": to3, "total": total_tok}
    print(f"  Total latency: {l3:.0f}ms")
    print(f"  Total tokens: {total_tok}")
    print(f"  Response: {len(a3)} chars")
    print(f"  Preview: {a3[:150]}...")

    # Comparison
    print("\n" + "=" * 60)
    print(f"CMP — {provider_name}")
    print("=" * 60)
    print(f"{'Metric':<20} {'Normal':>12} {'Compressed':>12} {'Flush':>12}")
    print("-" * 60)
    print(f"{'Latency(ms)':<20} {results['normal']['latency']:>12.0f} {results['compressed']['latency']:>12.0f} {results['flush']['latency']:>12.0f}")
    print(f"{'Tokens in':<20} {results['normal']['in']:>12} {results['compressed']['in']:>12} {results['flush']['in']:>12}")
    print(f"{'Tokens out':<20} {results['normal']['out']:>12} {results['compressed']['out']:>12} {results['flush']['out']:>12}")
    print(f"{'Total tokens':<20} {results['normal']['total']:>12} {results['compressed']['total']:>12} {results['flush']['total']:>12}")
    print("-" * 60)

    fname = f"/tmp/triple_{provider_name}.txt"
    with open(fname, "w") as f:
        f.write(f"NORMAL:\n{results['normal']['answer']}\n\n")
        f.write(f"COMPRESSED:\n{results['compressed']['answer']}\n\n")
        f.write(f"FLUSH:\n{results['flush']['answer']}\n")
    print(f"\nSaved to {fname}")
    return results

def main():
    available = get_available_providers()
    print("=" * 70)
    print("TRIPLE MODE TEST")
    print("=" * 70)
    print(f"Providers: {', '.join(available) or 'NONE'}")
    print(f"Prompt: {len(TEST_PROMPT)} chars")

    # Auto-select first available
    pname, cfg = select_provider(None)
    # FIXED: call_provider takes named args
    cf = lambda p, mt=2048, t=0.7, s=None: call_provider(pname, cfg, p, mt, t, s)
    run_single_test(cf, pname)

if __name__ == "__main__":
    main()
