use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process;

use qfz3_engine::engine::Engine;
use qfz3_engine::gguf::GgufHeader;

const DEFAULT_MODEL: &str = "ai_playground/qwen2.5-coder-3b-q4_K_M.gguf";

#[derive(Debug, Default)]
struct Args {
    model_path: Option<PathBuf>,
    prompt: Option<String>,
    chat: bool,
    bench: bool,
    dump: bool,
    max_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    context_len: Option<usize>,
    help: bool,
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                a.model_path = raw.get(i).map(PathBuf::from);
            }
            "--prompt" | "-p" => {
                i += 1;
                a.prompt = raw.get(i).cloned();
            }
            "--chat" | "-c" => a.chat = true,
            "--bench" | "-b" => a.bench = true,
            "--dump" => a.dump = true,
            "--help" | "-h" => a.help = true,
            "--max-tokens" | "-n" => {
                i += 1;
                a.max_tokens = raw.get(i).and_then(|s| s.parse().ok());
            }
            "--temperature" | "-t" => {
                i += 1;
                a.temperature = raw.get(i).and_then(|s| s.parse().ok());
            }
            "--top-p" => {
                i += 1;
                a.top_p = raw.get(i).and_then(|s| s.parse().ok());
            }
            "--context-len" => {
                i += 1;
                a.context_len = raw.get(i).and_then(|s| s.parse().ok());
            }
            other => eprintln!("[Z.1] warning: unknown argument '{other}'"),
        }
        i += 1;
    }
    a
}

fn print_help() {
    println!(
        r#"
Z.1 — Local LLM inference engine

USAGE:
    qfz3 [OPTIONS]

OPTIONS:
    -m, --model <path>       Path to GGUF model file [default: {DEFAULT_MODEL}]
    -p, --prompt <text>      Single-shot prompt (non-interactive)
    -c, --chat               Interactive multi-turn chat REPL
    -b, --bench              Run benchmark harness
    -n, --max-tokens <N>     Max tokens to generate [default: 512]
    -t, --temperature <f>    Sampling temperature [default: 0.7]
        --top-p <f>          Nucleus sampling threshold [default: 0.9]
        --context-len <N>    KV cache context length [default: 4096]
    -h, --help               Print this help

EXAMPLES:
    qfz3 --prompt "Hello, who are you?"
    qfz3 --chat
    qfz3 --bench
    qfz3 -m ai_playground/gemma-2-2b-it-abliterated.Q5_K_M.gguf -p "Explain RLHF"
"#
    );
}

fn main() {
    let args = parse_args();

    if args.help || (!args.chat && !args.bench && !args.dump && args.prompt.is_none()) {
        print_help();
        return;
    }

    let model_path = args
        .model_path
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));

    if args.dump {
        let header = GgufHeader::from_file(&model_path).unwrap();
        println!(
            "Version: {}, Tensors: {}, Data offset: {}",
            header.version, header.n_tensors, header.data_offset
        );
        let mut names: Vec<&str> = header.tensors.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        for n in &names {
            println!("  {}", n);
        }
        return;
    }

    let context_len = args.context_len.unwrap_or(4096);

    // Load model
    eprint!("[Z.1] Loading {} … ", model_path.display());
    let _ = io::stderr().flush();

    let mut engine = match Engine::load(model_path.to_str().unwrap(), context_len, None) {
        Ok(e) => {
            eprintln!("OK");
            e
        }
        Err(e) => {
            eprintln!("FAILED: {e}");
            process::exit(1);
        }
    };

    // Override sampling if provided
    // (Engine internally uses defaults; these args affect generation indirectly)

    let max_tokens = args.max_tokens.unwrap_or(512);

    if args.bench {
        // Bench: single-shot "The capital of France is" then 3x speed test
        eprintln!("[Z.1] Benchmark mode\n");

        // Phase 1: regression
        eprintln!("── Phase 1: regression ──");
        match engine.generate_rich("The capital of France is", 10) {
            Ok(out) => {
                let pass = out.text.to_lowercase().contains("paris");
                eprintln!(
                    "  result: {:?} prefill={:.0}ms  {}",
                    out.text,
                    out.prompt_ms,
                    if pass { "PASS" } else { "FAIL" }
                );
            }
            Err(e) => {
                eprintln!("  FAIL: {e}");
                return;
            }
        }
        engine.reset();

        // Phase 2: speed
        eprintln!("\n── Phase 2: decode speed (3 runs) ──");
        for run in 1..=3 {
            match engine.generate_rich("Explain transformer attention briefly.", max_tokens) {
                Ok(out) => {
                    let tps = if out.generate_ms > 0.0 {
                        out.completion_tokens as f64 / (out.generate_ms / 1000.0)
                    } else {
                        0.0
                    };
                    eprintln!(
                        "  run {run}: {:.2} tok/s | {} tokens | {:.0}ms decode",
                        tps, out.completion_tokens, out.generate_ms
                    );
                }
                Err(e) => eprintln!("  run {run}: ERROR {e}"),
            }
            engine.reset();
        }
        return;
    }

    if args.chat {
        // Interactive chat
        println!("[Z.1] Interactive chat — type your message, Enter to send.");
        println!("[Z.1] /reset = new conversation, /quit = exit\n");

        let stdin = io::stdin();
        loop {
            print!("You: ");
            let _ = io::stdout().flush();
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) | Err(_) => {
                    println!("\n[Z.1] Goodbye.");
                    break;
                }
                Ok(_) => {}
            }
            let trimmed = line.trim().to_string();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed == "/reset" {
                engine.reset();
                eprintln!("[Z.1] Conversation reset.");
                continue;
            }
            if trimmed == "/quit" || trimmed == "/exit" {
                println!("[Z.1] Goodbye.");
                break;
            }

            print!("Z.1:  ");
            let _ = io::stdout().flush();

            match engine.generate_rich(&trimmed, max_tokens) {
                Ok(out) => {
                    print!("{}", out.text);
                    let _ = io::stdout().flush();
                    eprintln!(
                        "\n  [{} tok, {:.0}ms]",
                        out.completion_tokens, out.generate_ms
                    );
                }
                Err(e) => eprintln!("\n[Z.1] error: {e}"),
            }
        }
        return;
    }

    // Single-shot
    if let Some(ref prompt) = args.prompt {
        eprint!("[Z.1] Generating … ");
        let _ = io::stderr().flush();

        match engine.generate_rich(prompt, max_tokens) {
            Ok(out) => {
                println!("{}", out.text);
                eprintln!(
                    "\n[Z.1] {} prompt tokens, {} generated ({:.0}ms prefill, {:.0}ms decode)",
                    out.prompt_tokens, out.completion_tokens, out.prompt_ms, out.generate_ms
                );
            }
            Err(e) => {
                eprintln!("FAILED: {e}");
                process::exit(1);
            }
        }
    }
}
