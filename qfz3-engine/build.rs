fn main() {
    // Look for vendor at the WORKSPACE ROOT
    let workspace_root = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| std::path::PathBuf::from(p).parent().unwrap().to_path_buf())
        .expect("CARGO_MANIFEST_DIR not set");

    let llama_dir = workspace_root.join("vendor/llama.cpp");
    let ggml_src  = llama_dir.join("ggml/src");
    let ggml_inc  = llama_dir.join("ggml/include");
    let llama_inc = llama_dir.join("include");

    if !ggml_src.exists() {
        panic!(
            "[Z.1] vendor/llama.cpp/ggml/src not found at {:?}!",
            llama_dir
        );
    }

    // ── Collect ALL C sources from ggml/src and ggml/src/ggml-cpu ────────────
    let mut cc = cc::Build::new();
    cc.warnings(false);
    cc.opt_level(3);
    cc.flag("-fPIC");
    cc.flag("-O3");
    cc.flag("-march=native");
    cc.define("_GNU_SOURCE", None);

    // Include paths
    cc.include(&ggml_inc);
    cc.include(&llama_inc);
    cc.include(&ggml_src);
    
    // Add CPU-specific include if it exists
    let cpu_src = ggml_src.join("ggml-cpu");
    if cpu_src.exists() {
        cc.include(&cpu_src);
    }

    // Helper to add all .c files in a directory
    fn add_c_files(dir: &std::path::Path, cc: &mut cc::Build) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("c") {
                    cc.file(&path);
                }
            }
        }
    }

    // Add files from ggml/src
    add_c_files(&ggml_src, &mut cc);
    
    // Add files from ggml/src/ggml-cpu if it exists
    if cpu_src.exists() {
        add_c_files(&cpu_src, &mut cc);
    }

    cc.compile("ggml");
    println!("cargo:rustc-link-lib=pthread");

    println!("cargo:rerun-if-changed={}", ggml_src.display());
    if cpu_src.exists() {
        println!("cargo:rerun-if-changed={}", cpu_src.display());
    }
}
