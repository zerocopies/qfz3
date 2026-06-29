fn main() {
    // Look up one level out of z1-core to find the vendor files
    let llama_dir = std::path::PathBuf::from("../vendor/llama.cpp");
    let ggml_src  = llama_dir.join("ggml/src");
    let ggml_inc  = llama_dir.join("ggml/include");
    let llama_inc = llama_dir.join("include");

    if !ggml_src.exists() {
        panic!(
            "[Z.1] ../vendor/llama.cpp/ggml/src not found!\n\
             Make sure the checkout exists at the workspace root repository."
        );
    }

    // ── GGML C sources ────────────────────────────────────────────────────────
    let c_sources = [
        "ggml.c",
        "ggml-alloc.c",
        "ggml-backend.c",
        "ggml-quants.c",
        "ggml-aarch64.c",
    ];

    let mut c_build = cc::Build::new();
    c_build
        .include(&ggml_src)
        .include(&ggml_inc)
        .include(&llama_inc)
        .flag_if_supported("-O3")
        .flag_if_supported("-march=native")  // Activates AVX2 hardware acceleration loops
        .flag_if_supported("-DNDEBUG")
        .flag_if_supported("-D_GNU_SOURCE")  // Required for CPU multi-threading affinity
        .flag_if_supported("-Wno-unused-function");

    for src in &c_sources {
        let file = ggml_src.join(src);
        if file.exists() {
            c_build.file(&file);
        } else {
            eprintln!("[Z.1] WARNING: missing ggml source {}", src);
        }
    }

    c_build.compile("ggml_c");

    println!("cargo:rustc-link-lib=static=ggml_c");
    println!("cargo:rerun-if-changed=../vendor/llama.cpp");
    println!("cargo:rerun-if-changed=build.rs");
}
