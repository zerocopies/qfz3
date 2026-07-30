fn main() {
    // qfz3-engine is a workspace member; vendor lives at the workspace root,
    // one level up from CARGO_MANIFEST_DIR.
    let workspace_root = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| std::path::PathBuf::from(p).parent().unwrap().to_path_buf())
        .expect("CARGO_MANIFEST_DIR not set");

    let llama_dir = workspace_root.join("vendor/llama.cpp");
    let ggml_src = llama_dir.join("ggml/src");
    let ggml_inc = llama_dir.join("ggml/include");
    let llama_inc = llama_dir.join("include");

    if !ggml_src.exists() {
        panic!(
            "[qfz3] vendor/llama.cpp/ggml/src not found at {:?}!",
            llama_dir
        );
    }

    // Golden's exact b3534 whitelist. Compiled as SEPARATE translation
    // units (as designed) — several files define same-named `static`
    // helpers (e.g. ggml_are_same_layout in both ggml-alloc.c and
    // ggml-backend.c), which is valid with per-file internal linkage but
    // collides if concatenated into one unit. Separate compilation avoids
    // that entirely.
    let c_sources = [
        "ggml.c",
        "ggml-alloc.c",
        "ggml-backend.c",
        "ggml-quants.c",
        "ggml-aarch64.c",
    ];

    let mut cc = cc::Build::new();
    cc.include(&ggml_src);
    cc.include(&ggml_inc);
    cc.include(&llama_inc);
    cc.flag_if_supported("-O3");
    cc.flag_if_supported("-march=native");
    cc.flag_if_supported("-DNDEBUG");
    cc.define("_GNU_SOURCE", None);
    cc.flag_if_supported("-Wno-unused-function");
    // ROOT CAUSE FIX: cc-rs enables -ffunction-sections -fdata-sections by
    // default (confirmed directly in the failing compile command). Combined
    // with Rust's --gc-sections on release links, a linker can legitimately
    // keep some functions from a compiled object while dropping others —
    // this is the confirmed explanation for every earlier "nm proves the
    // symbol is in the archive, but the linker calls it undefined" failure
    // (ggml_free/tensor_set/tensor_get resolving while their siblings in
    // the exact same file did not). Disabling function/data sections makes
    // each object one atomic unit: entirely kept or entirely dropped, never
    // partially. No archive tricks, no link-arg propagation games needed.
    cc.flag_if_supported("-fno-function-sections");
    cc.flag_if_supported("-fno-data-sections");

    for f in c_sources {
        let p = ggml_src.join(f);
        if !p.exists() {
            panic!(
                "[qfz3] expected ggml source missing: {:?} — vendor checkout is incomplete.\n\
                 Fallback: git clone --depth 1 --branch b3534 https://github.com/ggerganov/llama.cpp /tmp/b3534 && cp /tmp/b3534/ggml/src/{}  {:?}",
                p, f, ggml_src
            );
        }
        cc.file(&p);
    }

    // Plain compile — cc auto-emits standard, cross-crate-propagating
    // rustc-link-lib / rustc-link-search directives. No manual metadata,
    // no modifiers, no unity build.
    cc.compile("ggml");

    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rerun-if-changed={}", ggml_src.display());
}
