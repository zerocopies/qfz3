fn main() {
    // qfz3-engine is a workspace member; vendor lives at the workspace root,
    // one level up from CARGO_MANIFEST_DIR.
    let workspace_root = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| std::path::PathBuf::from(p).parent().unwrap().to_path_buf())
        .expect("CARGO_MANIFEST_DIR not set");

    let llama_dir = workspace_root.join("vendor/llama.cpp");
    let ggml_src  = llama_dir.join("ggml/src");
    let ggml_inc  = llama_dir.join("ggml/include");
    let llama_inc = llama_dir.join("include");

    if !ggml_src.exists() {
        panic!("[qfz3] vendor/llama.cpp/ggml/src not found at {:?}!", llama_dir);
    }

    // Golden's exact b3534 whitelist — NOT a directory scan.
    let c_sources = [
        "ggml.c",
        "ggml-alloc.c",
        "ggml-backend.c",
        "ggml-quants.c",
        "ggml-aarch64.c",
    ];

    let mut cc = cc::Build::new();
    cc.cargo_metadata(false);
    cc.include(&ggml_src);
    cc.include(&ggml_inc);
    cc.include(&llama_inc);
    cc.flag_if_supported("-O3");
    cc.flag_if_supported("-march=native");
    cc.flag_if_supported("-DNDEBUG");
    cc.define("_GNU_SOURCE", None);
    cc.flag_if_supported("-Wno-unused-function");

    let out_dir = std::path::PathBuf::from(
        std::env::var("OUT_DIR").expect("OUT_DIR not set"),
    );

    // Compile each source to a standalone .o and link every object DIRECTLY
    // (not archived into a .a). Two prior attempts to fix a static-archive
    // scan-order problem (double link-lib directive, then the +whole-archive
    // modifier) both failed to change the actual linker command line —
    // rather than keep guessing at Cargo metadata syntax, this sidesteps the
    // whole archive-selection mechanism: raw .o files passed on the link
    // line are ALWAYS included in full, unconditionally, by construction.
    let compiler = cc.get_compiler();
    let mut obj_paths: Vec<std::path::PathBuf> = Vec::new();

    for f in c_sources {
        let src_path = ggml_src.join(f);
        if !src_path.exists() {
            panic!(
                "[qfz3] expected ggml source missing: {:?} — vendor checkout is incomplete.\n\
                 Fallback: git clone --depth 1 --branch b3534 https://github.com/ggerganov/llama.cpp /tmp/b3534 && cp /tmp/b3534/ggml/src/{}  {:?}",
                src_path, f, ggml_src
            );
        }
        let obj_path = out_dir.join(format!("{}.o", f));
        let mut cmd = compiler.to_command();
        cmd.arg("-c").arg(&src_path).arg("-o").arg(&obj_path);
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("[qfz3] failed to invoke compiler for {}: {}", f, e));
        if !status.success() {
            panic!("[qfz3] compilation failed for {} (exit: {:?})", f, status.code());
        }
        obj_paths.push(obj_path);
    }

    for obj in &obj_paths {
        println!("cargo:rustc-link-arg={}", obj.display());
    }
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rerun-if-changed={}", ggml_src.display());
}
