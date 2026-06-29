use std::process::Command;
use std::thread;
use std::time::Duration;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  // Spawn Buzz Router as a child process
  thread::spawn(|| {
    // Kill any existing instance
    let _ = Command::new("fuser").args(["-k", "7474/tcp"]).status();
    thread::sleep(Duration::from_millis(500));

    // Start buffer_zone
    let mut child = Command::new("/home/prp/prubbs/buffer-zone/target/release/buffer_zone")
      .env("Z1_CTX_SIZE", "2048")
      .current_dir("/home/prp/prubbs/buffer-zone")
      .spawn()
      .expect("Failed to start Buzz Router");

    // Wait for it to be ready
    thread::sleep(Duration::from_secs(2));

    // Load default model
    let _ = Command::new("curl")
      .args([
        "-s", "-X", "POST", "http://127.0.0.1:7474/load_model",
        "-H", "Content-Type: application/json",
        "-d", r#"{"path":"/home/prp/Z3-Quantum-Flow/ai_playground/Qwen2.5-Coder-3B-Instruct-abliterated-Q4_K_M.gguf"}"#,
      ])
      .status();

    // Keep the child alive
    let _ = child.wait();
  });

  tauri::Builder::default()
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
