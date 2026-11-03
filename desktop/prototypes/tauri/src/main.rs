#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Deserialize;
use std::{fs::{self, OpenOptions}, io::{BufRead, BufReader, Write}, path::PathBuf, process::{Child, Command, Stdio}, sync::{Arc, Mutex}, time::{SystemTime, UNIX_EPOCH}};
use std::os::windows::process::CommandExt;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder, WindowEvent, menu::{Menu, MenuItem}, tray::{TrayIconBuilder, TrayIconEvent, MouseButton, MouseButtonState}, webview::DownloadEvent};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Config { python: String, controller: String, trial_root: String, trial_id: String, webview_profile: String, debug_port: u16 }

fn event(root: &str, name: &str) {
    if let Ok(mut stream) = OpenOptions::new().create(true).append(true).open(PathBuf::from(root).join("host-events.jsonl")) {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        let _ = writeln!(stream, "{}", serde_json::json!({"event": name, "utcSeconds": stamp}));
    }
}

fn main() {
    if let Err(error) = run() {
        let location = std::env::temp_dir().join("labviz-d1-tauri-error.txt");
        let _ = fs::write(location, error.to_string());
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::args().nth(1).map(PathBuf::from).unwrap_or(std::env::current_exe()?.with_file_name("launcher.json"));
    let config: Config = serde_json::from_slice(&fs::read(&config_path)?)?;
    event(&config.trial_root, "host-start");
    std::env::set_var("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", format!("--remote-debugging-port={} --remote-debugging-address=127.0.0.1", config.debug_port));
    let mut controller = Command::new(&config.python).arg(&config.controller).arg(&config_path).creation_flags(0x08000000).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
    let mut line = String::new();
    BufReader::new(controller.stdout.take().ok_or("Missing controller output")?).read_line(&mut line)?;
    let response: serde_json::Value = serde_json::from_str(&line)?;
    let url: tauri::Url = response["url"].as_str().ok_or("Missing bootstrap URL")?.parse()?;
    if url.scheme() != "http" || url.host_str() != Some("127.0.0.1") || url.port().is_none() { return Err("Invalid local service URL".into()); }
    event(&config.trial_root, "services-ready");
    let child: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(Some(controller)));
    let quit_child = child.clone();
    let root = config.trial_root.clone();
    let menu_root = root.clone();
    let command_child = child.clone();
    tauri::Builder::default()
        .setup(move |app| {
            let load_root = root.clone();
            let nav_origin = url.origin().ascii_serialization();
            let window = WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url.clone()))
                .title("LabViz D1 Tauri").inner_size(1280.0, 900.0)
                .data_directory(PathBuf::from(&config.webview_profile))
                .on_navigation(move |target| target.origin().ascii_serialization() == nav_origin)
                .on_page_load(move |_, _| event(&load_root, "page-load"))
                .on_download(|_, download| {
                    if let DownloadEvent::Requested { destination, .. } = download {
                        let name = destination.file_name().and_then(|s| s.to_str()).unwrap_or("LabViz-export");
                        if let Some(path) = rfd::FileDialog::new().set_file_name(name).save_file() { *destination = path; true } else { false }
                    } else { true }
                })
                .build()?;
            let close_root = root.clone();
            let hide_window = window.clone();
            window.on_window_event(move |action| {
                if let WindowEvent::CloseRequested { api, .. } = action { api.prevent_close(); let _ = hide_window.hide(); event(&close_root, "hidden-to-tray"); }
            });
            let open = MenuItem::with_id(app, "open", "Open LabViz", true, None::<&str>)?;
            let exit = MenuItem::with_id(app, "exit", "Exit LabViz", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open, &exit])?;
            let icon = tauri::image::Image::from_bytes(include_bytes!("../../../../V2.0/assets/labviz-logo.ico"))?;
            let tray_root = root.clone();
            TrayIconBuilder::new().icon(icon).tooltip("LabViz D1 Tauri").menu(&menu).show_menu_on_left_click(false)
                .on_tray_icon_event(move |tray, action| {
                    if matches!(action, TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. }) {
                        if let Some(window) = tray.app_handle().get_webview_window("main") { let _ = window.show(); let _ = window.set_focus(); event(&tray_root, "restored-from-tray"); }
                    }
                })
                .build(app)?;
            event(&root, "window-created");
            let command_app = app.handle().clone();
            let command_root = root.clone();
            std::thread::spawn(move || {
                let mut last = String::new();
                loop {
                    if let Ok(bytes) = fs::read(PathBuf::from(&command_root).join("host-command.json")) {
                        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                            let id = value["id"].as_str().unwrap_or("");
                            if value["trialId"].as_str() == Some(&config.trial_id) && !id.is_empty() && id != last {
                                last = id.to_string();
                                if let Some(window) = command_app.get_webview_window("main") {
                                    match value["action"].as_str() {
                                        Some("hide") => { let _ = window.hide(); event(&command_root, "benchmark-hidden"); },
                                        Some("show") => { let _ = window.show(); let _ = window.set_focus(); event(&command_root, "benchmark-restored"); },
                                        Some("exit") => {
                                            event(&command_root, "benchmark-exit");
                                            if let Some(mut controller) = command_child.lock().unwrap().take() {
                                                if let Some(mut input) = controller.stdin.take() { let _ = input.write_all(b"stop\n"); }
                                                let _ = controller.wait();
                                            }
                                            command_app.exit(0);
                                            return;
                                        },
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            });
            Ok(())
        })
        .on_menu_event(move |app, action| match action.id().as_ref() {
            "open" => { if let Some(window) = app.get_webview_window("main") { let _ = window.show(); let _ = window.set_focus(); event(&menu_root, "restored-from-tray"); } },
            "exit" => {
                event(&menu_root, "explicit-exit");
                if let Some(mut controller) = quit_child.lock().unwrap().take() {
                    if let Some(mut input) = controller.stdin.take() { let _ = input.write_all(b"stop\n"); }
                    let _ = controller.wait();
                }
                app.exit(0);
            },
            _ => {}
        })
        .run(tauri::generate_context!())?;
    Ok(())
}
