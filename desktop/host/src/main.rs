#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod downloads;

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    os::windows::{fs::MetadataExt, process::CommandExt},
    path::{Component, Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    webview::{DownloadEvent, NewWindowResponse},
    AppHandle, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent,
};

const ID: &str = "org.labviz.desktop.preview";
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Manifest {
    format: String,
    payload: String,
    controller: String,
    candidate_id: String,
    controller_sha256: String,
    preferences_sha256: String,
    process_job_sha256: String,
    session_sha256: String,
    service_runner_sha256: String,
    maintenance_sha256: String,
    installation_sha256: String,
}
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Acceptance {
    format: String,
    id: String,
    profile_root: String,
    debug_port: Option<u16>,
    startup_timeout_seconds: Option<u64>,
    idle_seconds: Option<u64>,
    ready_probe: Option<ReadyProbe>,
}
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReadyProbe {
    kind: String,
    expected_title: Option<String>,
}
#[derive(Clone)]
struct Spec {
    payload: PathBuf,
    controller: PathBuf,
    profile: PathBuf,
    profile_base: PathBuf,
    candidate: String,
    qa: Option<(PathBuf, Acceptance)>,
}
struct State {
    args: Vec<String>,
    boot: PathBuf,
    log: PathBuf,
    phase: Mutex<String>,
    profile: Mutex<Option<PathBuf>>,
    child: Mutex<Option<Child>>,
    visible: AtomicBool,
    show_pending: AtomicBool,
    exiting: AtomicBool,
    exit_requested: AtomicBool,
    generation: AtomicU64,
    download_sequence: AtomicU64,
    main_sequence: AtomicU64,
    entry_utc_seconds: f64,
    entry_instant: Instant,
    downloads: Mutex<downloads::Downloads>,
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
fn event(state: &State, name: &str) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&state.log)
    {
        let _ = writeln!(
            file,
            "{}",
            json!({"event":name,"utcSeconds":now(),"pid":std::process::id(),"phase":*state.phase.lock().unwrap()})
        );
    }
}
fn checked(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir))
    {
        return Err("A desktop path is invalid".into());
    }
    for member in path.ancestors() {
        if let Ok(metadata) = fs::symlink_metadata(member) {
            if metadata.file_attributes() & 0x400 != 0 {
                return Err("Linked desktop paths are not supported".into());
            }
        }
    }
    Ok(if path.exists() {
        dunce::canonicalize(path).map_err(|e| e.to_string())?
    } else {
        path.to_path_buf()
    })
}
fn load(args: &[String]) -> Result<Spec, String> {
    let bootstrap = std::env::current_exe()
        .map_err(|e| e.to_string())?
        .parent()
        .unwrap()
        .to_path_buf();
    let root = installed_runtime(&bootstrap)?;
    let manifest: Manifest = serde_json::from_slice(
        &fs::read(root.join("desktop-runtime.json"))
            .map_err(|_| "The local runtime manifest is missing")?,
    )
    .map_err(|_| "The local runtime manifest is invalid")?;
    if manifest.format != "labviz-desktop-runtime-v1" {
        return Err("Unknown desktop runtime".into());
    }
    let relative = |name: &str| -> Result<PathBuf, String> {
        if Path::new(name).is_absolute()
            || Path::new(name)
                .components()
                .any(|p| matches!(p, Component::ParentDir))
        {
            return Err("Runtime paths must stay inside the application folder".into());
        }
        checked(&root.join(name))
    };
    let payload = relative(&manifest.payload)?;
    let controller = relative(&manifest.controller)?;
    let hash = |path: &Path| -> Result<String, String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(fs::read(path).map_err(|e| e.to_string())?)
        ))
    };
    if hash(&controller)? != manifest.controller_sha256
        || hash(&controller.with_file_name("preferences.py"))? != manifest.preferences_sha256
        || hash(&controller.with_file_name("process_job.py"))? != manifest.process_job_sha256
        || hash(&controller.with_file_name("session.py"))? != manifest.session_sha256
        || hash(&controller.with_file_name("service_runner.py"))? != manifest.service_runner_sha256
        || hash(&controller.with_file_name("maintenance.py"))? != manifest.maintenance_sha256
        || hash(&controller.with_file_name("installation.py"))? != manifest.installation_sha256
    {
        return Err("A desktop runtime file has changed".into());
    }
    if !payload.join("runtime/python/python.exe").is_file()
        || !payload.join("candidate-build.json").is_file()
    {
        return Err("The bundled runtime is incomplete".into());
    }
    let mut profile = PathBuf::from(
        std::env::var_os("LOCALAPPDATA").ok_or("Local application data is unavailable")?,
    )
    .join("LabVizDesktopPreview");
    let profile_base = checked(&profile)?;
    if !args.iter().any(|arg| arg == "--acceptance") {
        let selection = profile.join("profile-selection.json");
        if selection.exists() {
            let value: Value =
                serde_json::from_slice(&fs::read(checked(&selection)?).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            let copy = value["copyId"].as_str().unwrap_or("");
            if value["appId"] != ID
                || value["format"] != "labviz-desktop-selection-v1"
                || copy.len() != 32
                || !copy.bytes().all(|b| b.is_ascii_hexdigit())
            {
                return Err("The restored profile selection is invalid".into());
            }
            profile = checked(&profile.join("restored-profiles").join(copy))?;
            let receipt: Value = serde_json::from_slice(
                &fs::read(checked(&profile.join("desktop-restored-copy.json"))?)
                    .map_err(|_| "This restored copy did not finish verification")?,
            )
            .map_err(|e| e.to_string())?;
            if receipt["format"] != "labviz-desktop-restored-copy-v1"
                || receipt["appId"] != ID
                || receipt["copyId"] != copy
            {
                return Err("The restored copy identity differs".into());
            }
        }
    }
    let mut qa = None;
    if let Some(pair) = args.windows(2).find(|pair| pair[0] == "--acceptance") {
        let session = checked(Path::new(&pair[1]))?;
        let allowed =
            PathBuf::from(std::env::var_os("USERPROFILE").ok_or("User profile is unavailable")?)
                .join("LabVizD2");
        if !session.starts_with(&allowed) {
            return Err("Acceptance sessions require an owned test directory".into());
        }
        let value: Acceptance =
            serde_json::from_slice(&fs::read(&session).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if value.format != "labviz-desktop-acceptance-v1"
            || value.id.len() != 32
            || !value.id.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("Invalid acceptance identity".into());
        }
        if let Some(probe) = &value.ready_probe {
            if !matches!(probe.kind.as_str(), "home" | "figure")
                || (probe.kind == "figure"
                    && probe
                        .expected_title
                        .as_ref()
                        .is_none_or(|title| title.len() > 640))
            {
                return Err("Invalid acceptance readiness probe".into());
            }
        }
        profile = checked(Path::new(&value.profile_root))?;
        if !profile.starts_with(&allowed) {
            return Err("Acceptance data cannot use normal user profiles".into());
        }
        qa = Some((session, value));
    }
    Ok(Spec {
        payload,
        controller,
        profile: checked(&profile)?,
        profile_base: if qa.is_some() {
            checked(&profile)?
        } else {
            profile_base
        },
        candidate: manifest.candidate_id,
        qa,
    })
}
fn installed_runtime(root: &Path) -> Result<PathBuf, String> {
    let marker = root.join("desktop-installation.json");
    if !marker.exists() {
        return checked(root);
    }
    let identity: Value =
        serde_json::from_slice(&fs::read(checked(&marker)?).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if identity["format"] != "labviz-desktop-installation-v1"
        || identity["appId"] != ID
        || identity["installerId"] != "CA2B3AF4-BFB0-4541-8EA5-2E978F105D63"
    {
        return Err("The desktop installation identity differs".into());
    }
    let pointer: Value = serde_json::from_slice(
        &fs::read(checked(&root.join("current-candidate.json"))?)
            .map_err(|_| "Run setup again to finish desktop activation")?,
    )
    .map_err(|e| e.to_string())?;
    let package = pointer["packageId"].as_str().unwrap_or("");
    if pointer["appId"] != ID
        || package.len() != 64
        || !package.bytes().all(|b| b.is_ascii_hexdigit())
        || root.join("pending-candidate.json").exists()
        || root.join("upgrade.json").exists()
    {
        return Err("Desktop setup has not finished. Retry setup before opening the app.".into());
    }
    checked(&root.join("versions").join(&package[..24]))
}
fn get_state(app: &AppHandle) -> Arc<State> {
    app.state::<Arc<State>>().inner().clone()
}
#[tauri::command]
fn acceptance_editor_ready(window: tauri::WebviewWindow, signal: Value) -> Result<(), String> {
    let state = get_state(window.app_handle());
    let received = now();
    let native_elapsed = state.entry_instant.elapsed().as_secs_f64() * 1000.0;
    if window.label() != "main" {
        return Err("Readiness requires the acceptance main window".into());
    }
    let spec = load(&state.args)?;
    let (_, qa) = spec
        .qa
        .ok_or("Readiness is available only for an acceptance session")?;
    let probe = qa
        .ready_probe
        .ok_or("No acceptance readiness probe is configured")?;
    let renderer = signal["rendererUtcSeconds"].as_f64();
    let elapsed = signal["documentElapsedMs"].as_f64();
    if signal["id"].as_str() != Some(&qa.id)
        || signal["sequence"].as_u64() != Some(state.main_sequence.load(Ordering::SeqCst))
        || signal["kind"].as_str() != Some(&probe.kind)
        || !renderer.is_some_and(|v| v.is_finite() && v <= received + 1.0 && v >= received - 600.0)
        || !elapsed.is_some_and(|v| v.is_finite() && (0.0..=600000.0).contains(&v))
    {
        return Err("Acceptance readiness identity or timing differs".into());
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&state.log)
        .map_err(|e| e.to_string())?;
    writeln!(
        file,
        "{}",
        json!({"event":"acceptance-editor-ready", "utcSeconds":received,
        "pid":std::process::id(),"sequence":signal["sequence"],"kind":probe.kind,
        "rendererUtcSeconds":renderer,"documentElapsedMs":elapsed,"focused":signal["focused"],
        "nativeEntryUtcSeconds":state.entry_utc_seconds,"nativeEntryToReceiptMs":native_elapsed,
        "twoAnimationFrames":true,"transport":"guarded-tauri-ipc"})
    )
    .map_err(|e| e.to_string())
}
fn inspect_acceptance_tray(app: &AppHandle, destination: &Path) {
    let tray = app.tray_by_id("desktop");
    let rectangle = tray.as_ref().and_then(|icon| icon.rect().ok().flatten());
    let value = json!({
        "format":"labviz-native-tray-inspection-v1",
        "pid":std::process::id(),
        "registered":tray.is_some(),
        "shellRectangle":rectangle.map(|rect| format!("{rect:?}")),
        "utcSeconds":now(),
        "scope":"Read-only native tray diagnostics for an explicit acceptance session"
    });
    if let Ok(bytes) = serde_json::to_vec_pretty(&value) {
        let _ = fs::write(destination, bytes);
    }
}
fn control(app: &AppHandle, action: &str) {
    let state = get_state(app);
    if let Some(child) = state.child.lock().unwrap().as_mut() {
        if let Some(input) = child.stdin.as_mut() {
            let _ = writeln!(input, "{}", json!({"action": action}));
        }
    };
}
fn build_startup(app: &AppHandle) -> Result<(), String> {
    if app.get_webview_window("startup").is_some() || app.get_webview_window("main").is_some() {
        return Ok(());
    }
    let state = get_state(app);
    let phase = state.phase.lock().unwrap().clone();
    let mut directory = state.boot.join("webview");
    let mut debug_port = None;
    if phase == "resuming" {
        // Prewarm the editor's existing profile while its services start.
        // The final editor then shares this WebView2 environment instead of
        // creating a second one, or cold-starting it after service readiness.
        let spec = load(&state.args)?;
        directory = spec.profile.join("webview");
        debug_port = spec.qa.as_ref().and_then(|(_, qa)| qa.debug_port);
        if let Some(tray) = app.tray_by_id("desktop") {
            let _ = tray.set_tooltip(Some("LabViz · Restoring workspace / 正在恢复工作区"));
        }
    }
    std::env::remove_var("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS");
    if let Some(port) = debug_port {
        std::env::set_var(
            "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
            format!("--remote-debugging-port={port} --remote-debugging-address=127.0.0.1"),
        );
    }
    let action_app = app.clone();
    WebviewWindowBuilder::new(app, "startup", WebviewUrl::App("index.html".into()))
        .title("LabViz · Starting")
        .inner_size(600.0, 360.0)
        .resizable(false)
        .visible(state.visible.load(Ordering::SeqCst))
        .data_directory(directory)
        .on_navigation(move |url| {
            if url.scheme() == "labviz-desktop-action" {
                // Return from WebView2's navigation callback before running
                // desktop actions or a modal message loop on the UI thread.
                let action = url.host_str().unwrap_or_default().to_string();
                let handle = action_app.clone();
                std::thread::spawn(move || {
                    let next = handle.clone();
                    let _ = handle.run_on_main_thread(move || match action.as_str() {
                        "retry" => begin(&next),
                        "logs" => open_logs(&next),
                        "exit" => quit(&next, false),
                        _ => {}
                    });
                });
                false
            } else {
                url.scheme() == "tauri" || url.host_str() == Some("tauri.localhost")
            }
        })
        .build()
        .map_err(|e| e.to_string())?;
    if phase == "resuming" {
        event(&state, "resume-editor-profile-prewarmed");
    }
    Ok(())
}
fn show(app: &AppHandle) {
    let state = get_state(app);
    if state.exiting.load(Ordering::SeqCst) || state.show_pending.swap(true, Ordering::SeqCst) {
        return;
    }
    // Single-instance and tray callbacks can run inside a Windows message
    // handler. WebView2 construction there can deadlock its initialization
    // message pump. Enqueue from another thread so the native callback returns
    // before any window is created; coalesce repeated wake requests meanwhile.
    let handle = app.clone();
    std::thread::spawn(move || {
        let next = handle.clone();
        if handle
            .run_on_main_thread(move || {
                let state = get_state(&next);
                if !state.exiting.load(Ordering::SeqCst) {
                    show_window(&next);
                }
                state.show_pending.store(false, Ordering::SeqCst);
            })
            .is_err()
        {
            get_state(&handle)
                .show_pending
                .store(false, Ordering::SeqCst);
        }
    });
}
fn show_window(app: &AppHandle) {
    let state = get_state(app);
    state.visible.store(true, Ordering::SeqCst);
    control(app, "show");
    {
        let mut phase = state.phase.lock().unwrap();
        if *phase == "tray" || *phase == "checkpointing" {
            *phase = "ready".into();
        } else if *phase == "suspended" || *phase == "suspending" {
            *phase = "resuming".into();
        }
    }
    if app.get_webview_window("main").is_none() {
        let _ = build_startup(app);
    }
    if let Some(window) = app
        .get_webview_window("main")
        .or_else(|| app.get_webview_window("startup"))
    {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
    event(
        &state,
        if app.get_webview_window("main").is_some() || app.get_webview_window("startup").is_some() {
            "window-shown"
        } else {
            "resume-requested"
        },
    );
}
fn open_logs(app: &AppHandle) {
    let state = get_state(app);
    let profile = state.profile.lock().unwrap().clone();
    if let Some(root) = profile {
        let logs = root.join("logs");
        if logs.is_dir() && root.join("desktop-profile.json").is_file() {
            let _ = fs::copy(
                &state.log,
                logs.join(format!("desktop-host-{}.jsonl", std::process::id())),
            );
            let _ = open::that(logs);
            return;
        }
    }
    let _ = open::that(&state.boot);
}
fn quit(app: &AppHandle, confirmed: bool) {
    let state = get_state(app);
    if state.exiting.load(Ordering::SeqCst) {
        return;
    }
    if !confirmed {
        let running = state
            .child
            .lock()
            .unwrap()
            .as_mut()
            .is_some_and(|child| matches!(child.try_wait(), Ok(None)));
        if running {
            if state.exit_requested.load(Ordering::SeqCst) {
                let answer=rfd::MessageDialog::new().set_title("Cancel waiting? / 取消等待退出？").set_description("Tasks are still running or being checked. Cancel waiting to continue using LabViz. / 任务仍在运行或检查中。取消等待后可继续使用。").set_buttons(rfd::MessageButtons::YesNo).show();
                if answer == rfd::MessageDialogResult::Yes {
                    control(app, "exit-cancel");
                    state.exit_requested.store(false, Ordering::SeqCst);
                }
                return;
            }
            let wait = "Wait and exit / 等待后退出";
            let interrupt = "Stop now / 立即中断";
            let answer=rfd::MessageDialog::new().set_title("Exit LabViz? / 退出 LabViz？").set_description("Wait for uploads, calculations, saves and downloads to finish, or explicitly interrupt them. Unsaved edits since the last checkpoint can be discarded on exit. Closing the window keeps work running in the tray. / 可等待上传、计算、保存和下载完成，也可明确中断。退出可能丢弃上次快照后的编辑；关闭窗口则保留后台任务。").set_buttons(rfd::MessageButtons::YesNoCancelCustom(wait.into(), interrupt.into(), "Cancel / 取消".into())).show();
            if answer == rfd::MessageDialogResult::Custom(wait.into()) {
                state.exit_requested.store(true, Ordering::SeqCst);
                control(app, "exit-wait");
            } else if answer == rfd::MessageDialogResult::Custom(interrupt.into()) {
                let answer=rfd::MessageDialog::new().set_title("Interrupt active work? / 中断正在进行的工作？").set_description("Uploads, calculations and downloads may stop before completion. Unsaved edits may be lost. Continue? / 上传、计算和下载可能中途停止，未保存编辑可能丢失。仍要退出吗？").set_buttons(rfd::MessageButtons::YesNo).show();
                if answer == rfd::MessageDialogResult::Yes {
                    event(&state, "explicit-interrupt-confirmed");
                    quit(app, true);
                }
            }
            return;
        }
        if app.get_webview_window("main").is_some() {
            let answer=rfd::MessageDialog::new().set_title("Exit LabViz? / 退出 LabViz？").set_description("Local services are disconnected. Unsaved edits may be lost on exit. / 本地服务已断开，退出可能丢失尚未保存的编辑。").set_buttons(rfd::MessageButtons::YesNo).show();
            if answer != rfd::MessageDialogResult::Yes {
                return;
            }
        }
    }
    if state.exiting.swap(true, Ordering::SeqCst) {
        return;
    }
    *state.phase.lock().unwrap() = "exiting".into();
    event(&state, "exit-requested");
    let app = app.clone();
    std::thread::spawn(move || {
        if let Some(mut child) = state.child.lock().unwrap().take() {
            if let Some(mut input) = child.stdin.take() {
                let _ = input.write_all(b"stop\n");
            }
            let _ = child.wait();
        }
        event(&state, "owned-controller-stopped");
        app.exit(0);
    });
}
fn startup_message(app: &AppHandle, message: &str, failed: bool) {
    if let Some(window) = app.get_webview_window("startup") {
        let encoded = serde_json::to_string(message).unwrap();
        let _=window.eval(format!("document.getElementById('message').textContent={encoded};document.getElementById('retry').hidden={};",if failed{"false"}else{"true"}));
    }
}
fn failed(app: &AppHandle, details: String) {
    let state = get_state(app);
    if state.exiting.load(Ordering::SeqCst) {
        return;
    }
    let has_main = app.get_webview_window("main").is_some();
    *state.phase.lock().unwrap() = if has_main { "disconnected" } else { "failed" }.into();
    if let Ok(mut log) = OpenOptions::new().append(true).open(&state.log) {
        let _ = writeln!(
            log,
            "{}",
            json!({"event":"startup-or-service-failure","message":details,"utcSeconds":now()})
        );
    }
    if has_main {
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.set_title("LabViz · Connection lost");
        }
    } else {
        // Recovery failure must still offer the full Retry / logs / Exit UI.
        let _ = build_startup(app);
        startup_message(
            app,
            "LabViz could not start. You can retry or open the local logs.",
            true,
        );
    }
    event(&state, "failure-visible");
}
fn build_main(app: &AppHandle, spec: Spec, value: Value) -> Result<(), String> {
    let url: tauri::Url = value["url"]
        .as_str()
        .ok_or("Missing local URL")?
        .parse()
        .map_err(|_| "Invalid local URL")?;
    if url.scheme() != "http"
        || url.host_str() != Some("127.0.0.1")
        || url.port().is_none()
        || value["candidateId"] != spec.candidate
    {
        return Err("Local service identity differs".into());
    }
    let origin = url.origin().ascii_serialization();
    let configuration =
        json!({"origin":origin,"profileId":value["profileId"],"preferences":value["preferences"]});
    let adapter = include_str!("../../runtime/adapter.js").replace(
        "__LABVIZ_DESKTOP_CONFIGURATION__",
        &configuration.to_string(),
    );
    let nav_origin = origin.clone();
    let probe = spec.qa.as_ref().and_then(|(_, qa)| {
        qa.ready_probe
            .as_ref()
            .map(|probe| (qa.id.clone(), probe.clone()))
    });
    let state = get_state(app);
    let sequence = state.main_sequence.fetch_add(1, Ordering::SeqCst) + 1;
    let probe_script = probe.as_ref().map(|(id, probe)| {
        include_str!("acceptance-ready.js").replace(
            "__LABVIZ_ACCEPTANCE_READY__",
            &json!({"id":id,"sequence":sequence,"origin":origin,"pathname":url.path(),
                "profileId":value["profileId"],"kind":probe.kind,"expectedTitle":probe.expected_title}).to_string(),
        )
    }).unwrap_or_default();
    let new_origin = origin.clone();
    let new_app = app.clone();
    if let Some(port) = spec.qa.as_ref().and_then(|(_, qa)| qa.debug_port) {
        std::env::set_var(
            "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
            format!("--remote-debugging-port={port} --remote-debugging-address=127.0.0.1"),
        );
    }
    let visible = get_state(app).visible.load(Ordering::SeqCst);
    let window = WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
        .title("LabViz Desktop Preview")
        .inner_size(1280.0, 900.0)
        .min_inner_size(820.0, 620.0)
        .zoom_hotkeys_enabled(true)
        .visible(visible)
        .data_directory(spec.profile.join("webview"))
        .initialization_script(adapter)
        .initialization_script(probe_script)
        .on_navigation(move |target| {
            if target.origin().ascii_serialization() == nav_origin {
                true
            } else {
                if matches!(target.scheme(), "http" | "https" | "mailto") {
                    let _ = open::that(target.as_str());
                }
                false
            }
        })
        .on_new_window(move |target, _| {
            if target.origin().ascii_serialization() == new_origin {
                if let Some(window) = new_app.get_webview_window("main") {
                    let _ = window.navigate(target);
                }
            } else if matches!(target.scheme(), "http" | "https" | "mailto") {
                let _ = open::that(target.as_str());
            }
            NewWindowResponse::Deny
        })
        .on_download(|webview, download| {
            if let DownloadEvent::Requested { url, destination } = download {
                let state = get_state(webview.app_handle());
                if state.exit_requested.load(Ordering::SeqCst)
                    || state.exiting.load(Ordering::SeqCst)
                {
                    state.downloads.lock().unwrap().cancelled(url.as_str());
                    return false;
                }
                let identifier = state
                    .download_sequence
                    .fetch_add(1, Ordering::SeqCst)
                    .to_string();
                control(
                    webview.app_handle(),
                    &format!("download-start:{identifier}"),
                );
                let name = destination
                    .file_name()
                    .and_then(|v| v.to_str())
                    .unwrap_or("LabViz-export");
                let extension = destination
                    .extension()
                    .and_then(|v| v.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let mut dialog = rfd::FileDialog::new().set_file_name(name);
                if ["png", "svg", "pdf", "csv", "json", "zip"].contains(&extension.as_str()) {
                    dialog = dialog.add_filter(extension.to_uppercase(), &[extension.as_str()]);
                }
                if let Some(selected) = dialog.save_file() {
                    if !downloads::matches_extension(&selected, &extension) {
                        state.downloads.lock().unwrap().cancelled(url.as_str());
                        rfd::MessageDialog::new()
                            .set_title("File format")
                            .set_description(format!(
                                "Choose a .{extension} filename for this export."
                            ))
                            .show();
                        control(webview.app_handle(), &format!("download-end:{identifier}"));
                        return false;
                    }
                    let started = state
                        .downloads
                        .lock()
                        .unwrap()
                        .start(selected.clone(), url.to_string(), identifier.clone());
                    if !started {
                        state.downloads.lock().unwrap().cancelled(url.as_str());
                        control(webview.app_handle(), &format!("download-end:{identifier}"));
                        rfd::MessageDialog::new().set_title("Download in progress / 正在下载")
                            .set_description("Another download is using this file. Choose a different filename. / 其他下载正在使用此文件，请选择不同文件名。")
                            .show();
                        return false;
                    }
                    *destination = selected;
                    true
                } else {
                    state.downloads.lock().unwrap().cancelled(url.as_str());
                    control(webview.app_handle(), &format!("download-end:{identifier}"));
                    false
                }
            } else if let DownloadEvent::Finished {
                url, path, success,
                ..
            } = download
            {
                let state = get_state(webview.app_handle());
                let identifiers = state.downloads.lock().unwrap().finish(url.as_str(), path.as_deref());
                let ended = !identifiers.is_empty();
                let failed = !success || identifiers.len() > usize::from(path.is_some());
                for identifier in identifiers {
                    control(webview.app_handle(), &format!("download-end:{identifier}"));
                }
                if ended {
                    event(&state, if failed { "download-failed" } else { "download-terminal" });
                    if failed {
                        rfd::MessageDialog::new().set_title("Download failed / 下载失败")
                            .set_description("The export was not saved completely. Check the destination and try again. / 导出未完整保存，请检查目标位置后重试。")
                            .show();
                    }
                }
                true
            } else {
                true
            }
        })
        .build()
        .map_err(|e| e.to_string())?;
    let state = get_state(app);
    *state.phase.lock().unwrap() = "ready".into();
    if let Some(startup) = app.get_webview_window("startup") {
        let _ = startup.destroy();
    }
    if let Some(tray) = app.tray_by_id("desktop") {
        let _ = tray.set_tooltip(Some("LabViz Desktop Preview"));
    }
    if visible {
        let _ = window.set_focus();
    }
    let status = json!({"format":"labviz-desktop-host-v1","pid":std::process::id(),"profileId":value["profileId"],"origin":origin,"candidateId":spec.candidate,"automaticSuspension":true,"hostLog":state.log});
    let _ = fs::write(
        spec.profile.join("logs/desktop-host.json"),
        status.to_string(),
    );
    event(&state, "main-window-created");
    Ok(())
}
fn begin(app: &AppHandle) {
    let state = get_state(app);
    let phase = state.phase.lock().unwrap().clone();
    if state.exiting.load(Ordering::SeqCst)
        || phase == "starting"
        || phase == "ready"
        || phase == "tray"
    {
        return;
    }
    if phase == "disconnected" {
        rfd::MessageDialog::new().set_title("Local connection lost").set_description("Your editing window is still open. Record unsaved work before exiting and reopening LabViz.").show();
        return;
    }
    state.exit_requested.store(false, Ordering::SeqCst);
    *state.phase.lock().unwrap() = "starting".into();
    startup_message(app, "Starting your local workspace…", false);
    let app = app.clone();
    let generation = state.generation.fetch_add(1, Ordering::SeqCst) + 1;
    std::thread::spawn(move || {
        let result = (|| -> Result<(), String> {
            if let Some(mut previous) = state.child.lock().unwrap().take() {
                if let Some(mut input) = previous.stdin.take() {
                    let _ = input.write_all(b"stop\n");
                }
                let _ = previous.wait();
            }
            if state.exiting.load(Ordering::SeqCst) {
                return Ok(());
            }
            let spec = load(&state.args)?;
            *state.profile.lock().unwrap() = Some(spec.profile.clone());
            let session = state
                .boot
                .join(format!("session-{}-{generation}.json", std::process::id()));
            let content = json!({"format":"labviz-desktop-session-v1","profileRoot":spec.profile,"profileBase":spec.profile_base,"payloadRoot":spec.payload,"candidateId":spec.candidate,"startupTimeoutSeconds":spec.qa.as_ref().and_then(|(_,qa)|qa.startup_timeout_seconds).unwrap_or(180),"idleSeconds":spec.qa.as_ref().and_then(|(_,qa)|qa.idle_seconds).unwrap_or(45)});
            fs::write(&session, content.to_string()).map_err(|e| e.to_string())?;
            let mut child = Command::new(spec.payload.join("runtime/python/python.exe"))
                .args(["-B", "-X", "utf8"])
                .arg(&spec.controller)
                .arg(&session)
                .creation_flags(0x08000000)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| e.to_string())?;
            let output = child.stdout.take().ok_or("Missing service status pipe")?;
            *state.child.lock().unwrap() = Some(child);
            event(&state, "controller-started");
            let mut failure = "The local service controller stopped unexpectedly.".to_string();
            for line in BufReader::new(output).lines() {
                let line = line.map_err(|e| e.to_string())?;
                if line.len() > 20 * 1024 * 1024 {
                    return Err("A local status message was too large".into());
                }
                let value: Value =
                    serde_json::from_str(&line).map_err(|_| "Invalid local status message")?;
                match value["type"].as_str() {
                    Some("progress") => {
                        let next = app.clone();
                        let message = value["message"]
                            .as_str()
                            .unwrap_or("Starting LabViz")
                            .to_string();
                        let _ = app.run_on_main_thread(move || {
                            let _ = build_startup(&next);
                            startup_message(&next, &message, false);
                        });
                    }
                    Some("ready") => {
                        let next = app.clone();
                        let selected = spec.clone();
                        let _ = app.run_on_main_thread(move || {
                            let current = get_state(&next);
                            if current.generation.load(Ordering::SeqCst) == generation
                                && !current.exiting.load(Ordering::SeqCst)
                            {
                                if let Err(error) = build_main(&next, selected, value) {
                                    failed(&next, error);
                                }
                            }
                        });
                    }
                    Some("failed") => {
                        failure = value["message"]
                            .as_str()
                            .unwrap_or("Local services could not start")
                            .to_string()
                    }
                    Some("checkpoint") => {
                        let next = app.clone();
                        let identifier = value["checkpointId"].clone();
                        let maintenance = value["maintenance"] == true;
                        let _ = app.run_on_main_thread(move || {
                            let state = get_state(&next);
                            if state.visible.load(Ordering::SeqCst) && !maintenance {
                                control(&next, "show");
                                return;
                            }
                            *state.phase.lock().unwrap() = "checkpointing".into();
                            event(&state, "checkpoint-requested");
                            if let Some(window) = next.get_webview_window("main") {
                                let _ = window.eval(format!(
                                    "window.labvizDesktopCheckpoint?.({identifier})"
                                ));
                            }
                        });
                    }
                    Some("suspend-commit") => {
                        let next = app.clone();
                        let maintenance = value["maintenance"] == true;
                        let _ = app.run_on_main_thread(move || {
                            let state = get_state(&next);
                            if (state.visible.load(Ordering::SeqCst) && !maintenance)
                                || state.exiting.load(Ordering::SeqCst)
                            {
                                control(&next, "show");
                                return;
                            }
                            *state.phase.lock().unwrap() = "suspending".into();
                            if let Some(window) = next.get_webview_window("main") {
                                if window.destroy().is_err() {
                                    control(&next, "show");
                                    return;
                                }
                            }
                            if let Some(window) = next.get_webview_window("startup") {
                                let _ = window.destroy();
                            }
                            event(&state, "webview-destroyed-for-suspension");
                            control(&next, "webview-destroyed");
                        });
                    }
                    Some("maintenance-exit") => {
                        let next = app.clone();
                        let _ = app.run_on_main_thread(move || {
                            get_state(&next).exiting.store(true, Ordering::SeqCst);
                            next.exit(0);
                        });
                    }
                    Some("exit-ready") => {
                        let next = app.clone();
                        let _ = app.run_on_main_thread(move || {
                            let state = get_state(&next);
                            if state.generation.load(Ordering::SeqCst) != generation || !state.exit_requested.load(Ordering::SeqCst) { return; }
                            let answer=rfd::MessageDialog::new().set_title("Exit LabViz? / 退出 LabViz？").set_description("Accepted tasks and downloads have finished. Exit now? Edits since the last checkpoint can be discarded. / 已接收的任务和下载已完成。现在退出吗？上次快照后的编辑可能被丢弃。").set_buttons(rfd::MessageButtons::YesNo).show();
                            if state.generation.load(Ordering::SeqCst) != generation || !state.exit_requested.load(Ordering::SeqCst) { return; }
                            if answer == rfd::MessageDialogResult::Yes {
                                control(&next, "exit-confirm");
                            } else {
                                control(&next, "exit-cancel");
                                state.exit_requested.store(false, Ordering::SeqCst);
                            }
                        });
                    }
                    Some("exit-approved") => {
                        let next = app.clone();
                        let _ = app.run_on_main_thread(move || {
                            let state = get_state(&next);
                            // Approval commits the fenced controller exit. A
                            // later cancel cannot revive its stopped services.
                            if state.generation.load(Ordering::SeqCst) == generation {
                                quit(&next, true);
                            }
                        });
                    }
                    Some("exit-cancelled") | Some("exit-blocked") => {
                        let next = app.clone();
                        let message = value["message"].as_str().map(str::to_string);
                        let _ = app.run_on_main_thread(move || {
                            if get_state(&next).generation.load(Ordering::SeqCst) != generation {
                                return;
                            }
                            get_state(&next)
                                .exit_requested
                                .store(false, Ordering::SeqCst);
                            if let Some(message) = message {
                                rfd::MessageDialog::new()
                                    .set_title("Exit postponed / 暂未退出")
                                    .set_description(message)
                                    .show();
                            }
                        });
                    }
                    Some("exit-waiting") => event(&state, "exit-waiting-for-active-work"),
                    Some("suspended") => {
                        let next = app.clone();
                        let _ = app.run_on_main_thread(move || {
                            let state = get_state(&next);
                            *state.phase.lock().unwrap() = "suspended".into();
                            event(&state, "services-suspended");
                        });
                    }
                    Some("sleep-cancelled") | Some("sleep-blocked") => {
                        let next = app.clone();
                        let blocked = value["type"] == "sleep-blocked";
                        let _ = app.run_on_main_thread(move || {
                            let state = get_state(&next);
                            *state.phase.lock().unwrap() =
                                if state.visible.load(Ordering::SeqCst) {
                                    "ready"
                                } else {
                                    "tray"
                                }
                                .into();
                            event(
                                &state,
                                if blocked {
                                    "suspension-blocked"
                                } else {
                                    "suspension-cancelled"
                                },
                            );
                        });
                    }
                    _ => return Err("Unknown local status message".into()),
                }
            }
            if let Some(mut child) = state.child.lock().unwrap().take() {
                let _ = child.wait();
            }
            if !state.exiting.load(Ordering::SeqCst) {
                return Err(failure);
            }
            Ok(())
        })();
        if let Err(error) = result {
            let next = app.clone();
            let _ = app.run_on_main_thread(move || {
                let current = get_state(&next);
                if current.generation.load(Ordering::SeqCst) == generation
                    && !current.exiting.load(Ordering::SeqCst)
                {
                    failed(&next, error);
                }
            });
        }
    });
}
fn main() {
    let entry_instant = Instant::now();
    let entry_utc_seconds = now();
    let arguments: Vec<String> = std::env::args().collect();
    let boot = PathBuf::from(std::env::var_os("LOCALAPPDATA").unwrap_or_default())
        .join("LabVizDesktopPreviewBoot");
    if checked(&boot).is_err() {
        rfd::MessageDialog::new()
            .set_title("LabViz startup")
            .set_description(
                "LabViz could not use its desktop cache. Existing files were left unchanged.",
            )
            .show();
        return;
    }
    let marker = boot.join("desktop-bootstrap.json");
    if !marker.exists()
        && boot.exists()
        && fs::read_dir(&boot)
            .map(|mut v| v.next().is_some())
            .unwrap_or(true)
    {
        rfd::MessageDialog::new().set_title("LabViz startup").set_description("The desktop cache directory belongs to another application. Existing files were left unchanged.").show();
        return;
    }
    let _ = fs::create_dir_all(&boot);
    if !marker.exists() {
        let _ = fs::write(
            &marker,
            json!({"format":"labviz-desktop-bootstrap-v1","appId":ID}).to_string(),
        );
    }
    if fs::read_to_string(&marker)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .map(|v| v["appId"] != ID || v["format"] != "labviz-desktop-bootstrap-v1")
        .unwrap_or(true)
    {
        rfd::MessageDialog::new()
            .set_title("LabViz startup")
            .set_description(
                "The desktop cache identity is invalid. Existing files were left unchanged.",
            )
            .show();
        return;
    }
    let state = Arc::new(State {
        args: arguments,
        boot: boot.clone(),
        log: boot.join(format!("host-{}.jsonl", std::process::id())),
        phase: Mutex::new("idle".into()),
        profile: Mutex::new(None),
        child: Mutex::new(None),
        visible: AtomicBool::new(true),
        show_pending: AtomicBool::new(false),
        exiting: AtomicBool::new(false),
        exit_requested: AtomicBool::new(false),
        generation: AtomicU64::new(0),
        download_sequence: AtomicU64::new(0),
        main_sequence: AtomicU64::new(0),
        entry_utc_seconds,
        entry_instant,
        downloads: Mutex::new(downloads::Downloads::default()),
    });
    let app_state = state.clone();
    std::env::remove_var("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS");
    std::env::remove_var("WEBVIEW2_USER_DATA_FOLDER");
    let application = tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![acceptance_editor_ready])
        .plugin(tauri_plugin_single_instance::init(|app, args, _| {
            let current = get_state(app);
            if let (Ok(other), Some(root)) = (load(&args), current.profile.lock().unwrap().clone())
            {
                if other.profile != root {
                    rfd::MessageDialog::new()
                        .set_title("LabViz is already running")
                        .set_description(
                            "Close the active desktop profile before opening another profile.",
                        )
                        .show();
                    return;
                }
            }
            event(&current, "second-instance-wake");
            show(app);
        }))
        .manage(state.clone())
        .setup(move |app| {
            build_startup(app.handle()).map_err(std::io::Error::other)?;
            let open_item = MenuItem::with_id(app, "open", "Open LabViz", true, None::<&str>)?;
            let retry = MenuItem::with_id(app, "retry", "Retry startup", true, None::<&str>)?;
            let logs = MenuItem::with_id(app, "logs", "Open logs", true, None::<&str>)?;
            let exit = MenuItem::with_id(app, "exit", "Exit LabViz", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open_item, &retry, &logs, &exit])?;
            let icon = tauri::image::Image::from_bytes(include_bytes!(
                "../../../V2.0/assets/labviz-logo.ico"
            ))?;
            TrayIconBuilder::with_id("desktop")
                .icon(icon)
                .tooltip("LabViz Desktop Preview")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_tray_icon_event(|tray, event| {
                    if matches!(
                        event,
                        TrayIconEvent::Click {
                            button: MouseButton::Left,
                            button_state: MouseButtonState::Up,
                            ..
                        }
                    ) {
                        show(tray.app_handle());
                    }
                })
                .build(app)?;
            event(&app_state, "startup-window-created");
            begin(app.handle());
            if let Ok(spec) = load(&app_state.args) {
                if let Some((session, qa)) = spec.qa {
                    let app = app.handle().clone();
                    std::thread::spawn(move || {
                        let mut previous = String::new();
                        while !get_state(&app).exiting.load(Ordering::SeqCst) {
                            if let Ok(bytes) = fs::read(session.with_file_name("commands.json")) {
                                if let Ok(command) = serde_json::from_slice::<Value>(&bytes) {
                                    let id = command["id"].as_str().unwrap_or("");
                                    if command["acceptanceId"] == qa.id
                                        && !id.is_empty()
                                        && id != previous
                                    {
                                        previous = id.to_string();
                                        let action =
                                            command["action"].as_str().unwrap_or("").to_string();
                                        let handle = app.clone();
                                        let tray_report =
                                            session.with_file_name("tray-inspection.json");
                                        let _ =
                                            app.run_on_main_thread(move || match action.as_str() {
                                                "show" => show(&handle),
                                                "hide" => {
                                                    if let Some(window) =
                                                        handle.get_webview_window("main")
                                                    {
                                                        let _ = window.close();
                                                    }
                                                }
                                                "retry" => begin(&handle),
                                                "tray-inspect" => {
                                                    inspect_acceptance_tray(&handle, &tray_report)
                                                }
                                                "tray-menu" => {
                                                    inspect_acceptance_tray(&handle, &tray_report);
                                                    if let Some(tray) = handle.tray_by_id("desktop")
                                                    {
                                                        let _ = tray.with_inner_tray_icon(|icon| {
                                                            icon.show_menu()
                                                        });
                                                    }
                                                }
                                                "exit-confirm" => quit(&handle, false),
                                                "exit" => quit(&handle, true),
                                                _ => {}
                                            });
                                    }
                                }
                            }
                            std::thread::sleep(std::time::Duration::from_millis(200));
                        }
                    });
                }
            }
            Ok(())
        })
        .on_menu_event(|app, action| match action.id().as_ref() {
            "open" => show(app),
            "retry" => begin(app),
            "logs" => open_logs(app),
            "exit" => quit(app, false),
            _ => {}
        })
        .on_window_event(|window, event_kind| {
            if let WindowEvent::CloseRequested { api, .. } = event_kind {
                let state = get_state(window.app_handle());
                if !state.exiting.load(Ordering::SeqCst) {
                    api.prevent_close();
                    state.visible.store(false, Ordering::SeqCst);
                    let _ = window.hide();
                    if window.label() == "main" {
                        *state.phase.lock().unwrap() = "tray".into();
                    }
                    control(window.app_handle(), "hide");
                    event(&state, "closed-to-tray");
                }
            }
        })
        .build(tauri::generate_context!());
    match application {
        Ok(app) => app.run(|app, event_kind| {
            if let RunEvent::ExitRequested { api, .. } = event_kind {
                if !get_state(app).exiting.load(Ordering::SeqCst) {
                    api.prevent_exit();
                    let phase = get_state(app).phase.lock().unwrap().clone();
                    if !["suspending", "suspended", "resuming"].contains(&phase.as_str()) {
                        quit(app, false);
                    }
                }
            }
        }),
        Err(error) => {
            rfd::MessageDialog::new()
                .set_title("LabViz startup")
                .set_description(format!("The desktop window could not start: {error}"))
                .show();
        }
    }
}
