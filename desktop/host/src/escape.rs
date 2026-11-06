use tauri::{Manager, WebviewWindow};
use webview2_com::{
    AcceleratorKeyPressedEventHandler,
    Microsoft::Web::WebView2::Win32::{
        COREWEBVIEW2_KEY_EVENT_KIND, COREWEBVIEW2_KEY_EVENT_KIND_KEY_DOWN,
    },
};

fn is_escape_press(key: u32, kind: COREWEBVIEW2_KEY_EVENT_KIND, lparam: i32) -> bool {
    key == 0x1b && kind == COREWEBVIEW2_KEY_EVENT_KIND_KEY_DOWN && (lparam as u32 & (1 << 30)) == 0
}

pub fn install(window: &WebviewWindow, origin: &str) -> tauri::Result<()> {
    let app = window.app_handle().clone();
    let script = format!(
        "if(window.location.origin==={})window.dispatchEvent(new Event('labviz:desktop-escape'));",
        serde_json::to_string(origin).unwrap()
    );
    window.with_webview(move |webview| unsafe {
        let callback = AcceleratorKeyPressedEventHandler::create(Box::new(move |_, args| {
            let Some(args) = args else { return Ok(()) };
            let mut key = 0;
            let mut kind = COREWEBVIEW2_KEY_EVENT_KIND::default();
            let mut lparam = 0;
            args.VirtualKey(&mut key)?;
            args.KeyEventKind(&mut kind)?;
            args.KeyEventLParam(&mut lparam)?;
            if is_escape_press(key, kind, lparam) {
                let app = app.clone();
                let script = script.clone();
                // The PDF viewer consumes its frame's DOM key events. Queue a
                // parent-document signal after this synchronous native callback.
                // Keep the browser's normal Escape handling outside our dialog.
                std::thread::spawn(move || {
                    let target = app.clone();
                    let _ = app.run_on_main_thread(move || {
                        if let Some(window) = target.get_webview_window("main") {
                            let _ = window.eval(script);
                        }
                    });
                });
            }
            Ok(())
        }));
        let mut token = 0;
        if let Err(error) = webview
            .controller()
            .add_AcceleratorKeyPressed(&callback, &mut token)
        {
            eprintln!("Could not register the desktop Escape handler: {error}");
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use webview2_com::Microsoft::Web::WebView2::Win32::COREWEBVIEW2_KEY_EVENT_KIND_KEY_UP;

    #[test]
    fn forwards_one_escape_press() {
        assert!(is_escape_press(
            0x1b,
            COREWEBVIEW2_KEY_EVENT_KIND_KEY_DOWN,
            1
        ));
    }

    #[test]
    fn ignores_releases_repeats_and_other_keys() {
        assert!(!is_escape_press(
            0x1b,
            COREWEBVIEW2_KEY_EVENT_KIND_KEY_UP,
            1
        ));
        assert!(!is_escape_press(
            0x1b,
            COREWEBVIEW2_KEY_EVENT_KIND_KEY_DOWN,
            1 << 30
        ));
        assert!(!is_escape_press(
            0x41,
            COREWEBVIEW2_KEY_EVENT_KIND_KEY_DOWN,
            1
        ));
    }
}
