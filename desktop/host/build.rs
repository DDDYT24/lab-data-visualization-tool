fn main() {
    println!("cargo:rerun-if-changed=permissions/acceptance-ready.toml");
    tauri_build::try_build(
        tauri_build::Attributes::new().app_manifest(
            tauri_build::AppManifest::new()
                .permissions_path_pattern("permissions/acceptance-ready.toml"),
        ),
    )
    .expect("Desktop build configuration failed")
}
