use labels::*;

#[tauri::command]
fn get_pictures(base_path: &str) -> Vec<String> {
    std::fs::read_dir(base_path)
        .map(|x| {
            x.into_iter()
                .filter_map(|y| y.ok())
                .flat_map(|y| y.path().to_str().map(|z| z.to_string()))
                .filter(|x| !is_labeled(&x))
                .collect()
        })
        .unwrap_or_default()
}
#[tauri::command]
fn label(file: &str, phase: char, minutes: u16) {
    let src_path = std::path::Path::new(file);
    let filename = src_path.file_name().unwrap().to_str().unwrap();
    let mut label: Label = filename.parse().unwrap();
    label.minutes_left = Some(minutes);
    label.status = CyclePhase::try_from(phase).ok();
    let new_filename = label.to_string();
    let dst_path = src_path.parent().unwrap().join(new_filename);
    std::fs::rename(src_path, dst_path).unwrap();
}

#[tauri::command]
fn get_phases() -> Vec<(String, char)> {
    [
        CyclePhase::AmmolloPrelavaggio,
        CyclePhase::Lavaggio,
        CyclePhase::Risciacquo,
        CyclePhase::StopConAcqua,
        CyclePhase::ScaricoCentrifuga,
        CyclePhase::FaseAntipiegaFine,
    ]
    .into_iter()
    .map(|x| (format!("{:?}", x), (&x).into()))
    .collect()
}

fn is_labeled(filename: &str) -> bool {
    std::path::Path::new(filename)
        .file_name()
        .and_then(|x| x.to_str())
        .and_then(|x| x.parse::<Label>().ok())
        .map(|x| x.is_fully_labeled())
        .unwrap_or_default()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![get_phases, get_pictures, label])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
