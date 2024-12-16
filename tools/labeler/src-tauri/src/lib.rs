use std::str::FromStr;

#[derive(Debug)]
enum CyclePhase {
    AmmolloPrelavaggio,
    Lavaggio,
    Risciacquo,
    StopConAcqua,
    ScaricoCentrifuga,
    FaseAntipiegaFine,
}

impl From<CyclePhase> for char {
    fn from(value: CyclePhase) -> Self {
        match value {
            CyclePhase::AmmolloPrelavaggio => 'A',
            CyclePhase::Lavaggio => 'L',
            CyclePhase::Risciacquo => 'R',
            CyclePhase::StopConAcqua => 'S',
            CyclePhase::ScaricoCentrifuga => 'C',
            CyclePhase::FaseAntipiegaFine => 'F',
        }
    }
}

impl TryFrom<char> for CyclePhase {
    type Error = ();

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'A' => Ok(Self::AmmolloPrelavaggio),
            'L' => Ok(Self::Lavaggio),
            'R' => Ok(Self::Risciacquo),
            'S' => Ok(Self::StopConAcqua),
            'C' => Ok(Self::ScaricoCentrifuga),
            'F' => Ok(Self::FaseAntipiegaFine),
            _ => Err(()),
        }
    }
}

#[derive(Debug)]
struct Label {
    timestamp: u64,
    status: Option<CyclePhase>,
    minutes_left: Option<u16>,
    extension: String,
}

impl Label {
    fn is_fully_labeled(&self) -> bool {
        self.status.is_some() && self.minutes_left.is_some()
    }
}

impl FromStr for Label {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut s = s.split('.');
        let file = s.next().ok_or("Empty file name!")?;
        let extension = s.next().ok_or("File name without extension!")?;
        let timestamp = file[0..10].parse().map_err(|_s| "Invalid timestamp")?;
        let status = file
            .chars()
            .nth(10)
            .ok_or("String too short")
            .and_then(|x| x.try_into().map_err(|_| "Parsing failed"))
            .ok();
        let minutes_left = if let Some(part) = file.get(11..) {
            part.parse().ok()
        } else {
            None
        };
        Ok(Self {
            timestamp,
            status,
            minutes_left,
            extension: extension.to_string(),
        })
    }
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
fn get_pictures(base_path: &str) -> Vec<String> {
    std::fs::read_dir(base_path)
        .map(|x| {
            x.into_iter()
                .filter_map(|y| y.ok())
                .flat_map(|y| y.path().to_str().map(|z| z.to_string()))
                .filter(|x| !is_labeled(dbg!(&x)))
                .collect()
        })
        .unwrap_or_default()
}

fn is_labeled(filename: &str) -> bool {
    std::path::Path::new(filename)
        .file_name()
        .and_then(|x| x.to_str())
        .and_then(|x| dbg!(x.parse::<Label>()).ok())
        .map(|x| dbg!(x.is_fully_labeled()))
        .unwrap_or_default()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![greet, get_pictures])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
