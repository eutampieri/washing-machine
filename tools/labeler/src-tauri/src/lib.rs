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

impl From<&CyclePhase> for char {
    fn from(value: &CyclePhase) -> Self {
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

impl ToString for Label {
    fn to_string(&self) -> String {
        if self.is_fully_labeled() {
            format!(
                "{}{}{}.{}",
                self.timestamp,
                char::from(self.status.as_ref().unwrap()),
                self.minutes_left.unwrap(),
                self.extension
            )
        } else {
            format!("{}.{}", self.timestamp, self.extension)
        }
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
