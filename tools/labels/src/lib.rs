use std::str::FromStr;

#[derive(Debug)]
pub enum CyclePhase {
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
pub struct Label {
    pub timestamp: u64,
    pub status: Option<CyclePhase>,
    pub minutes_left: Option<u16>,
    pub extension: String,
}

impl Label {
    pub fn is_fully_labeled(&self) -> bool {
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