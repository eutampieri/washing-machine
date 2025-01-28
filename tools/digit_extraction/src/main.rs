use std::fmt::{Display, Formatter};
use opencv as cv2;
use opencv::core::MatTraitConst;

const DISPLAY_REGION: cv2::core::Rect = cv2::core::Rect::new(454, 264, 46, 53);
const FIRST_DIGIT_REGION: cv2::core::Rect = cv2::core::Rect::new(0, 0, 20, 53);
const SECOND_DIGIT_REGION: cv2::core::Rect = cv2::core::Rect::new(20, 0, 11, 53);
const THIRD_DIGIT_REGION: cv2::core::Rect = cv2::core::Rect::new(31, 0, 15, 53);

enum DisplayDigit {
    Blank,
    Digit(u8),
}

impl Display for DisplayDigit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DisplayDigit::Blank => 10,
            DisplayDigit::Digit(x) => *x,
        }.fmt(f)
    }
}

fn get_digits(n: u8) -> (DisplayDigit, DisplayDigit, DisplayDigit) {
    let minutes = n % 60;
    let third = minutes % 10;
    let second = (minutes / 10) % 10;
    let first = n / 60;
    let hour = if first > 0 { DisplayDigit::Digit(first)} else {DisplayDigit::Blank};
    let second = if first > 0 || second > 0 { DisplayDigit::Digit(second)} else {DisplayDigit::Blank};
    let third = DisplayDigit::Digit(third);
    (hour, second, third)
}


fn main() {
    let directory = std::env::args().nth(1).expect("Run with path!");
    std::fs::read_dir(&directory).expect("Failed to open directory")
        .flat_map(Result::ok)
        .filter_map(|x| match x.file_type() {
            Ok(t) => {
                if t.is_file() {
                    Some(x.file_name())
                } else {
                    None
                }
            }
            _ => None,
        })
        .map(|x| {
            (
                x.to_str().unwrap().to_owned(),
                x.to_str().unwrap().parse::<labels::Label>().ok(),
            )
        })
        .filter_map(|(f, l)| {l.map(|x| (f, x))})
        .filter(|(_, l)| l.is_fully_labeled())
        .map(|(f, l)| (f, l.minutes_left.unwrap(), get_digits(l.minutes_left.unwrap() as u8)))
        .map(|(f, l, d)| {
            (
                cv2::imgcodecs::imread(&format!("{}/{f}", &directory), cv2::imgcodecs::IMREAD_GRAYSCALE).unwrap(),
                l,
                d,
            )
        })
        .enumerate()
        .for_each(|(n, (i, l, (a, b, c)))| {
            let display = i.roi(DISPLAY_REGION).unwrap();
            [FIRST_DIGIT_REGION, SECOND_DIGIT_REGION, THIRD_DIGIT_REGION].into_iter()
                .zip([a, b, c].into_iter())
                .map(|(r, d)| (display.roi(r).unwrap().clone_pointee(), d))
                .enumerate()
                .map(|(dn, (r, d))| (r, format!("{d}_{dn}_{n}.jpg")))
                .chain([(display.clone_pointee(), format!("{l}_display_{n}.jpg"))])
                .for_each(|(image, filename)| {
                    cv2::imgcodecs::imwrite(&format!("{directory}/extracted/{filename}"), &image, &opencv::core::Vector::new()).unwrap();
                });
        });
}
