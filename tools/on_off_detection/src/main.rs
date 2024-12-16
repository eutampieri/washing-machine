use cv2::prelude::*;
use opencv as cv2;

const REGION: cv2::core::Rect = cv2::core::Rect::new(568, 276, 27, 87);
const THRESHOLD: f64 = 220f64;

fn is_on<M: cv2::core::ToInputArray>(image: M) -> bool {
    // load image
    let image = cv2::imgcodecs::imdecode(&image, cv2::imgcodecs::IMREAD_UNCHANGED).unwrap();

    // convert to grey
    let mut image_grey =
        unsafe { cv2::core::Mat::new_size(image.size().unwrap(), cv2::core::CV_8UC3) }.unwrap();
    cv2::imgproc::cvt_color(&image, &mut image_grey, cv2::imgproc::COLOR_BGR2GRAY, 0).unwrap();

    // extract roi
    let image_region = image_grey.roi(REGION).unwrap();

    // binarise roi
    let mut binarized_region =
        unsafe { cv2::core::Mat::new_size(REGION.size(), cv2::core::CV_8UC3) }.unwrap();
    cv2::imgproc::threshold(
        &image_region,
        &mut binarized_region,
        THRESHOLD,
        255f64,
        cv2::imgproc::THRESH_BINARY,
    )
    .unwrap();

    // calculate average pixel value
    let (count, sum) = binarized_region
        .iter::<u8>()
        .unwrap()
        .map(|x| x.1)
        .fold((0usize, 0usize), |(count, sum), x| {
            (count + 1, sum + x as usize)
        });
    let avg = sum as f32 / count as f32;
    dbg!(avg) >= 0.85
}

fn fetch_image() -> Vec<u8> {
    let mut f = vec![];
    ureq::get(&std::env::var("FRAME_URL").unwrap())
        .call()
        .unwrap()
        .into_reader()
        .read_to_end(&mut f)
        .unwrap();
    f
}

fn upload(image: &[u8], status: bool) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let filename = format!("{}.jpg", timestamp);
    std::fs::write(&filename, &frame).unwrap();
    let image = cv2::core::Mat::from_slice(&frame).unwrap();

    let base_path = std::env::var("BASE_RCLONE_PATH").unwrap();

    let dst_path = if status {
        "/labeled/on"
    } else {
        "/labeled/off"
    };
    println!("Uploading {filename} to {dst_path}...");
    let output = std::process::Command::new("rclone")
        .arg("copy")
        .arg(&filename)
        .arg(format!("{base_path}{dst_path}"))
        .output()
        .unwrap();
    println!("status: {}", output.status);
    std::io::Write::write_all(&mut std::io::stdout(), &output.stdout).unwrap();
    std::io::Write::write_all(&mut std::io::stderr(), &output.stderr).unwrap();
    std::fs::remove_file(filename).unwrap();
}

fn job() -> bool {
    let frame = fetch_image();
    let status = is_on(frame.as_slice());
    upload(&frame, status);
    status
}

fn main() {
    let status = job();
    if status {
        std::thread::sleep(std::time::Duration::from_secs(30));
        job();
    }
}
