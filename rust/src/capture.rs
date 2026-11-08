//! Grabs a single, fresh frame from an RTSP (or any FFmpeg-readable) stream.
//!
//! The stream is opened for every capture instead of being kept open: an idle
//! open capture keeps buffering, so a later `read` would return a stale frame.

use std::time::Duration;

use anyhow::{bail, ensure, Result};
use opencv::{
    core::{Mat, Vector},
    prelude::*,
    videoio::{self, VideoCapture},
};

pub struct FrameSource {
    url: String,
    timeout: Duration,
    warmup_frames: u32,
}

impl FrameSource {
    /// `warmup_frames` frames are decoded and discarded first: the first
    /// frames of a stream joined mid-GOP are often grey or smeared.
    pub fn new(url: String, timeout: Duration, warmup_frames: u32) -> Self {
        Self {
            url,
            timeout,
            warmup_frames,
        }
    }

    pub fn grab(&self) -> Result<Mat> {
        let timeout_ms = i32::try_from(self.timeout.as_millis()).unwrap_or(i32::MAX);
        let params = Vector::<i32>::from_slice(&[
            videoio::CAP_PROP_OPEN_TIMEOUT_MSEC,
            timeout_ms,
            videoio::CAP_PROP_READ_TIMEOUT_MSEC,
            timeout_ms,
        ]);
        let mut capture =
            VideoCapture::from_file_with_params(&self.url, videoio::CAP_FFMPEG, &params)?;
        ensure!(
            capture.is_opened()?,
            "cannot open stream {}",
            redact(&self.url)
        );

        let mut frame = Mat::default();
        let mut buffer = Mat::default();
        for _ in 0..=self.warmup_frames {
            if capture.read(&mut buffer)? && !buffer.empty() {
                std::mem::swap(&mut frame, &mut buffer);
            }
        }
        capture.release()?;
        if frame.empty() {
            bail!("no frame received from {}", redact(&self.url));
        }
        Ok(frame)
    }
}

/// Hides the password of `scheme://user:password@host/...` URLs in logs.
pub fn redact(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_string();
    };
    let authority_end = rest.find('/').unwrap_or(rest.len());
    match rest[..authority_end].rsplit_once('@') {
        Some((userinfo, host)) => {
            let user = userinfo.split_once(':').map_or(userinfo, |(u, _)| u);
            format!("{scheme}://{user}:***@{host}{}", &rest[authority_end..])
        }
        None => url.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::redact;

    #[test]
    fn redacts_credentials() {
        assert_eq!(
            redact("rtsp://admin:s3cr@t@cam:554/live"),
            "rtsp://admin:***@cam:554/live"
        );
        assert_eq!(redact("rtsp://cam/live"), "rtsp://cam/live");
        assert_eq!(redact("not a url"), "not a url");
    }
}
