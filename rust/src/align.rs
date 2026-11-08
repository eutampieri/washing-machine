//! Compensates for camera movement.
//!
//! The templates, search offsets, digit cells and phase bands all assume the
//! framing of the dataset. Each frame is therefore registered onto a reference
//! dataset frame (`assets/reference_frame.jpg`): SIFT features of the frame are
//! matched to those of the machine's front panel in the reference, a homography
//! is fitted with RANSAC, and the frame is warped into the reference geometry.
//! A frame that is already aligned (as the whole dataset is) is left untouched.

use anyhow::{bail, ensure, Context, Result};
use opencv::{
    calib3d,
    core::{self, DMatch, KeyPoint, Mat, Point, Point2f, Scalar, Vector},
    features2d::{self, SIFT},
    imgcodecs, imgproc,
    prelude::*,
};
use serde::Serialize;

use crate::recognise::REFERENCE_SIZE;

const REFERENCE_FRAME: &[u8] = include_bytes!("../assets/reference_frame.jpg");

/// Outline of the machine's front panel in the reference frame. Only features
/// inside it are used: the boxes on top of the machine and the door move.
const PANEL: [(i32, i32); 5] = [(250, 205), (1150, 265), (1100, 595), (440, 415), (255, 335)];

/// Lowe's ratio test for feature matches.
const MATCH_RATIO: f32 = 0.75;
/// RANSAC reprojection threshold, in pixels.
const RANSAC_THRESHOLD: f64 = 4.0;
/// Fewer RANSAC inliers than this means the panel was not found.
const MIN_INLIERS: usize = 25;
/// The panel may appear between this factor smaller and larger.
const MAX_SCALE_CHANGE: f64 = 3.0;
/// Below this panel displacement (px) the frame is considered aligned.
const ALIGNED_TOLERANCE: f64 = 1.0;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Alignment {
    /// Ratio-test matches and RANSAC inliers between frame and reference.
    pub matches: usize,
    pub inliers: usize,
    /// Mean displacement of the panel corners from their reference position,
    /// in pixels: how far the camera has moved from the dataset framing.
    pub panel_shift: f64,
    /// Whether the frame was warped (false when already aligned).
    pub warped: bool,
}

pub struct Aligner {
    keypoints: Vector<KeyPoint>,
    descriptors: Mat,
}

impl Aligner {
    pub fn new() -> Result<Self> {
        let reference = imgcodecs::imdecode(
            &Vector::<u8>::from_slice(REFERENCE_FRAME),
            imgcodecs::IMREAD_GRAYSCALE,
        )?;
        ensure!(
            reference.size()? == REFERENCE_SIZE,
            "reference frame must be {}x{}",
            REFERENCE_SIZE.width,
            REFERENCE_SIZE.height
        );
        let mut mask = Mat::new_size_with_default(REFERENCE_SIZE, core::CV_8UC1, Scalar::all(0.0))?;
        imgproc::fill_convex_poly_def(&mut mask, &panel_outline(), Scalar::all(255.0))?;
        let (keypoints, descriptors) = detect(&reference, &mask)?;
        ensure!(
            keypoints.len() >= MIN_INLIERS,
            "too few features in the reference panel"
        );
        Ok(Self {
            keypoints,
            descriptors,
        })
    }

    /// Warps a grey `REFERENCE_SIZE` frame into the reference geometry.
    pub fn align(&self, image: &Mat) -> Result<(Mat, Alignment)> {
        let (keypoints, descriptors) = detect(image, &core::no_array())?;
        ensure!(
            keypoints.len() >= MIN_INLIERS,
            "too few features in the frame ({}); is the camera covered or the room dark?",
            keypoints.len()
        );

        let matcher = features2d::BFMatcher::create(core::NORM_L2, false)?;
        let mut candidates = Vector::<Vector<DMatch>>::new();
        matcher.knn_train_match_def(&descriptors, &self.descriptors, &mut candidates, 2)?;
        let mut frame_points = Vector::<Point2f>::new();
        let mut reference_points = Vector::<Point2f>::new();
        for pair in &candidates {
            if pair.len() < 2 {
                continue;
            }
            let (best, second) = (pair.get(0)?, pair.get(1)?);
            if best.distance < MATCH_RATIO * second.distance {
                frame_points.push(keypoints.get(best.query_idx as usize)?.pt());
                reference_points.push(self.keypoints.get(best.train_idx as usize)?.pt());
            }
        }
        let matches = frame_points.len();
        ensure!(
            matches >= MIN_INLIERS,
            "the machine's panel is not in view ({matches} feature matches)"
        );

        let mut inlier_mask = Mat::default();
        let homography = calib3d::find_homography(
            &frame_points,
            &reference_points,
            &mut inlier_mask,
            // MAGSAC: more robust than plain RANSAC with many outliers (it
            // avoided degenerate fits on daylight frames).
            calib3d::USAC_MAGSAC,
            RANSAC_THRESHOLD,
        )?;
        ensure!(
            !homography.empty(),
            "cannot register the frame onto the reference"
        );
        let inliers = core::count_non_zero(&inlier_mask)? as usize;
        ensure!(
            inliers >= MIN_INLIERS,
            "the machine's panel is not in view ({inliers} consistent matches)"
        );

        let panel_shift = check_and_measure(&homography)?;
        let alignment = Alignment {
            matches,
            inliers,
            panel_shift,
            warped: panel_shift >= ALIGNED_TOLERANCE,
        };
        if !alignment.warped {
            return Ok((image.try_clone()?, alignment));
        }
        let mut warped = Mat::default();
        imgproc::warp_perspective(
            image,
            &mut warped,
            &homography,
            REFERENCE_SIZE,
            // Bicubic: sharper than bilinear, which blurred weak digits into others.
            imgproc::INTER_CUBIC,
            core::BORDER_CONSTANT,
            Scalar::all(0.0),
        )?;
        Ok((warped, alignment))
    }
}

fn panel_outline() -> Vector<Point> {
    PANEL.iter().map(|&(x, y)| Point::new(x, y)).collect()
}

/// SIFT features, detected on a contrast-equalised (CLAHE) copy of the image:
/// the reference is a night-time infrared frame, the camera also delivers
/// daylight colour frames, and equalising both doubles to triples the matches.
fn detect(image: &Mat, mask: &impl core::ToInputArray) -> Result<(Vector<KeyPoint>, Mat)> {
    let mut clahe = imgproc::create_clahe(2.0, core::Size::new(8, 8))?;
    let mut equalised = Mat::default();
    clahe.apply(image, &mut equalised)?;
    let image = &equalised;
    let mut sift = SIFT::create_def()?;
    let mut keypoints = Vector::new();
    let mut descriptors = Mat::default();
    sift.detect_and_compute(image, mask, &mut keypoints, &mut descriptors, false)
        .context("feature detection")?;
    Ok((keypoints, descriptors))
}

/// Rejects degenerate homographies and returns the mean distance between the
/// panel corners in the reference and where they appear in the frame.
fn check_and_measure(frame_to_reference: &Mat) -> Result<f64> {
    let mut reference_to_frame = Mat::default();
    core::invert(frame_to_reference, &mut reference_to_frame, core::DECOMP_LU)?;
    let reference: Vector<Point2f> = PANEL
        .iter()
        .map(|&(x, y)| Point2f::new(x as f32, y as f32))
        .collect();
    let mut in_frame = Vector::<Point2f>::new();
    core::perspective_transform(&reference, &mut in_frame, &reference_to_frame)?;

    let area = |points: &Vector<Point2f>| imgproc::contour_area(points, true);
    let (reference_area, frame_area) = (area(&reference)?, area(&in_frame)?);
    // Same orientation (no mirroring) and a plausible change of scale.
    let ratio = frame_area / reference_area;
    if !(ratio > 0.0 && ratio > 1.0 / MAX_SCALE_CHANGE.powi(2) && ratio < MAX_SCALE_CHANGE.powi(2))
    {
        bail!("implausible registration (panel area ratio {ratio:.2})");
    }
    if !imgproc::is_contour_convex(&in_frame)? {
        bail!("implausible registration (panel outline not convex)");
    }
    let shift = reference
        .iter()
        .zip(&in_frame)
        .map(|(a, b)| (a - b).norm())
        .sum::<f64>()
        / PANEL.len() as f64;
    Ok(shift)
}
