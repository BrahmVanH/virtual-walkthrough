mod blend;
mod canvas;
mod cylinder;
mod homography;
mod rotation;

use canvas::Canvas;
use homography::estimate_homography;
use std::ops::Range;

/// A pair needs this many matches explained by its fitted model to count as
/// overlapping.
pub const MIN_INLIERS: usize = 20;

/// Below this fraction of the homography's inliers, a pure rotation is too poor
/// a fit to be the real camera motion. A rotation is a special homography, so
/// with the lens held in place both explain about the same matches; a flat
/// surface seen from two positions fits only the homography. A wrong focal
/// setting distorts the rotation model the same way, so it also trips this.
const CAMERA_MOVED_RATIO: f64 = 0.7;

/// Below this fraction the rotation fits only a minor part of the overlap (for
/// example a far wall while the floor and furniture disagree), so its yaw is
/// unreliable and the pair is rejected. Between this and `CAMERA_MOVED_RATIO`
/// the pair is kept with a warning.
const CAMERA_MOVED_REJECT_RATIO: f64 = 0.5;

use akaze::Akaze;
use bitarray::BitArray;
use image::{DynamicImage, ImageReader, RgbImage, imageops::FilterType};
use js_sys::Uint8Array;
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::new(log::Level::Info));
}

/// Round-trip check for the JS harness.
#[wasm_bindgen]
pub fn hello(name: &str) -> String {
    format!("hello from wasm, {name}")
}

/// Focal length as a fraction of the image's long side when the caller gives
/// none. 0.75 matches a 26 mm-equivalent phone main camera on a 4:3 sensor
/// (about 67° across the long side).
pub const DEFAULT_FOCAL_PER_LONG_SIDE: f64 = 0.75;

/// Result of matching image `from` against image `to = from + 1`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairReport {
    pub from: usize,
    pub to: usize,
    /// Symmetric, ratio-tested descriptor matches fed to RANSAC.
    pub matches: usize,
    /// Matches a pure camera rotation explains. The pair is accepted when this
    /// reaches `MIN_INLIERS` and at least half of `homography_inliers`.
    pub inliers: usize,
    /// Matches the best homography explains, for comparison. `None` below
    /// `MIN_INLIERS`.
    pub homography_inliers: Option<usize>,
    /// Row-major 3x3 homography mapping original-resolution pixel coords in
    /// `from` to `to`. Diagnostic only; placement uses `rotation`.
    pub homography: Option<[f64; 9]>,
    /// Row-major 3x3 rotation from camera `from`'s rays into camera `to`'s
    /// coords. `None` when the pair was rejected.
    pub rotation: Option<[f64; 9]>,
    /// How far camera `to` is turned right of camera `from`, in degrees, under
    /// the assumed focal length.
    pub yaw_degrees: Option<f64>,
    /// A homography explains far more matches than a rotation: the lens moved
    /// between the shots, or the focal setting is off.
    pub camera_moved: bool,
}

/// Why an image has no outline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Unplaced {
    /// Not in the longest run of images joined by accepted pairs.
    OutsideGroup,
    /// Its border reaches the cylinder's axis (camera pointing straight up or down).
    AlongAxis,
}

/// Where image `index` lands on the reference cylinder.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagePlacement {
    pub index: usize,
    pub width: u32,
    pub height: u32,
    /// Focal length assumed for this image, in its own pixels.
    pub focal: f64,
    /// Row-major 3x3 rotation from this camera's ray coords (+x right, +y down,
    /// +z forward) into the reference image's. `None` outside the placed group.
    pub rotation: Option<[f64; 9]>,
    /// Image border, clockwise from top-left, in output canvas pixels (centres
    /// at integers, +x = turning right, +y down). `None` when unplaced.
    pub outline: Option<Vec<[f64; 2]>>,
    /// Set exactly when `outline` is `None`.
    pub unplaced: Option<Unplaced>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RowAlignment {
    /// Index of the image the cylinder is centred on: the first image of the
    /// longest run joined by accepted pairs.
    pub reference: usize,
    /// Cylinder radius in cylinder pixels (the reference image's focal length).
    /// One canvas pixel spans `1 / (canvas.scale * radius)` radians of yaw.
    pub radius: f64,
    /// Output canvas covering every placed image. `None` only when nothing was placed.
    pub canvas: Option<Canvas>,
    pub images: Vec<ImagePlacement>,
    pub pairs: Vec<PairReport>,
}

/// Lowe ratio when the caller gives none. 0.8 is Lowe's original value and the
/// one OpenCV's AKAZE matching tutorial uses; applied in both directions it is
/// already strict. 0.7 discarded too many true matches in rooms full of
/// identical chairs and floorboards.
pub const DEFAULT_RATIO: f32 = 0.8;

/// Default long-side cap (px) for the copy AKAZE runs on. A 4000x3000 phone
/// photo drops to 1280x960, which keeps AKAZE's per-level float buffers around
/// 5 MB instead of ~48 MB while leaving plenty of texture to match on.
pub const DEFAULT_FEATURE_MAX_DIM: u32 = 1280;

/// Default keypoints kept per image (strongest response first). Brute-force
/// matching is quadratic in this.
pub const DEFAULT_MAX_FEATURES: usize = 3000;

/// Default long-side cap (px) for the output canvas. 4096 px of RGBA8 is at
/// most 64 MB, and covers 154° at a 1125 px focal without shrinking.
pub const DEFAULT_OUTPUT_MAX_DIM: u32 = 4096;

/// Optional tuning passed from JS. Every field may be omitted, `null`, or `undefined`.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AlignOptions {
    pub focal_per_long_side: Option<f64>,
    pub ratio: Option<f32>,
    pub max_features: Option<usize>,
    pub feature_max_dim: Option<u32>,
    pub output_max_dim: Option<u32>,
    pub cylindrical_matching: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MatchSettings {
    /// Focal length as a fraction of each image's long side. The rotation fit,
    /// the unrolling and the placement all use it.
    focal_scale: f64,
    ratio: f32,
    max_features: usize,
    feature_max_dim: u32,
    /// Unroll feature images onto a cylinder before AKAZE runs.
    cylindrical: bool,
}

impl Default for MatchSettings {
    fn default() -> Self {
        Self {
            focal_scale: DEFAULT_FOCAL_PER_LONG_SIDE,
            ratio: DEFAULT_RATIO,
            max_features: DEFAULT_MAX_FEATURES,
            feature_max_dim: DEFAULT_FEATURE_MAX_DIM,
            cylindrical: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Settings {
    matching: MatchSettings,
    output_max_dim: u32,
}

fn resolve_options(o: AlignOptions) -> Result<Settings, String> {
    let focal_scale = o.focal_per_long_side.unwrap_or(DEFAULT_FOCAL_PER_LONG_SIDE);
    if !(focal_scale.is_finite() && focal_scale > 0.0) {
        return Err(format!("focalPerLongSide must be a positive number, got {focal_scale}"));
    }
    let ratio = o.ratio.unwrap_or(DEFAULT_RATIO);
    if !(ratio > 0.0 && ratio <= 1.0) {
        return Err(format!("ratio must be in (0, 1], got {ratio}"));
    }
    // Below the inlier floor no pair could ever be accepted.
    let max_features = o.max_features.unwrap_or(DEFAULT_MAX_FEATURES);
    if max_features < MIN_INLIERS {
        return Err(format!("maxFeatures must be at least {MIN_INLIERS}, got {max_features}"));
    }
    // AKAZE skips any octave whose short side is under 40 px; a 160 px long side
    // still leaves one octave for images up to 4:1.
    let feature_max_dim = o.feature_max_dim.unwrap_or(DEFAULT_FEATURE_MAX_DIM);
    if feature_max_dim < 160 {
        return Err(format!("featureMaxDim must be at least 160, got {feature_max_dim}"));
    }
    let output_max_dim = o.output_max_dim.unwrap_or(DEFAULT_OUTPUT_MAX_DIM);
    if output_max_dim < 64 {
        return Err(format!("outputMaxDim must be at least 64, got {output_max_dim}"));
    }
    Ok(Settings {
        matching: MatchSettings {
            focal_scale,
            ratio,
            max_features,
            feature_max_dim,
            cylindrical: o.cylindrical_matching.unwrap_or(true),
        },
        output_max_dim,
    })
}

fn parse_options(js: &JsValue) -> Result<Settings, String> {
    let options = if js.is_undefined() || js.is_null() {
        AlignOptions::default()
    } else {
        serde_wasm_bindgen::from_value(js.clone()).map_err(|e| format!("invalid options: {e}"))?
    };
    resolve_options(options)
}

/// Decodes each encoded image (PNG or JPEG), matches AKAZE features between
/// adjacent images, fits a pure camera rotation per pair (flagging pairs where
/// the camera moved), places the longest run of joined images on a cylinder
/// centred on its first image, and renders the feather-blended panorama.
///
/// `options` is an optional `AlignOptions` object (camelCase keys):
/// `focalPerLongSide` (default 0.75), `ratio` (default 0.8), `maxFeatures`
/// (default 3000), `featureMaxDim` (default 1280), `outputMaxDim` (default
/// 4096), `cylindricalMatching` (default true: unroll feature images onto a
/// cylinder at `focalPerLongSide` before AKAZE runs).
///
/// Returns `{ alignment, panorama }`: `alignment` is a `RowAlignment`;
/// `panorama` is `{ width, height, pixels }` with `pixels` a
/// `Uint8ClampedArray` of row-major RGBA8 (ready for `new ImageData`), or
/// `null` when no image was placed. Uncovered pixels have alpha 0.
/// Throws if any image fails to decode or an option is out of range.
#[wasm_bindgen]
pub fn stitch_row(images: Vec<Uint8Array>, options: JsValue) -> Result<JsValue, JsError> {
    let settings = parse_options(&options).map_err(|e| JsError::new(&e))?;
    let encoded: Vec<Vec<u8>> = images.iter().map(Uint8Array::to_vec).collect();
    let imgs = decode_images(&encoded).map_err(|e| JsError::new(&e))?;
    let alignment = align_row_images(&imgs, &settings);
    let panorama = render_panorama(&imgs, &alignment);
    stitch_to_js(&alignment, panorama.as_ref())
}

/// RGBA8, row-major from the top-left, `width * height * 4` bytes.
struct Panorama {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Built by hand: serde would turn `pixels` into a plain JS array of numbers,
/// one boxed value per byte.
fn stitch_to_js(alignment: &RowAlignment, panorama: Option<&Panorama>) -> Result<JsValue, JsError> {
    let set = |obj: &js_sys::Object, key: &str, value: &JsValue| {
        js_sys::Reflect::set(obj, &JsValue::from_str(key), value)
            .map(|_| ())
            .map_err(|_| JsError::new(&format!("failed to set {key} on result")))
    };
    let out = js_sys::Object::new();
    set(&out, "alignment", &to_js(alignment)?)?;
    let pano = match panorama {
        None => JsValue::NULL,
        Some(p) => {
            let obj = js_sys::Object::new();
            set(&obj, "width", &JsValue::from(p.width))?;
            set(&obj, "height", &JsValue::from(p.height))?;
            set(&obj, "pixels", &js_sys::Uint8ClampedArray::from(p.pixels.as_slice()))?;
            obj.into()
        }
    };
    set(&out, "panorama", &pano)?;
    Ok(out.into())
}

/// JPEGs decode straight to RGB8, so this borrows rather than copies in the
/// common case. Source alpha is ignored.
fn rgb_view(img: &DynamicImage) -> Cow<'_, RgbImage> {
    match img.as_rgb8() {
        Some(rgb) => Cow::Borrowed(rgb),
        None => Cow::Owned(img.to_rgb8()),
    }
}

/// Warps every placed image onto `alignment.canvas`. `None` when nothing was placed.
fn render_panorama(imgs: &[DynamicImage], alignment: &RowAlignment) -> Option<Panorama> {
    let canvas = alignment.canvas?;
    let placed: Vec<_> = alignment
        .images
        .iter()
        .zip(imgs)
        .filter_map(|(p, img)| {
            let rotation = Matrix3::from_row_slice(&p.rotation?);
            let outline = p.outline.as_deref()?;
            let k = cylinder::intrinsics(p.focal, p.width, p.height);
            Some((rotation, k, outline, rgb_view(img)))
        })
        .collect();
    let sources: Vec<_> = placed
        .iter()
        .map(|(rotation, k, outline, pixels)| blend::Source::new(pixels, rotation, k, outline, &canvas))
        .collect();

    let pixels = blend::render(&canvas, alignment.radius, &sources);
    log::info!(
        "rendered {}x{} panorama from {} images",
        canvas.width,
        canvas.height,
        sources.len()
    );
    Some(Panorama {
        width: canvas.width,
        height: canvas.height,
        pixels,
    })
}

/// serde-wasm-bindgen turns `None` into `undefined` by default; the TS types promise `null`.
fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_missing_as_null(true);
    value
        .serialize(&serializer)
        .map_err(|e| JsError::new(&e.to_string()))
}

fn align_row_images(imgs: &[DynamicImage], settings: &Settings) -> RowAlignment {
    let pairs = match_adjacent_pairs(imgs, &settings.matching);
    let mut place = place_on_cylinder(imgs, &pairs, settings.matching.focal_scale);
    let canvas = fit_canvas(&mut place.images, settings.output_max_dim);
    RowAlignment {
        reference: place.reference,
        radius: place.radius,
        canvas,
        images: place.images,
        pairs,
    }
}

/// Bounds every placed outline with a canvas and rewrites the outlines from
/// cylinder coords into that canvas's pixel coords.
fn fit_canvas(images: &mut [ImagePlacement], max_dim: u32) -> Option<Canvas> {
    let points = images.iter().filter_map(|img| img.outline.as_ref()).flatten().copied();
    let canvas = Canvas::bounding(points, max_dim)?;
    for outline in images.iter_mut().filter_map(|img| img.outline.as_mut()) {
        for p in outline.iter_mut() {
            *p = canvas.to_canvas(*p);
        }
    }
    Some(canvas)
}

struct CylinderPlacement {
    reference: usize,
    radius: f64,
    images: Vec<ImagePlacement>,
}

/// Longest run of images joined by accepted adjacent pairs, as an image index
/// range (`accepted[i]` joins images `i` and `i + 1`). Ties go to the earliest
/// run. Empty only when there are no images.
fn largest_group(accepted: &[bool], n_images: usize) -> Range<usize> {
    if n_images == 0 {
        return 0..0;
    }
    let mut best = 0..1;
    let mut start = 0;
    for (i, &ok) in accepted.iter().enumerate().take(n_images - 1) {
        if !ok {
            start = i + 1;
        }
        if i + 2 - start > best.len() {
            best = start..i + 2;
        }
    }
    best
}

/// Chains the pair rotations inside `group` so each image there is expressed in
/// the group's first image's coords. Images outside the group get `None`.
fn chain_group(
    pairwise: &[Option<Matrix3<f64>>],
    group: Range<usize>,
    n_images: usize,
) -> Vec<Option<Matrix3<f64>>> {
    let mut out = vec![None; n_images];
    if group.is_empty() {
        return out;
    }
    let chained = cylinder::chain_rotations(&pairwise[group.start..group.end - 1]);
    for (offset, r) in chained.into_iter().enumerate() {
        out[group.start + offset] = r;
    }
    out
}

/// Chains the accepted pair rotations of the longest joined run of images and
/// places that run on a cylinder centred on its first image.
fn place_on_cylinder(imgs: &[DynamicImage], pairs: &[PairReport], focal_scale: f64) -> CylinderPlacement {
    let focals: Vec<f64> = imgs
        .iter()
        .map(|img| focal_scale * img.width().max(img.height()) as f64)
        .collect();
    let ks: Vec<Matrix3<f64>> = imgs
        .iter()
        .zip(&focals)
        .map(|(img, &f)| cylinder::intrinsics(f, img.width(), img.height()))
        .collect();

    let pairwise: Vec<Option<Matrix3<f64>>> = pairs
        .iter()
        .map(|p| p.rotation.map(|r| Matrix3::from_row_slice(&r)))
        .collect();
    let accepted: Vec<bool> = pairwise.iter().map(Option::is_some).collect();
    let group = largest_group(&accepted, imgs.len());
    let reference = group.start;

    let radius = focals.get(reference).copied().unwrap_or(0.0);
    let mut near_theta = 0.0;
    let images = chain_group(&pairwise, group, imgs.len())
        .into_iter()
        .zip(imgs)
        .enumerate()
        .map(|(index, (rotation, img))| {
            let outline = rotation.and_then(|r| {
                cylinder::cylinder_outline(&r, &ks[index], img.width(), img.height(), radius, near_theta)
            });
            let outline = outline.map(|(outline, centre_theta)| {
                near_theta = centre_theta;
                outline
            });
            let unplaced = match (&rotation, &outline) {
                (None, _) => Some(Unplaced::OutsideGroup),
                (Some(_), None) => Some(Unplaced::AlongAxis),
                (Some(_), Some(_)) => None,
            };
            ImagePlacement {
                index,
                width: img.width(),
                height: img.height(),
                focal: focals[index],
                rotation: rotation.as_ref().map(row_major),
                outline,
                unplaced,
            }
        })
        .collect();

    CylinderPlacement {
        reference,
        radius,
        images,
    }
}

fn normalize_h(h: Matrix3<f64>) -> Matrix3<f64> {
    if h[(2, 2)].abs() > 1e-12 { h / h[(2, 2)] } else { h }
}

fn decode_images(encoded: &[Vec<u8>]) -> Result<Vec<DynamicImage>, String> {
    encoded
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            ImageReader::new(Cursor::new(bytes.as_slice()))
                .with_guessed_format()
                .map_err(|e| format!("image {i}: {e}"))?
                .decode()
                .map_err(|e| format!("image {i}: {e}"))
        })
        .collect()
}

struct ImageFeatures {
    /// Keypoints AKAZE found, before any were dropped or capped.
    detected: usize,
    /// Keypoints dropped because their descriptor would sample the black fill
    /// around an unrolled image.
    near_border: usize,
    /// Positions in the downscaled planar feature image (already mapped back
    /// from the unrolled image when cylindrical matching is on).
    keypoints: Vec<akaze::KeyPoint>,
    descriptors: Vec<BitArray<64>>,
    /// Maps feature-image pixel coords to original-image pixel coords.
    to_full: Matrix3<f64>,
    /// Intrinsics of the planar feature image, for the rotation fit.
    k: Matrix3<f64>,
}

/// Grayscale copy no larger than `max_dim` on its long side. Never upscales.
fn feature_image(img: &DynamicImage, max_dim: u32) -> DynamicImage {
    let gray = img.grayscale();
    if img.width().max(img.height()) <= max_dim {
        return gray;
    }
    gray.resize(max_dim, max_dim, FilterType::Triangle)
}

fn extract_features(akaze: &Akaze, img: &DynamicImage, settings: &MatchSettings) -> ImageFeatures {
    let small = feature_image(img, settings.feature_max_dim);
    let to_full = scale_to_full(small.width(), small.height(), img.width(), img.height());
    let focal = settings.focal_scale * small.width().max(small.height()) as f64;
    let k = cylinder::intrinsics(focal, small.width(), small.height());

    if !settings.cylindrical {
        let (keypoints, descriptors) = akaze.extract(&small);
        return select_features(keypoints, descriptors, 0, settings.max_features, to_full, k);
    }

    let warp = cylinder::CylinderWarp::new(focal, small.width(), small.height());
    let unrolled = DynamicImage::ImageLuma8(warp.warp_gray(&small.to_luma8()));
    let (keypoints, descriptors) = akaze.extract(&unrolled);

    let detected = keypoints.len();
    let (keypoints, descriptors): (Vec<_>, Vec<_>) = keypoints
        .into_iter()
        .zip(descriptors)
        .filter(|(kp, _)| {
            let (u, v) = kp.point;
            let r = descriptor_radius(kp, akaze.descriptor_pattern_size);
            warp.contains_disc(u as f64, v as f64, r)
        })
        .map(|(mut kp, d)| {
            let (x, y) = warp.to_plane(kp.point.0 as f64, kp.point.1 as f64);
            kp.point = (x as f32, y as f32);
            (kp, d)
        })
        .unzip();
    let near_border = detected - keypoints.len();
    select_features(keypoints, descriptors, near_border, settings.max_features, to_full, k)
}

/// Farthest a keypoint's MLDB descriptor samples from its centre, in image px.
/// akaze's descriptors.rs samples a `±pattern_size` grid at a step of
/// `0.5 * size`, rotated by the keypoint angle, so the corners reach √2 further.
fn descriptor_radius(kp: &akaze::KeyPoint, pattern_size: usize) -> f64 {
    pattern_size as f64 * 0.5 * kp.size as f64 * std::f64::consts::SQRT_2
}

fn select_features(
    keypoints: Vec<akaze::KeyPoint>,
    descriptors: Vec<BitArray<64>>,
    near_border: usize,
    max: usize,
    to_full: Matrix3<f64>,
    k: Matrix3<f64>,
) -> ImageFeatures {
    let keep = strongest_keypoints(&keypoints, max);
    ImageFeatures {
        detected: keypoints.len() + near_border,
        near_border,
        keypoints: keep.iter().map(|&i| keypoints[i]).collect(),
        descriptors: keep.iter().map(|&i| descriptors[i]).collect(),
        to_full,
        k,
    }
}

/// Indices (ascending) of the `max` keypoints with the highest detector response.
fn strongest_keypoints(keypoints: &[akaze::KeyPoint], max: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keypoints.len()).collect();
    order.sort_by(|&a, &b| {
        keypoints[b]
            .response
            .total_cmp(&keypoints[a].response)
            .then(a.cmp(&b))
    });
    order.truncate(max);
    order.sort_unstable();
    order
}

/// Pixel centres sit at integer coords in both images, and the outer pixel
/// edges (-0.5 and size - 0.5) must line up, so the map is
/// `x_full = (x + 0.5) * s - 0.5` rather than a plain scale.
fn scale_to_full(small_w: u32, small_h: u32, full_w: u32, full_h: u32) -> Matrix3<f64> {
    let sx = full_w as f64 / small_w as f64;
    let sy = full_h as f64 / small_h as f64;
    Matrix3::new(sx, 0.0, 0.5 * sx - 0.5, 0.0, sy, 0.5 * sy - 0.5, 0.0, 0.0, 1.0)
}

/// Converts a homography between two feature images into one between the
/// corresponding originals.
fn lift_homography(
    h: &Matrix3<f64>,
    to_full_a: &Matrix3<f64>,
    to_full_b: &Matrix3<f64>,
) -> Option<Matrix3<f64>> {
    to_full_a
        .try_inverse()
        .map(|from_full_a| normalize_h(to_full_b * h * from_full_a))
}

/// `imgs` stay full colour for the later blend; features come from small grayscale copies.
fn match_adjacent_pairs(imgs: &[DynamicImage], settings: &MatchSettings) -> Vec<PairReport> {
    // Uncapped so the log can show whether max_features is actually binding.
    let akaze = Akaze::default();
    let features: Vec<ImageFeatures> = imgs
        .iter()
        .enumerate()
        .map(|(i, img)| {
            let f = extract_features(&akaze, img, settings);
            log::info!(
                "image {i}: {} keypoints, {} dropped at the unrolled border, kept {}",
                f.detected,
                f.near_border,
                f.keypoints.len()
            );
            f
        })
        .collect();

    features
        .windows(2)
        .enumerate()
        .map(|(idx, pair)| {
            let (a, b) = (&pair[0], &pair[1]);

            let point_pairs: Vec<_> = symmetric_matches(&a.descriptors, &b.descriptors, settings.ratio)
                .iter()
                .map(|&(i, j)| (a.keypoints[i].point, b.keypoints[j].point))
                .collect();
            let report = fit_pair(idx, &point_pairs, a, b);
            log::info!(
                "pair {}->{}: {} matches, {} rotation inliers, {:?} homography inliers{}",
                report.from,
                report.to,
                report.matches,
                report.inliers,
                report.homography_inliers,
                if report.camera_moved { ", camera moved" } else { "" }
            );
            report
        })
        .collect()
}

/// Fits both models to one pair's matches (planar feature-image coords) and
/// accepts the rotation when `accepts_rotation` allows it.
///
/// Both fits run in feature-image coords so the shared 3 px threshold stays
/// matched to AKAZE's localization precision at that scale. Mapping unrolled
/// keypoints back to the plane stretches that error by up to 1/cos² of the yaw
/// to the edge (about 1.9x at 43°).
fn fit_pair(
    from: usize,
    point_pairs: &[((f32, f32), (f32, f32))],
    a: &ImageFeatures,
    b: &ImageFeatures,
) -> PairReport {
    let homography = estimate_homography(point_pairs);
    let homography_inliers = homography.as_ref().map(|(_, inliers)| inliers.len());
    let fit = rotation::estimate_rotation(point_pairs, &a.k, &b.k);
    let inliers = fit.as_ref().map_or(0, |(_, inliers)| inliers.len());
    let accepted = fit
        .filter(|(_, inliers)| accepts_rotation(inliers.len(), homography_inliers))
        .map(|(r, _)| r);

    PairReport {
        from,
        to: from + 1,
        matches: point_pairs.len(),
        inliers,
        homography_inliers,
        homography: homography
            .as_ref()
            .and_then(|(h, _)| lift_homography(h, &a.to_full, &b.to_full))
            .as_ref()
            .map(row_major),
        rotation: accepted.as_ref().map(row_major),
        yaw_degrees: accepted.as_ref().map(cylinder::yaw_degrees),
        camera_moved: camera_moved(inliers, homography_inliers),
    }
}

/// Whether the rotation explains under `ratio` of the homography's inliers.
/// Says nothing when no homography reached `MIN_INLIERS`.
fn rotation_below(ratio: f64, rotation_inliers: usize, homography_inliers: Option<usize>) -> bool {
    homography_inliers.is_some_and(|h| (rotation_inliers as f64) < ratio * h as f64)
}

/// See `CAMERA_MOVED_RATIO`.
fn camera_moved(rotation_inliers: usize, homography_inliers: Option<usize>) -> bool {
    rotation_below(CAMERA_MOVED_RATIO, rotation_inliers, homography_inliers)
}

/// Enough support in absolute terms, and not dwarfed by the homography's (see
/// `CAMERA_MOVED_REJECT_RATIO`).
fn accepts_rotation(rotation_inliers: usize, homography_inliers: Option<usize>) -> bool {
    rotation_inliers >= MIN_INLIERS
        && !rotation_below(CAMERA_MOVED_REJECT_RATIO, rotation_inliers, homography_inliers)
}

fn row_major(h: &Matrix3<f64>) -> [f64; 9] {
    std::array::from_fn(|i| h[(i / 3, i % 3)])
}

/// Distance between two SINGLE descriptors (512 bits for AKAZE's MLDB).
fn hamming(a: &BitArray<64>, b: &BitArray<64>) -> u32 {
    a.bytes()
        .iter()
        .zip(b.bytes().iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

fn passes_lowe_ratio_test(best: u32, second_best: u32, ratio: f32) -> bool {
    // Ratio test: best must be clearly better than runner-up.
    // Multiply instead of dividing to avoid the div-by-zero special case.
    (best as f32) < ratio * (second_best as f32)
}

/// For each descriptor in `a`, find its best match in `b`.
/// Returns (index_in_a, index_in_b) for matches that pass the ratio test.
fn ratio_matches(a: &[BitArray<64>], b: &[BitArray<64>], ratio: f32) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();

    for (i, desc_a) in a.iter().enumerate() {
        let mut best = (u32::MAX, usize::MAX); // (distance, index)
        let mut second_best = u32::MAX;

        for (j, desc_b) in b.iter().enumerate() {
            let d = hamming(desc_a, desc_b);
            if d < best.0 {
                second_best = best.0;
                best = (d, j);
            } else if d < second_best {
                second_best = d;
            }
        }

        // Need at least 2 candidates in b for the test to mean anything
        if best.1 != usize::MAX
            && second_best != u32::MAX
            && passes_lowe_ratio_test(best.0, second_best, ratio)
        {
            matches.push((i, best.1));
        }
    }
    matches
}

/// Keep only matches where A→B and B→A agree (symmetric / cross-check).
fn symmetric_matches(a: &[BitArray<64>], b: &[BitArray<64>], ratio: f32) -> Vec<(usize, usize)> {
    let a_to_b = ratio_matches(a, b, ratio);
    let b_to_a = ratio_matches(b, a, ratio);

    a_to_b
        .into_iter()
        .filter(|&(i, j)| b_to_a.contains(&(j, i)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, RgbImage};
    use rand_core::{RngCore, SeedableRng};
    use nalgebra::Vector3;
    use rand_pcg::Pcg64;

    fn random_descriptors(n: usize, seed: u64) -> Vec<BitArray<64>> {
        let mut rng = Pcg64::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let mut bytes = [0u8; 64];
                rng.fill_bytes(&mut bytes);
                BitArray::new(bytes)
            })
            .collect()
    }

    fn flip_bits(d: &BitArray<64>, count: usize) -> BitArray<64> {
        let mut bytes = *d.bytes();
        for bit in 0..count {
            bytes[bit / 8] ^= 1 << (bit % 8);
        }
        BitArray::new(bytes)
    }

    #[test]
    fn hello_round_trips_the_name() {
        assert_eq!(hello("demo"), "hello from wasm, demo");
    }

    #[test]
    fn hamming_counts_differing_bits() {
        let a = BitArray::new([0xAA; 64]);
        assert_eq!(hamming(&a, &a), 0);
        assert_eq!(hamming(&a, &BitArray::new([0x55; 64])), 512);
        assert_eq!(hamming(&a, &flip_bits(&a, 13)), 13);
    }

    #[test]
    fn lowe_ratio_rejects_ambiguous_and_accepts_distinct() {
        assert!(passes_lowe_ratio_test(10, 100, 0.8));
        assert!(passes_lowe_ratio_test(0, 1, 0.8));
        assert!(!passes_lowe_ratio_test(85, 100, 0.8));
        assert!(!passes_lowe_ratio_test(0, 0, 0.8));
    }

    #[test]
    fn lowe_ratio_threshold_is_configurable() {
        assert!(passes_lowe_ratio_test(75, 100, 0.8));
        assert!(!passes_lowe_ratio_test(75, 100, 0.7));
    }

    #[test]
    fn ratio_matches_needs_two_candidates() {
        let a = random_descriptors(3, 1);
        assert!(ratio_matches(&a, &a[..1], DEFAULT_RATIO).is_empty());
        assert!(ratio_matches(&a, &[], DEFAULT_RATIO).is_empty());
    }

    #[test]
    fn looser_ratio_keeps_a_match_with_a_close_runner_up() {
        // b[0] is 30 bits from a[0], b[1] is 40 bits away: ratio 0.75, like two
        // near-identical chairs where the true match is only slightly better.
        let a = random_descriptors(1, 4);
        let b = vec![flip_bits(&a[0], 30), flip_bits(&a[0], 40)];
        assert!(ratio_matches(&a, &b, 0.7).is_empty());
        assert_eq!(ratio_matches(&a, &b, 0.8), vec![(0, 0)]);
    }

    #[test]
    fn symmetric_matches_recovers_a_permutation() {
        let a = random_descriptors(20, 2);
        let perm: Vec<usize> = (0..20).map(|i| (i * 7) % 20).collect();
        let b: Vec<_> = perm.iter().map(|&k| flip_bits(&a[k], 5)).collect();

        let mut got = symmetric_matches(&a, &b, DEFAULT_RATIO);
        got.sort();
        let mut expected: Vec<_> = perm.iter().enumerate().map(|(j, &i)| (i, j)).collect();
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn symmetric_matches_drops_one_directional_matches() {
        let x = random_descriptors(2, 3);
        // a[1] is a near-copy of a[0], so a[1] -> b[0] passes A→B but
        // B→A picks a[0] for b[0].
        let a = vec![x[0], flip_bits(&x[0], 10)];
        let b = vec![x[0], x[1]];

        assert_eq!(ratio_matches(&a, &b, DEFAULT_RATIO), vec![(0, 0), (1, 0)]);
        assert_eq!(symmetric_matches(&a, &b, DEFAULT_RATIO), vec![(0, 0)]);
    }

    #[test]
    fn decode_images_reads_png_and_reports_bad_index() {
        let mut png = Vec::new();
        RgbImage::from_pixel(4, 3, image::Rgb([10, 20, 30]))
            .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
            .unwrap();

        let imgs = decode_images(&[png.clone()]).unwrap();
        assert_eq!((imgs[0].width(), imgs[0].height()), (4, 3));

        let err = decode_images(&[png, b"not an image".to_vec()]).unwrap_err();
        assert!(err.starts_with("image 1:"), "{err}");
    }

    fn translation(dx: f64, dy: f64) -> Matrix3<f64> {
        Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0)
    }

    fn assert_matrix_close(a: &Matrix3<f64>, b: &Matrix3<f64>) {
        assert!((a - b).amax() < 1e-9, "{a} != {b}");
    }

    /// Camera turned right by `yaw` radians; columns are its axes in the previous camera's coords.
    fn yaw_rotation(yaw: f64) -> Matrix3<f64> {
        let (s, c) = yaw.sin_cos();
        Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
    }

    fn blank_images(n: usize, width: u32, height: u32) -> Vec<DynamicImage> {
        (0..n)
            .map(|_| DynamicImage::ImageRgb8(RgbImage::new(width, height)))
            .collect()
    }

    fn pair(from: usize, rotation: Option<Matrix3<f64>>) -> PairReport {
        PairReport {
            from,
            to: from + 1,
            matches: 0,
            inliers: if rotation.is_some() { 50 } else { 0 },
            homography_inliers: None,
            homography: None,
            rotation: rotation.as_ref().map(row_major),
            yaw_degrees: rotation.as_ref().map(cylinder::yaw_degrees),
            camera_moved: false,
        }
    }

    /// Pair rotation (camera i rays into camera i + 1 coords) when i + 1 is turned right by `yaw`.
    fn yaw_step(yaw: f64) -> Matrix3<f64> {
        yaw_rotation(yaw).transpose()
    }

    #[test]
    fn place_on_cylinder_uses_the_reference_focal_as_radius() {
        let imgs = blank_images(1, 400, 300);
        let placed = place_on_cylinder(&imgs, &[], 0.75);
        assert_eq!(placed.reference, 0);
        assert!((placed.radius - 300.0).abs() < 1e-12);
        assert!((placed.images[0].focal - 300.0).abs() < 1e-12);
        assert_eq!(placed.images[0].unplaced, None);
        assert_eq!(
            placed.images[0].outline.as_ref().unwrap().len(),
            4 * cylinder::OUTLINE_SAMPLES_PER_EDGE
        );
    }

    #[test]
    fn place_on_cylinder_shifts_each_image_by_its_yaw() {
        let imgs = blank_images(3, 400, 300);
        let yaw = 30f64.to_radians();
        let pairs = [pair(0, Some(yaw_step(yaw))), pair(1, Some(yaw_step(yaw)))];

        let placed = place_on_cylinder(&imgs, &pairs, 0.75);

        let base = placed.images[0].outline.as_ref().unwrap();
        for i in 1..3 {
            let moved = placed.images[i].outline.as_ref().unwrap();
            for (a, b) in base.iter().zip(moved) {
                assert!((b[0] - a[0] - placed.radius * yaw * i as f64).abs() < 1e-6);
                assert!((b[1] - a[1]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn place_on_cylinder_keeps_going_past_half_a_turn() {
        // 8 images 40° apart reach 280°; x must keep increasing rather than wrap to negative.
        let imgs = blank_images(8, 400, 300);
        let pairs: Vec<_> = (0..7).map(|i| pair(i, Some(yaw_step(40f64.to_radians())))).collect();

        let placed = place_on_cylinder(&imgs, &pairs, 0.75);
        let first_x: Vec<f64> = placed
            .images
            .iter()
            .map(|img| img.outline.as_ref().unwrap()[0][0])
            .collect();
        assert!(first_x.windows(2).all(|w| w[1] > w[0]), "{first_x:?}");
    }

    #[test]
    fn place_on_cylinder_places_the_longest_joined_run() {
        // Images 0 | 1-2-3 | 4: the middle run wins and image 1 becomes the reference.
        let imgs = blank_images(5, 400, 300);
        let step = Some(yaw_step(0.3));
        let pairs = [pair(0, None), pair(1, step), pair(2, step), pair(3, None)];

        let placed = place_on_cylinder(&imgs, &pairs, 0.75);

        assert_eq!(placed.reference, 1);
        for i in [0, 4] {
            let img = &placed.images[i];
            assert!(img.rotation.is_none() && img.outline.is_none());
            assert_eq!(img.unplaced, Some(Unplaced::OutsideGroup));
        }
        for i in 1..4 {
            assert!(placed.images[i].outline.is_some());
            assert_eq!(placed.images[i].unplaced, None);
        }
        let reference = Matrix3::from_row_slice(&placed.images[1].rotation.unwrap());
        assert!((reference - Matrix3::identity()).amax() < 1e-12);
    }

    #[test]
    fn place_on_cylinder_flags_a_camera_pointing_along_the_axis() {
        let imgs = blank_images(2, 400, 300);
        let (s, c) = (std::f64::consts::FRAC_PI_2).sin_cos();
        // Camera 1 pitched 90°, so its centre ray is the cylinder axis.
        let pitch = Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
        let placed = place_on_cylinder(&imgs, &[pair(0, Some(pitch))], 0.75);
        assert_eq!(placed.images[1].unplaced, Some(Unplaced::AlongAxis));
        assert!(placed.images[1].rotation.is_some());
    }

    #[test]
    fn largest_group_picks_the_longest_run_and_the_earliest_on_ties() {
        assert_eq!(largest_group(&[], 0), 0..0);
        assert_eq!(largest_group(&[], 1), 0..1);
        assert_eq!(largest_group(&[true, true], 3), 0..3);
        assert_eq!(largest_group(&[false, false], 3), 0..1);
        assert_eq!(largest_group(&[true, false, true, true], 5), 2..5);
        assert_eq!(largest_group(&[true, false, true], 4), 0..2);
    }

    #[test]
    fn chain_group_offsets_into_the_full_image_list() {
        let step = Some(yaw_step(0.2));
        let chained = chain_group(&[None, step, step, None], 1..4, 5);
        assert!(chained[0].is_none() && chained[4].is_none());
        assert!((chained[1].unwrap() - Matrix3::identity()).amax() < 1e-12);
        assert!((chained[3].unwrap() - yaw_rotation(0.4)).amax() < 1e-12);
        assert!(chain_group(&[], 0..0, 0).is_empty());
    }

    fn features_with_k(k: Matrix3<f64>) -> ImageFeatures {
        ImageFeatures {
            detected: 0,
            near_border: 0,
            keypoints: vec![],
            descriptors: vec![],
            to_full: Matrix3::identity(),
            k,
        }
    }

    /// Pixel in a 400x300, f = 300 camera at `t` (turned right by `yaw`) seeing world point `p`.
    fn project_world(p: Vector3<f64>, t: Vector3<f64>, yaw: f64) -> Option<(f32, f32)> {
        let k = cylinder::intrinsics(300.0, 400, 300);
        let q = k * (yaw_rotation(yaw).transpose() * (p - t));
        let (u, v) = (q.x / q.z, q.y / q.z);
        (q.z > 0.0 && (0.0..400.0).contains(&u) && (0.0..300.0).contains(&v)).then_some((u as f32, v as f32))
    }

    /// Floor points 1.5 m below the camera seen from two cameras `baseline` apart,
    /// the second turned right by `yaw`.
    fn floor_matches(baseline: f64, yaw: f64) -> Vec<((f32, f32), (f32, f32))> {
        let mut out = Vec::new();
        for gx in 0..30 {
            for gz in 0..30 {
                let p = Vector3::new(-3.0 + gx as f64 * 0.2, 1.5, 1.0 + gz as f64 * 0.15);
                let a = project_world(p, Vector3::zeros(), 0.0);
                let b = project_world(p, Vector3::new(baseline, 0.0, 0.0), yaw);
                if let (Some(a), Some(b)) = (a, b) {
                    out.push((a, b));
                }
            }
        }
        out
    }

    #[test]
    fn fit_pair_accepts_a_pure_rotation_without_flagging_movement() {
        let k = cylinder::intrinsics(300.0, 400, 300);
        let matches = floor_matches(0.0, 20f64.to_radians());
        assert!(matches.len() >= 100, "{}", matches.len());

        let report = fit_pair(0, &matches, &features_with_k(k), &features_with_k(k));

        assert_eq!(report.inliers, matches.len());
        assert!(report.rotation.is_some());
        assert!((report.yaw_degrees.unwrap() - 20.0).abs() < 0.1);
        assert!(!report.camera_moved);
    }

    #[test]
    fn fit_pair_flags_a_floor_seen_from_two_positions() {
        // 60 cm sideways with the visible floor 3-5 m away: a homography fits the
        // floor exactly, but the near floor shifts ~25 px more than the far floor,
        // which no rotation reproduces.
        let k = cylinder::intrinsics(300.0, 400, 300);
        let matches = floor_matches(0.6, 20f64.to_radians());
        assert!(matches.len() >= 100, "{}", matches.len());

        let report = fit_pair(0, &matches, &features_with_k(k), &features_with_k(k));

        assert_eq!(report.homography_inliers, Some(matches.len()));
        assert!(report.camera_moved, "{report:?}");
    }

    #[test]
    fn accepts_rotation_rejects_fits_dwarfed_by_the_homography() {
        // The 4->5 and 5->6 pairs from the venue test set: a rotation fitting
        // under a third of what a plane fits must not be chained.
        assert!(!accepts_rotation(22, Some(78)));
        assert!(!accepts_rotation(26, Some(81)));
        // The 1->2 pair at 66%: kept, with the camera-moved warning.
        assert!(accepts_rotation(58, Some(88)));
        assert!(camera_moved(58, Some(88)));

        assert!(accepts_rotation(40, Some(80)));
        assert!(!accepts_rotation(39, Some(80)));
        assert!(!accepts_rotation(MIN_INLIERS - 1, None));
        assert!(accepts_rotation(MIN_INLIERS, None));
    }

    #[test]
    fn camera_moved_compares_rotation_to_homography_support() {
        assert!(!camera_moved(95, Some(100)));
        assert!(!camera_moved(70, Some(100)));
        assert!(camera_moved(69, Some(100)));
        assert!(!camera_moved(0, None));
    }

    #[test]
    fn place_on_cylinder_scales_focal_with_the_long_side() {
        let imgs = vec![
            DynamicImage::ImageRgb8(RgbImage::new(300, 400)),
            DynamicImage::ImageRgb8(RgbImage::new(400, 300)),
        ];
        let placed = place_on_cylinder(&imgs, &[pair(0, None)], 0.5);
        assert!((placed.images[0].focal - 200.0).abs() < 1e-12);
        assert!((placed.images[1].focal - 200.0).abs() < 1e-12);
    }

    fn apply(h: &Matrix3<f64>, x: f64, y: f64) -> (f64, f64) {
        let p = h * Vector3::new(x, y, 1.0);
        (p.x / p.z, p.y / p.z)
    }

    #[test]
    fn feature_image_shrinks_long_side_and_keeps_aspect() {
        let landscape = DynamicImage::ImageRgb8(RgbImage::new(2560, 1920));
        let small = feature_image(&landscape, 1280);
        assert_eq!((small.width(), small.height()), (1280, 960));
        assert!(matches!(small, DynamicImage::ImageLuma8(_)));

        let portrait = DynamicImage::ImageRgb8(RgbImage::new(1920, 2560));
        let small = feature_image(&portrait, 1280);
        assert_eq!((small.width(), small.height()), (960, 1280));

        let small = feature_image(&portrait, 640);
        assert_eq!((small.width(), small.height()), (480, 640));
    }

    #[test]
    fn feature_image_never_upscales() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(640, 480));
        let small = feature_image(&img, 1280);
        assert_eq!((small.width(), small.height()), (640, 480));
        assert!(matches!(small, DynamicImage::ImageLuma8(_)));
    }

    #[test]
    fn scale_to_full_aligns_pixel_edges() {
        let s = scale_to_full(1280, 960, 4000, 3000);
        let (x0, y0) = apply(&s, -0.5, -0.5);
        let (x1, y1) = apply(&s, 1279.5, 959.5);
        assert!((x0 + 0.5).abs() < 1e-9 && (y0 + 0.5).abs() < 1e-9);
        assert!((x1 - 3999.5).abs() < 1e-9 && (y1 - 2999.5).abs() < 1e-9);
    }

    #[test]
    fn scale_to_full_is_identity_without_downscale() {
        assert_matrix_close(&scale_to_full(640, 480, 640, 480), &Matrix3::identity());
    }

    #[test]
    fn lift_homography_scales_a_translation() {
        let s = scale_to_full(1000, 750, 3000, 2250);
        let lifted = lift_homography(&translation(-10.0, 4.0), &s, &s).unwrap();
        assert_matrix_close(&lifted, &translation(-30.0, 12.0));
    }

    #[test]
    fn lift_homography_commutes_with_rescaling() {
        let h = Matrix3::new(0.98, 0.03, -150.0, -0.01, 1.01, 6.0, 4e-5, -1e-5, 1.0);
        let s_a = scale_to_full(1280, 960, 4000, 3000);
        let s_b = scale_to_full(1280, 720, 3840, 2160);
        let lifted = lift_homography(&h, &s_a, &s_b).unwrap();

        // Feature point -> original in A -> lifted H must equal feature H -> original in B.
        let (fx, fy) = (400.0, 300.0);
        let (ax, ay) = apply(&s_a, fx, fy);
        let (hx, hy) = apply(&h, fx, fy);
        let (want_x, want_y) = apply(&s_b, hx, hy);
        let (got_x, got_y) = apply(&lifted, ax, ay);
        assert!((got_x - want_x).abs() < 1e-6 && (got_y - want_y).abs() < 1e-6);
    }

    fn keypoint(x: f32, y: f32, response: f32) -> akaze::KeyPoint {
        akaze::KeyPoint {
            point: (x, y),
            response,
            size: 1.0,
            octave: 0,
            class_id: 0,
            angle: 0.0,
        }
    }

    fn placement(index: usize, outline: Option<Vec<[f64; 2]>>) -> ImagePlacement {
        ImagePlacement {
            index,
            width: 4,
            height: 3,
            focal: 3.0,
            rotation: outline.as_ref().map(|_| row_major(&Matrix3::identity())),
            unplaced: outline.is_none().then_some(Unplaced::OutsideGroup),
            outline,
        }
    }

    #[test]
    fn fit_canvas_bounds_all_outlines_and_moves_them_into_canvas_pixels() {
        let mut images = vec![
            placement(0, Some(vec![[-100.0, -50.0], [0.0, 50.0]])),
            placement(1, None),
            placement(2, Some(vec![[50.0, -20.0], [300.0, 150.0]])),
        ];
        let canvas = fit_canvas(&mut images, 4096).unwrap();

        assert_eq!((canvas.width, canvas.height), (400, 200));
        let near = images[0].outline.as_ref().unwrap()[0];
        let far = images[2].outline.as_ref().unwrap()[1];
        assert!((near[0] + 0.5).abs() < 1e-9 && (near[1] + 0.5).abs() < 1e-9);
        assert!((far[0] - 399.5).abs() < 1e-9 && (far[1] - 199.5).abs() < 1e-9);
        assert!(images[1].outline.is_none());
    }

    #[test]
    fn fit_canvas_scales_outlines_with_the_cap() {
        let mut images = vec![placement(0, Some(vec![[0.0, 0.0], [8000.0, 2000.0]]))];
        let canvas = fit_canvas(&mut images, 4000).unwrap();
        assert_eq!((canvas.width, canvas.height), (4000, 1000));
        let far = images[0].outline.as_ref().unwrap()[1];
        assert!((far[0] - 3999.5).abs() < 1e-9 && (far[1] - 999.5).abs() < 1e-9);
    }

    #[test]
    fn fit_canvas_is_none_when_nothing_is_placed() {
        let mut images = vec![placement(0, None)];
        assert!(fit_canvas(&mut images, 4096).is_none());
        assert!(fit_canvas(&mut [], 4096).is_none());
    }

    fn solid_image(rgb: [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(400, 300, image::Rgb(rgb)))
    }

    fn rgba_at(pano: &Panorama, canvas: &Canvas, cylinder_xy: [f64; 2]) -> [u8; 4] {
        let [u, v] = canvas.to_canvas(cylinder_xy).map(|c| c.round() as usize);
        let o = (v * canvas.width as usize + u) * 4;
        std::array::from_fn(|i| pano.pixels[o + i])
    }

    #[test]
    fn render_panorama_fills_the_canvas_from_each_placed_image() {
        let imgs = vec![solid_image([255, 0, 0]), solid_image([0, 0, 255])];
        let pairs = [pair(0, Some(yaw_step(30f64.to_radians())))];
        let mut place = place_on_cylinder(&imgs, &pairs, 0.75);
        let canvas = fit_canvas(&mut place.images, 4096);
        let alignment = RowAlignment {
            reference: place.reference,
            radius: place.radius,
            canvas,
            images: place.images,
            pairs: pairs.into(),
        };

        let pano = render_panorama(&imgs, &alignment).unwrap();
        let c = alignment.canvas.unwrap();
        assert_eq!((pano.width, pano.height), (c.width, c.height));
        assert_eq!(pano.pixels.len(), c.width as usize * c.height as usize * 4);

        // Each image spans ±33.7° around its own centre at f = 300, w = 400.
        let r = alignment.radius;
        assert_eq!(rgba_at(&pano, &c, [r * (-20f64).to_radians(), 0.0]), [255, 0, 0, 255]);
        assert_eq!(rgba_at(&pano, &c, [r * 50f64.to_radians(), 0.0]), [0, 0, 255, 255]);
    }

    #[test]
    fn render_panorama_skips_unplaced_images() {
        let imgs = vec![solid_image([255, 0, 0]), solid_image([0, 0, 255])];
        let pairs = [pair(0, None)];
        let mut place = place_on_cylinder(&imgs, &pairs, 0.75);
        let canvas = fit_canvas(&mut place.images, 4096);
        let alignment = RowAlignment {
            reference: place.reference,
            radius: place.radius,
            canvas,
            images: place.images,
            pairs: pairs.into(),
        };

        let pano = render_panorama(&imgs, &alignment).unwrap();
        assert!(pano.pixels.chunks(4).all(|p| p[3] == 0 || p == [255, 0, 0, 255]));
        assert!(pano.pixels.chunks(4).any(|p| p[3] == 255));
    }

    #[test]
    fn render_panorama_is_none_without_a_canvas() {
        let alignment = RowAlignment {
            reference: 0,
            radius: 3.0,
            canvas: None,
            images: vec![placement(0, None)],
            pairs: vec![],
        };
        assert!(render_panorama(&blank_images(1, 4, 3), &alignment).is_none());
    }

    #[test]
    fn rgb_view_borrows_rgb_and_converts_everything_else() {
        let rgb = DynamicImage::ImageRgb8(RgbImage::new(2, 2));
        let rgba = DynamicImage::ImageRgba8(image::RgbaImage::new(2, 2));
        assert!(matches!(rgb_view(&rgb), Cow::Borrowed(_)));
        assert!(matches!(rgb_view(&rgba), Cow::Owned(_)));
    }

    #[test]
    fn placed_row_canvas_spans_the_chained_yaw() {
        // Three 400x300 images 30° apart at f = 300: the canvas must cover the
        // reference's own width plus 60° of yaw at radius 300.
        let imgs = blank_images(3, 400, 300);
        let step = Some(yaw_step(30f64.to_radians()));
        let pairs = [pair(0, step), pair(1, step)];
        let mut placed = place_on_cylinder(&imgs, &pairs, 0.75);
        let canvas = fit_canvas(&mut placed.images, 4096).unwrap();

        let reference_width = 2.0 * 300.0 * (200.0f64 / 300.0).atan();
        let expected = reference_width + 300.0 * 60f64.to_radians();
        assert!((canvas.width as f64 - expected).abs() <= 1.0, "{} vs {expected}", canvas.width);
        for img in &placed.images {
            for p in img.outline.as_ref().unwrap() {
                assert!(p[0] >= -0.5 - 1e-9 && p[0] <= canvas.width as f64 - 0.5 + 1e-9);
                assert!(p[1] >= -0.5 - 1e-9 && p[1] <= canvas.height as f64 - 0.5 + 1e-9);
            }
        }
    }

    #[test]
    fn strongest_keypoints_keeps_everything_under_the_cap() {
        let kps = [keypoint(5.0, 5.0, 0.1), keypoint(6.0, 5.0, 0.9)];
        assert_eq!(strongest_keypoints(&kps, 2), vec![0, 1]);
        assert!(strongest_keypoints(&[], 10).is_empty());
    }

    #[test]
    fn strongest_keypoints_keeps_highest_response_in_index_order() {
        let kps: Vec<_> = [0.5, 0.9, 0.1, 0.7, 0.3]
            .iter()
            .map(|&r| keypoint(0.0, 0.0, r))
            .collect();
        assert_eq!(strongest_keypoints(&kps, 3), vec![0, 1, 3]);
    }

    #[test]
    fn resolve_options_defaults_when_empty() {
        let s = resolve_options(AlignOptions::default()).unwrap();
        assert_eq!(
            s,
            Settings {
                matching: MatchSettings::default(),
                output_max_dim: DEFAULT_OUTPUT_MAX_DIM,
            }
        );
    }

    #[test]
    fn resolve_options_applies_overrides() {
        let s = resolve_options(AlignOptions {
            focal_per_long_side: Some(0.4),
            ratio: Some(0.75),
            max_features: Some(6000),
            feature_max_dim: Some(1600),
            output_max_dim: Some(2048),
            cylindrical_matching: Some(false),
        })
        .unwrap();
        assert!(!s.matching.cylindrical);
        assert!((s.matching.focal_scale - 0.4).abs() < 1e-12);
        assert!((s.matching.ratio - 0.75).abs() < 1e-6);
        assert_eq!(s.matching.max_features, 6000);
        assert_eq!(s.matching.feature_max_dim, 1600);
        assert_eq!(s.output_max_dim, 2048);
    }

    #[test]
    fn cylindrical_matching_is_on_by_default() {
        assert!(resolve_options(AlignOptions::default()).unwrap().matching.cylindrical);
    }

    #[test]
    fn descriptor_radius_scales_with_keypoint_size() {
        let kp = keypoint(0.0, 0.0, 1.0);
        let expected = 10.0 * 0.5 * kp.size as f64 * std::f64::consts::SQRT_2;
        assert!((descriptor_radius(&kp, 10) - expected).abs() < 1e-12);
    }

    /// 8 px blocks of seeded random gray: plenty of corners for AKAZE.
    fn blocky_texture(width: u32, height: u32, seed: u64) -> DynamicImage {
        let mut rng = Pcg64::seed_from_u64(seed);
        let cols = width.div_ceil(8) as usize;
        let blocks: Vec<u8> = (0..cols * height.div_ceil(8) as usize)
            .map(|_| (rng.next_u32() & 0xFF) as u8)
            .collect();
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
            let g = blocks[(y / 8) as usize * cols + (x / 8) as usize];
            image::Rgb([g, g, g])
        }))
    }

    #[test]
    fn unrolled_extraction_returns_planar_keypoints_clear_of_the_border() {
        let img = blocky_texture(320, 240, 7);
        let settings = MatchSettings {
            focal_scale: 0.5,
            cylindrical: true,
            max_features: 400,
            ..MatchSettings::default()
        };
        let f = extract_features(&Akaze::default(), &img, &settings);

        assert!(!f.keypoints.is_empty());
        assert!(f.keypoints.len() <= 400);
        assert_eq!(f.keypoints.len(), f.descriptors.len());
        assert!(f.detected >= f.near_border + f.keypoints.len());
        for kp in &f.keypoints {
            let (x, y) = kp.point;
            assert!((-0.5..=319.5).contains(&x) && (-0.5..=239.5).contains(&y), "{x}, {y}");
        }
    }

    #[test]
    fn planar_extraction_drops_nothing_at_the_border() {
        let img = blocky_texture(320, 240, 7);
        let settings = MatchSettings {
            cylindrical: false,
            max_features: usize::MAX,
            ..MatchSettings::default()
        };
        let f = extract_features(&Akaze::default(), &img, &settings);
        assert_eq!(f.near_border, 0);
        assert_eq!(f.detected, f.keypoints.len());
    }

    #[test]
    fn resolve_options_rejects_out_of_range_values() {
        let bad = [
            AlignOptions { focal_per_long_side: Some(0.0), ..Default::default() },
            AlignOptions { focal_per_long_side: Some(f64::NAN), ..Default::default() },
            AlignOptions { ratio: Some(0.0), ..Default::default() },
            AlignOptions { ratio: Some(1.5), ..Default::default() },
            AlignOptions { ratio: Some(f32::NAN), ..Default::default() },
            AlignOptions { max_features: Some(19), ..Default::default() },
            AlignOptions { feature_max_dim: Some(100), ..Default::default() },
            AlignOptions { output_max_dim: Some(63), ..Default::default() },
        ];
        for o in bad {
            let desc = format!("{o:?}");
            assert!(resolve_options(o).is_err(), "{desc} should be rejected");
        }
    }

    #[test]
    fn row_major_flattens_by_rows() {
        let h = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        assert_eq!(row_major(&h), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}

// JsValue only exists inside a JS host, so these run under `wasm-pack test`, not native `cargo test`.
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;
    use js_sys::{Array, Reflect};
    use wasm_bindgen_test::wasm_bindgen_test;

    fn get(obj: &JsValue, key: &str) -> JsValue {
        Reflect::get(obj, &JsValue::from_str(key)).unwrap()
    }

    fn first(obj: &JsValue, key: &str) -> JsValue {
        Array::from(&get(obj, key)).get(0)
    }

    fn serialize(alignment: &RowAlignment) -> JsValue {
        let Ok(js) = to_js(alignment) else {
            panic!("to_js failed");
        };
        js
    }

    #[wasm_bindgen_test]
    fn none_fields_cross_the_boundary_as_null() {
        let js = serialize(&RowAlignment {
            reference: 0,
            radius: 3.0,
            canvas: None,
            images: vec![ImagePlacement {
                index: 1,
                width: 4,
                height: 3,
                focal: 3.0,
                rotation: None,
                outline: None,
                unplaced: Some(Unplaced::OutsideGroup),
            }],
            pairs: vec![PairReport {
                from: 0,
                to: 1,
                matches: 7,
                inliers: 3,
                homography_inliers: None,
                homography: None,
                rotation: None,
                yaw_degrees: None,
                camera_moved: false,
            }],
        });

        assert!(get(&js, "canvas").is_null());

        let image = first(&js, "images");
        assert!(get(&image, "rotation").is_null());
        assert!(get(&image, "outline").is_null());
        assert_eq!(get(&image, "unplaced").as_string().as_deref(), Some("outsideGroup"));

        let pair = first(&js, "pairs");
        assert!(get(&pair, "homographyInliers").is_null());
        assert!(get(&pair, "homography").is_null());
        assert!(get(&pair, "rotation").is_null());
        assert!(get(&pair, "yawDegrees").is_null());
    }

    #[wasm_bindgen_test]
    fn some_fields_keep_their_values_and_camel_case_keys() {
        let js = serialize(&RowAlignment {
            reference: 2,
            radius: 3.0,
            canvas: Some(Canvas {
                width: 400,
                height: 200,
                min_x: -100.0,
                min_y: -50.0,
                scale: 0.5,
            }),
            images: vec![ImagePlacement {
                index: 0,
                width: 4,
                height: 3,
                focal: 3.0,
                rotation: Some(row_major(&Matrix3::identity())),
                outline: Some(vec![[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]),
                unplaced: None,
            }],
            pairs: vec![PairReport {
                from: 0,
                to: 1,
                matches: 40,
                inliers: 25,
                homography_inliers: Some(38),
                homography: Some(row_major(&Matrix3::identity())),
                rotation: Some(row_major(&Matrix3::identity())),
                yaw_degrees: Some(12.5),
                camera_moved: true,
            }],
        });

        assert_eq!(get(&js, "radius").as_f64(), Some(3.0));
        assert_eq!(get(&js, "reference").as_f64(), Some(2.0));

        let canvas = get(&js, "canvas");
        assert_eq!(get(&canvas, "width").as_f64(), Some(400.0));
        assert_eq!(get(&canvas, "height").as_f64(), Some(200.0));
        assert_eq!(get(&canvas, "minX").as_f64(), Some(-100.0));
        assert_eq!(get(&canvas, "scale").as_f64(), Some(0.5));

        let image = first(&js, "images");
        assert_eq!(Array::from(&get(&image, "rotation")).length(), 9);
        let outline = Array::from(&get(&image, "outline"));
        assert_eq!(outline.length(), 4);
        assert_eq!(Array::from(&outline.get(2)).get(0).as_f64(), Some(4.0));

        assert!(get(&image, "unplaced").is_null());

        let pair = first(&js, "pairs");
        assert_eq!(get(&pair, "inliers").as_f64(), Some(25.0));
        assert_eq!(get(&pair, "homographyInliers").as_f64(), Some(38.0));
        assert_eq!(get(&pair, "cameraMoved").as_bool(), Some(true));
        assert_eq!(get(&pair, "yawDegrees").as_f64(), Some(12.5));
        assert_eq!(Array::from(&get(&pair, "homography")).length(), 9);
    }

    fn js_object(entries: &[(&str, JsValue)]) -> JsValue {
        let obj = js_sys::Object::new();
        for (key, value) in entries {
            Reflect::set(&obj, &JsValue::from_str(key), value).unwrap();
        }
        obj.into()
    }

    #[wasm_bindgen_test]
    fn missing_options_use_defaults() {
        let defaults = resolve_options(AlignOptions::default()).unwrap();
        assert_eq!(parse_options(&JsValue::UNDEFINED), Ok(defaults));
        assert_eq!(parse_options(&JsValue::NULL), Ok(defaults));
        assert_eq!(parse_options(&js_object(&[])), Ok(defaults));
    }

    #[wasm_bindgen_test]
    fn options_read_camel_case_keys_and_treat_null_as_default() {
        let s = parse_options(&js_object(&[
            ("focalPerLongSide", JsValue::from_f64(0.4)),
            ("ratio", JsValue::from_f64(0.75)),
            ("maxFeatures", JsValue::from_f64(6000.0)),
            ("featureMaxDim", JsValue::NULL),
            ("cylindricalMatching", JsValue::FALSE),
        ]))
        .unwrap();
        assert!(!s.matching.cylindrical);
        assert!((s.matching.focal_scale - 0.4).abs() < 1e-12);
        assert!((s.matching.ratio - 0.75).abs() < 1e-6);
        assert_eq!(s.matching.max_features, 6000);
        assert_eq!(s.matching.feature_max_dim, DEFAULT_FEATURE_MAX_DIM);
    }

    fn empty_alignment() -> RowAlignment {
        RowAlignment {
            reference: 0,
            radius: 3.0,
            canvas: None,
            images: vec![],
            pairs: vec![],
        }
    }

    #[wasm_bindgen_test]
    fn stitch_result_carries_pixels_as_a_clamped_array() {
        let pano = Panorama {
            width: 2,
            height: 1,
            pixels: vec![1, 2, 3, 255, 4, 5, 6, 0],
        };
        let Ok(js) = stitch_to_js(&empty_alignment(), Some(&pano)) else {
            panic!("stitch_to_js failed");
        };

        assert_eq!(get(&get(&js, "alignment"), "radius").as_f64(), Some(3.0));
        let panorama = get(&js, "panorama");
        assert_eq!(get(&panorama, "width").as_f64(), Some(2.0));
        assert_eq!(get(&panorama, "height").as_f64(), Some(1.0));
        let pixels = get(&panorama, "pixels");
        assert!(pixels.is_instance_of::<js_sys::Uint8ClampedArray>());
        assert_eq!(js_sys::Uint8ClampedArray::from(pixels).to_vec(), pano.pixels);
    }

    #[wasm_bindgen_test]
    fn stitch_result_panorama_is_null_when_nothing_was_placed() {
        let Ok(js) = stitch_to_js(&empty_alignment(), None) else {
            panic!("stitch_to_js failed");
        };
        assert!(get(&js, "panorama").is_null());
        assert!(get(&get(&js, "alignment"), "canvas").is_null());
    }

    #[wasm_bindgen_test]
    fn options_with_wrong_types_are_rejected() {
        let err = parse_options(&js_object(&[("ratio", JsValue::from_str("0.8"))])).unwrap_err();
        assert!(err.starts_with("invalid options"), "{err}");
    }
}
