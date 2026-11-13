//! Pure-rotation camera model and projection onto a cylinder around the camera.
//!
//! Camera ray coords: +x right, +y down, +z forward (same axes as pixel coords).
//! Cylinder coords: `theta` is yaw in radians (0 along the reference image's
//! optical axis, positive to the right), `height` is the ray's height on a unit-radius cylinder
//! (positive down). Multiplying both by the cylinder radius gives cylinder
//! pixels; `canvas.rs` maps those onto the output canvas.

use image::{GrayImage, Luma};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::{PI, TAU};

/// Border points sampled per image edge for outlines; edges curve on the cylinder.
pub const OUTLINE_SAMPLES_PER_EDGE: usize = 16;

/// Pinhole intrinsics with the principal point at the image centre. Pixel centres
/// are at integer coords, so the centre is `(size - 1) / 2`.
pub fn intrinsics(focal: f64, width: u32, height: u32) -> Matrix3<f64> {
    let (cx, cy) = ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0);
    Matrix3::new(focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0)
}

/// `pairwise[i]` rotates camera-i rays into camera-(i+1) coords. Returns, per
/// image, the rotation from its camera coords into the first image's. Once a
/// pair is missing, every later image is unreachable and gets `None`.
pub fn chain_rotations(pairwise: &[Option<Matrix3<f64>>]) -> Vec<Option<Matrix3<f64>>> {
    let mut current = Some(Matrix3::identity());
    let mut out = vec![current];
    for r in pairwise {
        // d_{i+1} = R d_i, so d_0 = G_i d_i = G_i R^T d_{i+1}.
        current = current.zip(*r).map(|(g, r)| g * r.transpose());
        out.push(current);
    }
    out
}

/// Yaw of camera b relative to camera a, in degrees (positive = turned right),
/// given the rotation taking a-rays into b coords.
pub fn yaw_degrees(r_ab: &Matrix3<f64>) -> f64 {
    // b's optical axis in a's coords is R^T e_z, i.e. row 2 of R.
    r_ab[(2, 0)].atan2(r_ab[(2, 2)]).to_degrees()
}

/// Maps `angle` into [-π, π).
pub fn wrap_angle(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

/// `(theta, height)` for a ray in reference coords. `None` for rays (near)
/// parallel to the cylinder axis, which never hit the cylinder wall.
pub fn ray_to_cylinder(d: &Vector3<f64>) -> Option<(f64, f64)> {
    let r = d.x.hypot(d.z);
    if r < 1e-9 * d.norm().max(1e-300) {
        return None;
    }
    Some((d.x.atan2(d.z), d.y / r))
}

/// Image border sampled clockwise from the top-left pixel edge, on the cylinder
/// in cylinder pixels (`radius * theta`, `radius * height`), plus the yaw of the
/// image centre.
///
/// `theta` is unwrapped to stay continuous across ±π: the centre lands within π of
/// `near_theta`, and every border point within π of the centre. Pass the previous
/// image's centre yaw so a row longer than 180° keeps going instead of jumping.
pub fn cylinder_outline(
    rotation: &Matrix3<f64>,
    k: &Matrix3<f64>,
    width: u32,
    height: u32,
    radius: f64,
    near_theta: f64,
) -> Option<(Vec<[f64; 2]>, f64)> {
    let to_ray = rotation * k.try_inverse()?;
    let (w, h) = (width as f64, height as f64);

    let (centre_theta, _) =
        ray_to_cylinder(&(to_ray * Vector3::new((w - 1.0) / 2.0, (h - 1.0) / 2.0, 1.0)))?;
    let centre_theta = near_theta + wrap_angle(centre_theta - near_theta);

    // Outer pixel edges sit half a pixel outside the first and last centres.
    let (left, top, right, bottom) = (-0.5, -0.5, w - 0.5, h - 0.5);
    let n = OUTLINE_SAMPLES_PER_EDGE;
    let lerp = |a: f64, b: f64, i: usize| a + (b - a) * i as f64 / n as f64;
    let border = (0..n)
        .map(|i| (lerp(left, right, i), top))
        .chain((0..n).map(|i| (right, lerp(top, bottom, i))))
        .chain((0..n).map(|i| (lerp(right, left, i), bottom)))
        .chain((0..n).map(|i| (left, lerp(bottom, top, i))));

    let outline = border
        .map(|(x, y)| {
            let (theta, height) = ray_to_cylinder(&(to_ray * Vector3::new(x, y, 1.0)))?;
            let theta = centre_theta + wrap_angle(theta - centre_theta);
            Some([radius * theta, radius * height])
        })
        .collect::<Option<Vec<_>>>()?;

    Some((outline, centre_theta))
}

/// One image unrolled onto a cylinder around its own optical axis, at its own
/// focal length, for feature matching only.
///
/// Pure yaw between two unrolled images is a horizontal shift, so a scene patch
/// looks the same in both. On the plane, the same patch is skewed in opposite
/// directions when it sits at opposite edges of two wide-lens shots, which
/// AKAZE's descriptors don't tolerate.
///
/// Both images put pixel centres at integer coords. Unrolled pixel `(u, v)` sits
/// at yaw `(u - cu) / focal` and cylinder height `(v - cv) / focal`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CylinderWarp {
    focal: f64,
    width: u32,
    height: u32,
    cx: f64,
    cy: f64,
    cu: f64,
    cv: f64,
    pub out_width: u32,
    pub out_height: u32,
}

impl CylinderWarp {
    /// `focal` is in pixels of the `width x height` planar image.
    pub fn new(focal: f64, width: u32, height: u32) -> Self {
        let out_width = ((2.0 * Self::half_yaw(focal, width) * focal).ceil() as u32).max(1);
        Self {
            focal,
            width,
            height,
            cx: (width as f64 - 1.0) / 2.0,
            cy: (height as f64 - 1.0) / 2.0,
            cu: (out_width as f64 - 1.0) / 2.0,
            cv: (height as f64 - 1.0) / 2.0,
            out_width,
            // The centre column keeps planar heights unchanged and every other
            // column shrinks toward the middle, so the height never grows.
            out_height: height,
        }
    }

    /// Yaw from the optical axis to the image's outer left/right pixel edge.
    fn half_yaw(focal: f64, width: u32) -> f64 {
        (width as f64 / 2.0 / focal).atan()
    }

    /// Unrolled pixel coords to planar pixel coords.
    pub fn to_plane(&self, u: f64, v: f64) -> (f64, f64) {
        let theta = (u - self.cu) / self.focal;
        let height = (v - self.cv) / self.focal;
        (
            self.cx + self.focal * theta.tan(),
            self.cy + self.focal * height / theta.cos(),
        )
    }

    /// Whether a disc of radius `r` around unrolled `(u, v)` lies entirely on
    /// the image, away from the black fill outside its curved border.
    pub fn contains_disc(&self, u: f64, v: f64, r: f64) -> bool {
        let du = (u - self.cu).abs() + r;
        if du > self.focal * Self::half_yaw(self.focal, self.width) {
            return false;
        }
        // Top and bottom edges sit at ±(height / 2) cos(yaw), so the binding
        // limit is at the disc's side farthest from the centre column.
        let half_h = self.height as f64 / 2.0 * (du / self.focal).cos();
        (v - self.cv).abs() + r <= half_h
    }

    /// Bilinear resample of `img` (which must be `width x height`). Pixels that
    /// fall outside the planar image are black.
    pub fn warp_gray(&self, img: &GrayImage) -> GrayImage {
        let (w, h) = (self.width as f64, self.height as f64);
        GrayImage::from_fn(self.out_width, self.out_height, |u, v| {
            let (x, y) = self.to_plane(u as f64, v as f64);
            if x < -0.5 || y < -0.5 || x > w - 0.5 || y > h - 0.5 {
                return Luma([0]);
            }
            Luma([bilinear_gray(img, x, y).round() as u8])
        })
    }
}

/// Gray value at sub-pixel `(x, y)` (pixel centres at integers), clamped to the border.
fn bilinear_gray(img: &GrayImage, x: f64, y: f64) -> f64 {
    let (max_x, max_y) = (img.width() - 1, img.height() - 1);
    let x = x.clamp(0.0, max_x as f64);
    let y = y.clamp(0.0, max_y as f64);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(max_x), (y0 + 1).min(max_y));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let p = |px: u32, py: u32| img.get_pixel(px, py).0[0] as f64;
    let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
    let bottom = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
    top * (1.0 - fy) + bottom * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Camera turned right by `yaw` (radians): columns are its axes in the
    /// original camera's coords.
    fn yaw_rotation(yaw: f64) -> Matrix3<f64> {
        let (s, c) = yaw.sin_cos();
        Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
    }

    fn pitch_rotation(pitch: f64) -> Matrix3<f64> {
        let (s, c) = pitch.sin_cos();
        Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
    }

    fn assert_close(a: &Matrix3<f64>, b: &Matrix3<f64>, tol: f64) {
        assert!((a - b).amax() < tol, "{a} != {b}");
    }

    #[test]
    fn warp_keeps_the_centre_and_maps_edges_to_edges() {
        let warp = CylinderWarp::new(600.0, 1125, 1500);
        // 2 * 600 * atan(562.5 / 600) = 903.7 px.
        assert_eq!((warp.out_width, warp.out_height), (904, 1500));

        let (x, y) = warp.to_plane(warp.cu, warp.cv);
        assert!((x - 562.0).abs() < 1e-9 && (y - 749.5).abs() < 1e-9);

        let half = 600.0 * (562.5f64 / 600.0).atan();
        let (left, _) = warp.to_plane(warp.cu - half, warp.cv);
        let (right, _) = warp.to_plane(warp.cu + half, warp.cv);
        assert!((left + 0.5).abs() < 1e-9 && (right - 1124.5).abs() < 1e-9);

        // At the centre column heights are unchanged.
        let (_, top) = warp.to_plane(warp.cu, -0.5);
        assert!((top + 0.5).abs() < 1e-9);
    }

    #[test]
    fn pure_yaw_becomes_a_horizontal_shift_after_unrolling() {
        let (w, h, f) = (400, 300, 300.0);
        let k = intrinsics(f, w, h);
        let warp = CylinderWarp::new(f, w, h);
        let yaw = 20f64.to_radians();
        // Rays in camera a, re-expressed in camera b turned right by `yaw`.
        let a_to_b = yaw_rotation(yaw).transpose();

        for &(u, v) in &[(150.0, 100.0), (230.0, 40.0), (200.0, 260.0)] {
            let (xa, ya) = warp.to_plane(u, v);
            let pb = k * a_to_b * k.try_inverse().unwrap() * Vector3::new(xa, ya, 1.0);
            let (xb, yb) = warp.to_plane(u - f * yaw, v);
            assert!((pb.x / pb.z - xb).abs() < 1e-6 && (pb.y / pb.z - yb).abs() < 1e-6, "({u}, {v})");
        }
    }

    #[test]
    fn contains_disc_rejects_the_curved_border() {
        let warp = CylinderWarp::new(300.0, 400, 300);
        assert!(warp.contains_disc(warp.cu, warp.cv, 20.0));
        // Top edge at the centre column is at v = -0.5.
        assert!(warp.contains_disc(warp.cu, 10.0, 10.0));
        assert!(!warp.contains_disc(warp.cu, 5.0, 10.0));
        // Near the left edge the top border has dropped, so the same v is outside.
        assert!(!warp.contains_disc(15.0, 15.0, 10.0));
        assert!(!warp.contains_disc(-5.0, warp.cv, 1.0));
    }

    #[test]
    fn warp_gray_fills_outside_black_and_keeps_flat_regions_flat() {
        let img = GrayImage::from_pixel(400, 300, Luma([180]));
        let warp = CylinderWarp::new(300.0, 400, 300);
        let out = warp.warp_gray(&img);
        assert_eq!(out.dimensions(), (warp.out_width, warp.out_height));
        assert_eq!(out.get_pixel(warp.cu.round() as u32, warp.cv.round() as u32).0, [180]);
        // Top-left corner lies above the curved top edge.
        assert_eq!(out.get_pixel(0, 0).0, [0]);
    }

    #[test]
    fn warp_gray_moves_content_with_the_mapping() {
        let img = GrayImage::from_fn(400, 300, |x, _| Luma([(x * 255 / 399) as u8]));
        let warp = CylinderWarp::new(300.0, 400, 300);
        let out = warp.warp_gray(&img);
        let (u, v) = (60u32, 150u32);
        let (x, _) = warp.to_plane(u as f64, v as f64);
        let expected = x * 255.0 / 399.0;
        assert!((out.get_pixel(u, v).0[0] as f64 - expected).abs() <= 1.5);
    }

    #[test]
    fn intrinsics_centre_the_principal_point() {
        let k = intrinsics(3000.0, 4000, 3000);
        assert_close(
            &k,
            &Matrix3::new(3000.0, 0.0, 1999.5, 0.0, 3000.0, 1499.5, 0.0, 0.0, 1.0),
            1e-12,
        );
    }

    #[test]
    fn yaw_degrees_ignores_a_small_tilt() {
        let r_ab = (yaw_rotation(35f64.to_radians()) * pitch_rotation(0.03)).transpose();
        assert!((yaw_degrees(&r_ab) - 35.0).abs() < 0.5);
    }

    #[test]
    fn yaw_degrees_reports_right_turn_as_positive() {
        assert!((yaw_degrees(&yaw_rotation(0.5).transpose()) - 0.5f64.to_degrees()).abs() < 1e-9);
        assert!((yaw_degrees(&yaw_rotation(-0.2).transpose()) + 0.2f64.to_degrees()).abs() < 1e-9);
    }

    #[test]
    fn chain_rotations_accumulates_yaw() {
        let step = Some(yaw_rotation(0.3).transpose());
        let chain = chain_rotations(&[step, step, step]);
        assert_eq!(chain.len(), 4);
        for (i, g) in chain.iter().enumerate() {
            assert_close(g.as_ref().unwrap(), &yaw_rotation(0.3 * i as f64), 1e-12);
        }
    }

    #[test]
    fn chain_rotations_composes_in_the_right_order() {
        let a = (yaw_rotation(0.4) * pitch_rotation(0.1)).transpose();
        let b = (pitch_rotation(-0.05) * yaw_rotation(0.6)).transpose();
        let chain = chain_rotations(&[Some(a), Some(b)]);

        let d2 = Vector3::new(0.1, -0.2, 1.0);
        let d0 = a.transpose() * (b.transpose() * d2);
        assert!((chain[2].unwrap() * d2 - d0).amax() < 1e-12);
    }

    #[test]
    fn chain_rotations_breaks_after_a_missing_pair() {
        let step = Some(yaw_rotation(0.3));
        let chain = chain_rotations(&[step, None, step]);
        assert!(chain[0].is_some() && chain[1].is_some());
        assert!(chain[2].is_none() && chain[3].is_none());
    }

    #[test]
    fn chain_of_no_pairs_is_just_the_reference() {
        let chain = chain_rotations(&[]);
        assert_eq!(chain.len(), 1);
        assert_close(chain[0].as_ref().unwrap(), &Matrix3::identity(), 1e-12);
    }

    #[test]
    fn wrap_angle_maps_into_half_open_range() {
        assert!((wrap_angle(3.0 * PI / 2.0) + PI / 2.0).abs() < 1e-12);
        assert!((wrap_angle(-3.0 * PI / 2.0) - PI / 2.0).abs() < 1e-12);
        assert!((wrap_angle(PI) + PI).abs() < 1e-12);
        assert!((wrap_angle(0.25) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn ray_to_cylinder_maps_known_directions() {
        let (t, h) = ray_to_cylinder(&Vector3::new(0.0, 0.0, 1.0)).unwrap();
        assert!(t.abs() < 1e-12 && h.abs() < 1e-12);

        let (t, h) = ray_to_cylinder(&Vector3::new(1.0, 0.0, 1.0)).unwrap();
        assert!((t - PI / 4.0).abs() < 1e-12 && h.abs() < 1e-12);

        let (t, h) = ray_to_cylinder(&Vector3::new(0.0, 1.0, 1.0)).unwrap();
        assert!(t.abs() < 1e-12 && (h - 1.0).abs() < 1e-12);

        // Scale-invariant: rays are directions.
        let (t, h) = ray_to_cylinder(&Vector3::new(0.0, 3.0, 3.0)).unwrap();
        assert!(t.abs() < 1e-12 && (h - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ray_to_cylinder_rejects_rays_along_the_axis() {
        assert!(ray_to_cylinder(&Vector3::new(0.0, 1.0, 0.0)).is_none());
        assert!(ray_to_cylinder(&Vector3::new(0.0, -1.0, 0.0)).is_none());
    }

    #[test]
    fn reference_outline_is_centred_and_symmetric() {
        let (w, h, f) = (400, 300, 300.0);
        let k = intrinsics(f, w, h);
        let (outline, centre) =
            cylinder_outline(&Matrix3::identity(), &k, w, h, f, 0.0).unwrap();

        assert_eq!(outline.len(), 4 * OUTLINE_SAMPLES_PER_EDGE);
        assert!(centre.abs() < 1e-12);

        let top_left = outline[0];
        let bottom_right = outline[2 * OUTLINE_SAMPLES_PER_EDGE];
        assert!(top_left[0] < 0.0 && top_left[1] < 0.0);
        assert!((top_left[0] + bottom_right[0]).abs() < 1e-9);
        assert!((top_left[1] + bottom_right[1]).abs() < 1e-9);

        // Left edge sits at yaw atan(200 / 300), scaled by the radius.
        let expected_left = f * (-200.0f64 / f).atan();
        assert!((top_left[0] - expected_left).abs() < 1e-9);
    }

    #[test]
    fn yawed_image_shifts_by_radius_times_angle() {
        let (w, h, f) = (400, 300, 300.0);
        let k = intrinsics(f, w, h);
        let (base, _) = cylinder_outline(&Matrix3::identity(), &k, w, h, f, 0.0).unwrap();
        let (moved, centre) = cylinder_outline(&yaw_rotation(0.5), &k, w, h, f, 0.0).unwrap();

        assert!((centre - 0.5).abs() < 1e-12);
        for (a, b) in base.iter().zip(&moved) {
            assert!((b[0] - a[0] - f * 0.5).abs() < 1e-9);
            assert!((b[1] - a[1]).abs() < 1e-9);
        }
    }

    #[test]
    fn outline_unwraps_past_half_a_turn() {
        let (w, h, f) = (400, 300, 300.0);
        let k = intrinsics(f, w, h);
        let yaw = 200f64.to_radians();
        let (outline, centre) =
            cylinder_outline(&yaw_rotation(yaw), &k, w, h, f, 170f64.to_radians()).unwrap();

        assert!((centre - yaw).abs() < 1e-9, "centre {centre} should stay near 200°");
        let xs: Vec<f64> = outline.iter().map(|p| p[0]).collect();
        let span = xs.iter().cloned().fold(f64::MIN, f64::max) - xs.iter().cloned().fold(f64::MAX, f64::min);
        assert!(span < f * PI / 2.0, "outline must not split across ±π (span {span})");
    }

    #[test]
    fn outline_fails_when_camera_looks_along_the_axis() {
        let (w, h, f) = (400, 300, 300.0);
        let k = intrinsics(f, w, h);
        // A 90° pitch puts the centre ray on the cylinder axis, which has no yaw.
        assert!(cylinder_outline(&pitch_rotation(-PI / 2.0), &k, w, h, f, 0.0).is_none());
    }
}
