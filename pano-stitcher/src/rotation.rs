//! Pure-rotation model between two cameras, fitted with ARRSAC.
//!
//! A homography can also explain a flat surface (usually the floor) seen from
//! two camera positions. A rotation can't, so fitting one directly rejects
//! matches that only line up because of such a plane, instead of turning them
//! into a tilted, skewed placement.

use arrsac::Arrsac;
use nalgebra::{Matrix3, Vector3};
use rand_core::SeedableRng;
use rand_pcg::Pcg64;
use sample_consensus::{Consensus, Estimator, Model};

/// Same pixel threshold as the homography fit, so their inlier counts compare.
pub const INLIER_THRESHOLD_PX: f64 = 3.0;

#[derive(Clone, Copy, Debug)]
struct BearingMatch {
    /// Unit ray in camera a.
    a: Vector3<f64>,
    /// Unit ray in camera b.
    b: Vector3<f64>,
    /// Matched pixel in image b.
    b_px: (f64, f64),
}

#[derive(Clone, Debug)]
struct RotationModel {
    /// Camera-a rays into camera-b coords.
    r: Matrix3<f64>,
    k_b: Matrix3<f64>,
}

impl Model<BearingMatch> for RotationModel {
    /// Pixel distance in image b between the rotated ray from a and the match.
    fn residual(&self, m: &BearingMatch) -> f64 {
        let p = self.k_b * (self.r * m.a);
        if p.z <= 1e-9 {
            return f64::MAX;
        }
        (p.x / p.z - m.b_px.0).hypot(p.y / p.z - m.b_px.1)
    }
}

struct RotationEstimator {
    k_b: Matrix3<f64>,
}

impl Estimator<BearingMatch> for RotationEstimator {
    type Model = RotationModel;
    type ModelIter = Option<RotationModel>;
    const MIN_SAMPLES: usize = 2;

    fn estimate<I>(&self, data: I) -> Self::ModelIter
    where
        I: Iterator<Item = BearingMatch> + Clone,
    {
        kabsch(data.map(|m| (m.a, m.b))).map(|r| RotationModel { r, k_b: self.k_b })
    }
}

/// Rotation `R` minimising Σ |b - R a|² over `(a, b)` unit-vector pairs (Kabsch).
/// `None` when the vectors span less than two directions, which leaves the
/// rotation about them undetermined.
pub fn kabsch(pairs: impl Iterator<Item = (Vector3<f64>, Vector3<f64>)>) -> Option<Matrix3<f64>> {
    let h: Matrix3<f64> = pairs.map(|(a, b)| b * a.transpose()).sum();
    let svd = h.svd(true, true);
    // svd() sorts singular values in descending order; two independent
    // directions give rank 2, which already pins down a rotation.
    let s = svd.singular_values;
    if s[1] <= 1e-9 * s[0] || s[0] == 0.0 {
        return None;
    }
    let (u, v_t) = (svd.u?, svd.v_t?);
    // Flip the least-constrained axis if U V^T would be a reflection.
    let d = (u * v_t).determinant().signum();
    Some(u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * v_t)
}

/// Fits the rotation taking camera-a rays into camera-b coords from matched
/// pixels `(a, b)` in images with intrinsics `k_a`, `k_b`. Returns it, refit on
/// all inliers, with the indices of matches within `INLIER_THRESHOLD_PX` of the
/// refit. Applies no minimum inlier count.
pub fn estimate_rotation(
    pairs: &[((f32, f32), (f32, f32))],
    k_a: &Matrix3<f64>,
    k_b: &Matrix3<f64>,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    let (ka_inv, kb_inv) = (k_a.try_inverse()?, k_b.try_inverse()?);
    let ray = |k_inv: &Matrix3<f64>, (x, y): (f32, f32)| {
        (k_inv * Vector3::new(x as f64, y as f64, 1.0)).normalize()
    };
    let data: Vec<BearingMatch> = pairs
        .iter()
        .map(|&(a, b)| BearingMatch {
            a: ray(&ka_inv, a),
            b: ray(&kb_inv, b),
            b_px: (b.0 as f64, b.1 as f64),
        })
        .collect();

    // Fixed seed, as in the homography fit: reproducible and no getrandom on WASM.
    let mut arrsac = Arrsac::new(INLIER_THRESHOLD_PX, Pcg64::seed_from_u64(0xBEEF));
    let estimator = RotationEstimator { k_b: *k_b };
    let (model, sampled) = arrsac.model_inliers(&estimator, data.iter().copied())?;

    let refined = kabsch(sampled.iter().map(|&i| (data[i].a, data[i].b)))
        .map(|r| RotationModel { r, k_b: *k_b })
        .unwrap_or(model);
    let inliers = (0..data.len())
        .filter(|&i| refined.residual(&data[i]) <= INLIER_THRESHOLD_PX)
        .collect();
    Some((refined.r, inliers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cylinder;

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

    /// Grid of pixels in a 400x300 image a, matched to where camera b (rays a -> b
    /// via `r_ab`) sees the same direction, keeping those visible in b.
    fn rotated_matches(r_ab: &Matrix3<f64>) -> Vec<((f32, f32), (f32, f32))> {
        let k = cylinder::intrinsics(300.0, 400, 300);
        let k_inv = k.try_inverse().unwrap();
        let mut out = Vec::new();
        for gx in 0..20 {
            for gy in 0..15 {
                let (x, y) = (10.0 + gx as f64 * 20.0, 10.0 + gy as f64 * 20.0);
                let p = k * r_ab * k_inv * Vector3::new(x, y, 1.0);
                let (u, v) = (p.x / p.z, p.y / p.z);
                if p.z > 0.0 && (0.0..400.0).contains(&u) && (0.0..300.0).contains(&v) {
                    out.push(((x as f32, y as f32), (u as f32, v as f32)));
                }
            }
        }
        out
    }

    #[test]
    fn kabsch_recovers_an_exact_rotation() {
        let r = yaw_rotation(0.4) * pitch_rotation(-0.1);
        let rays = [Vector3::new(0.1, 0.2, 1.0), Vector3::new(-0.3, 0.0, 1.0), Vector3::new(0.0, -0.4, 1.0)]
            .map(|v| v.normalize());
        assert_close(&kabsch(rays.iter().map(|&a| (a, r * a))).unwrap(), &r, 1e-12);
        // Two independent directions are enough.
        assert_close(&kabsch(rays[..2].iter().map(|&a| (a, r * a))).unwrap(), &r, 1e-12);
    }

    #[test]
    fn kabsch_rejects_degenerate_input() {
        let a = Vector3::new(0.1, 0.2, 1.0).normalize();
        assert!(kabsch([(a, a), (a, a)].into_iter()).is_none());
        assert!(kabsch(std::iter::empty()).is_none());
    }

    #[test]
    fn estimate_rotation_recovers_camera_motion() {
        let r_ab = (yaw_rotation(25f64.to_radians()) * pitch_rotation(0.05)).transpose();
        let pairs = rotated_matches(&r_ab);
        let k = cylinder::intrinsics(300.0, 400, 300);

        let (r, inliers) = estimate_rotation(&pairs, &k, &k).unwrap();
        assert_close(&r, &r_ab, 1e-4);
        assert_eq!(inliers.len(), pairs.len());
    }

    #[test]
    fn estimate_rotation_excludes_outliers() {
        let r_ab = yaw_rotation(20f64.to_radians()).transpose();
        let mut pairs = rotated_matches(&r_ab);
        let good = pairs.len();
        for i in 0..30 {
            let (a, (u, v)) = pairs[i * 3];
            pairs.push((a, (u + 40.0 + i as f32, v - 25.0)));
        }
        let k = cylinder::intrinsics(300.0, 400, 300);

        let (r, inliers) = estimate_rotation(&pairs, &k, &k).unwrap();
        assert_close(&r, &r_ab, 1e-4);
        assert_eq!(inliers.len(), good);
        assert!(inliers.iter().all(|&i| i < good));
    }

    #[test]
    fn estimate_rotation_needs_at_least_two_matches() {
        let k = cylinder::intrinsics(300.0, 400, 300);
        assert!(estimate_rotation(&[((10.0, 10.0), (12.0, 10.0))], &k, &k).is_none());
    }
}
