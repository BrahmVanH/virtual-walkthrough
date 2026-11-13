use arrsac::Arrsac;
use nalgebra::{DMatrix, Matrix3, Point2, Vector3};
use rand_core::SeedableRng;
use rand_pcg::Pcg64;
use sample_consensus::{Consensus, Estimator, Model};

#[derive(Clone, Copy, Debug)]
pub struct Match {
    pub a: Point2<f64>,
    pub b: Point2<f64>,
}

#[derive(Clone, Debug)]
pub struct Homography(pub Matrix3<f64>);

// Residual = reprojection error in pixels (how far H*a lands from b)
impl Model<Match> for Homography {
    fn residual(&self, m: &Match) -> f64 {
        let p = self.0 * Vector3::new(m.a.x, m.a.y, 1.0);
        if p.z.abs() < 1e-12 {
            return f64::MAX;
        }
        let (x, y) = (p.x / p.z, p.y / p.z);
        ((x - m.b.x).powi(2) + (y - m.b.y).powi(2)).sqrt()
    }
}

pub struct HomographyEstimator;

impl Estimator<Match> for HomographyEstimator {
    type Model = Homography;
    type ModelIter = Option<Homography>;
    const MIN_SAMPLES: usize = 4;

    fn estimate<I>(&self, data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Match> + Clone,
    {
        let pts: Vec<Match> = data.collect();
        dlt(&pts).map(Homography)
    }
}

/// Hartley normalization: center at origin, mean distance = sqrt(2).
/// Without this the DLT is numerically awful on pixel coordinates.
fn normalize(pts: &[Point2<f64>]) -> Option<(Vec<Point2<f64>>, Matrix3<f64>)> {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.x).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.y).sum::<f64>() / n;
    let mean_dist = pts
        .iter()
        .map(|p| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    if mean_dist < 1e-9 {
        return None; // all points identical, degenerate
    }
    let s = 2f64.sqrt() / mean_dist;
    let t = Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0);
    let out = pts
        .iter()
        .map(|p| Point2::new((p.x - cx) * s, (p.y - cy) * s))
        .collect();
    Some((out, t))
}

/// Direct Linear Transform. Works for 4 points (minimal) or any larger set (least squares).
fn dlt(m: &[Match]) -> Option<Matrix3<f64>> {
    if m.len() < 4 {
        return None;
    }
    let a_pts: Vec<_> = m.iter().map(|x| x.a).collect();
    let b_pts: Vec<_> = m.iter().map(|x| x.b).collect();
    let (na, ta) = normalize(&a_pts)?;
    let (nb, tb) = normalize(&b_pts)?;

    // Each correspondence contributes 2 rows. Pad to at least 9 rows so the
    // SVD returns a full 9x9 V^T (see note below).
    let rows = (2 * m.len()).max(9);
    let mut a = DMatrix::<f64>::zeros(rows, 9);
    for (i, (p, q)) in na.iter().zip(nb.iter()).enumerate() {
        let (x, y, u, v) = (p.x, p.y, q.x, q.y);
        let r0 = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u];
        let r1 = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v];
        for c in 0..9 {
            a[(2 * i, c)] = r0[c];
            a[(2 * i + 1, c)] = r1[c];
        }
    }

    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    // Singular values are sorted descending, so the last row of V^T is the
    // null-space vector, which is our flattened H.
    let h: Vec<f64> = v_t.row(8).iter().cloned().collect();
    let h_norm = Matrix3::from_row_slice(&h);

    // Undo normalization: H = Tb^-1 * Hn * Ta
    let mut hm = tb.try_inverse()? * h_norm * ta;
    if hm[(2, 2)].abs() > 1e-12 {
        hm /= hm[(2, 2)];
    }
    Some(hm)
}

/// Returns the refined homography (A -> B) and the indices of inlier matches.
pub fn estimate_homography(
    pairs: &[((f32, f32), (f32, f32))],
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    let data: Vec<Match> = pairs
        .iter()
        .map(|&((ax, ay), (bx, by))| Match {
            a: Point2::new(ax as f64, ay as f64),
            b: Point2::new(bx as f64, by as f64),
        })
        .collect();

    // Fixed seed: deterministic runs (great for debugging) and no getrandom/WASM headaches
    let mut arrsac = Arrsac::new(3.0, Pcg64::seed_from_u64(0xBEEF)); // 3px inlier threshold

    let (model, inliers) = arrsac.model_inliers(&HomographyEstimator, data.iter().copied())?;

    if inliers.len() < crate::MIN_INLIERS {
        return None; // not enough support, so these images probably don't overlap
    }

    // Final least-squares refit on all inliers instead of the 4-point minimal sample
    let inlier_data: Vec<Match> = inliers.iter().map(|&i| data[i]).collect();
    let refined = dlt(&inlier_data).unwrap_or(model.0);

    Some((refined, inliers))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn true_h() -> Matrix3<f64> {
        Matrix3::new(1.0, 0.02, 30.0, -0.01, 1.0, 5.0, 1e-5, 0.0, 1.0)
    }

    fn project(h: &Matrix3<f64>, x: f64, y: f64) -> (f64, f64) {
        let p = h * Vector3::new(x, y, 1.0);
        (p.x / p.z, p.y / p.z)
    }

    /// `n * n` grid over a 1000x1000 px image, mapped through `true_h`.
    fn grid_pairs(n: usize) -> Vec<((f32, f32), (f32, f32))> {
        let h = true_h();
        let step = 1000.0 / n as f64;
        (0..n * n)
            .map(|k| {
                let (x, y) = ((k % n) as f64 * step, (k / n) as f64 * step);
                let (u, v) = project(&h, x, y);
                ((x as f32, y as f32), (u as f32, v as f32))
            })
            .collect()
    }

    fn max_reprojection_gap(estimated: &Matrix3<f64>) -> f64 {
        let h = true_h();
        let mut worst: f64 = 0.0;
        for x in (0..=1000).step_by(100) {
            for y in (0..=1000).step_by(100) {
                let (ex, ey) = project(estimated, x as f64, y as f64);
                let (tx, ty) = project(&h, x as f64, y as f64);
                worst = worst.max(((ex - tx).powi(2) + (ey - ty).powi(2)).sqrt());
            }
        }
        worst
    }

    #[test]
    fn residual_is_zero_for_exact_correspondence() {
        let h = Homography(true_h());
        let (u, v) = project(&true_h(), 200.0, 300.0);
        let m = Match {
            a: Point2::new(200.0, 300.0),
            b: Point2::new(u, v),
        };
        assert!(h.residual(&m) < 1e-9);
    }

    #[test]
    fn dlt_recovers_known_homography_from_minimal_sample() {
        let h = true_h();
        let m: Vec<Match> = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0)]
            .iter()
            .map(|&(x, y)| {
                let (u, v) = project(&h, x, y);
                Match {
                    a: Point2::new(x, y),
                    b: Point2::new(u, v),
                }
            })
            .collect();
        let est = dlt(&m).unwrap();
        assert!(max_reprojection_gap(&est) < 1e-4);
    }

    #[test]
    fn dlt_rejects_degenerate_input() {
        let p = Match {
            a: Point2::new(5.0, 5.0),
            b: Point2::new(5.0, 5.0),
        };
        assert!(dlt(&[p; 3]).is_none());
        assert!(dlt(&[p; 4]).is_none());
    }

    #[test]
    fn estimate_recovers_homography_with_all_inliers() {
        let pairs = grid_pairs(10);
        let (h, inliers) = estimate_homography(&pairs).unwrap();
        assert_eq!(inliers.len(), pairs.len());
        // Inputs are f32, so allow sub-pixel slack.
        assert!(max_reprojection_gap(&h) < 0.01);
    }

    #[test]
    fn estimate_excludes_gross_outliers() {
        let mut pairs = grid_pairs(10);
        let good = pairs.len();
        for k in 0..30 {
            let (a, (u, v)) = pairs[k * 3];
            let offset = 50.0 + k as f32;
            pairs.push((a, (u + offset, v - offset)));
        }

        let (h, inliers) = estimate_homography(&pairs).unwrap();
        assert_eq!(inliers.len(), good);
        assert!(inliers.iter().all(|&i| i < good));
        assert!(max_reprojection_gap(&h) < 0.01);
    }

    #[test]
    fn estimate_rejects_pairs_with_too_few_inliers() {
        // 4x4 grid = 16 perfect correspondences, below the 20-inlier floor.
        assert!(estimate_homography(&grid_pairs(4)).is_none());
    }
}
