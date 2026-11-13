//! Output canvas over the unrolled cylinder.
//!
//! Cylinder coords are the cylinder's own output pixels: x = radius * yaw,
//! y = radius * height. Canvas pixel `(u, v)` has its centre at cylinder coords
//! `(min_x + (u + 0.5) / scale, min_y + (v + 0.5) / scale)`, so the bounding box
//! edges land on the canvas's outer pixel edges at -0.5 and `size - 0.5`.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    /// Top-left corner of the bounding box, in cylinder coords.
    pub min_x: f64,
    pub min_y: f64,
    /// Canvas pixels per cylinder pixel, at most 1 (never upscales).
    pub scale: f64,
}

impl Canvas {
    /// Smallest canvas covering every point, shrunk so neither side exceeds
    /// `max_dim`. `None` when there are no points or any point is not finite.
    pub fn bounding(points: impl IntoIterator<Item = [f64; 2]>, max_dim: u32) -> Option<Self> {
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        let mut any = false;
        for p in points {
            if !(p[0].is_finite() && p[1].is_finite()) {
                return None;
            }
            any = true;
            for axis in 0..2 {
                min[axis] = min[axis].min(p[axis]);
                max[axis] = max[axis].max(p[axis]);
            }
        }
        if !any {
            return None;
        }

        let (w, h) = (max[0] - min[0], max[1] - min[1]);
        // A zero extent divides to infinity, which min() turns back into 1.
        let scale = (max_dim as f64 / w.max(h)).min(1.0);
        // clamp: rounding can push `extent * scale` a hair past max_dim.
        let size = |extent: f64| ((extent * scale).ceil() as u32).clamp(1, max_dim.max(1));
        Some(Canvas {
            width: size(w),
            height: size(h),
            min_x: min[0],
            min_y: min[1],
            scale,
        })
    }

    /// Cylinder coords to canvas pixel coords (pixel centres at integers).
    pub fn to_canvas(&self, [x, y]: [f64; 2]) -> [f64; 2] {
        [
            (x - self.min_x) * self.scale - 0.5,
            (y - self.min_y) * self.scale - 0.5,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 2], b: [f64; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9
    }

    #[test]
    fn bounding_box_maps_to_outer_pixel_edges() {
        let c = Canvas::bounding([[-100.0, -50.0], [300.0, 150.0], [0.0, 0.0]], 4096).unwrap();
        assert_eq!((c.width, c.height), (400, 200));
        assert!((c.scale - 1.0).abs() < 1e-12);
        assert!(close(c.to_canvas([-100.0, -50.0]), [-0.5, -0.5]));
        assert!(close(c.to_canvas([300.0, 150.0]), [399.5, 199.5]));
    }

    #[test]
    fn fractional_extent_rounds_up() {
        let c = Canvas::bounding([[0.0, 0.0], [10.2, 3.7]], 4096).unwrap();
        assert_eq!((c.width, c.height), (11, 4));
    }

    #[test]
    fn long_side_is_capped_and_aspect_kept() {
        let c = Canvas::bounding([[0.0, 0.0], [8000.0, 2000.0]], 4000).unwrap();
        assert!((c.scale - 0.5).abs() < 1e-12);
        assert_eq!((c.width, c.height), (4000, 1000));
        assert!(close(c.to_canvas([8000.0, 2000.0]), [3999.5, 999.5]));

        let tall = Canvas::bounding([[0.0, 0.0], [1000.0, 5000.0]], 2500).unwrap();
        assert_eq!((tall.width, tall.height), (500, 2500));
    }

    #[test]
    fn never_upscales() {
        let c = Canvas::bounding([[0.0, 0.0], [100.0, 50.0]], 4096).unwrap();
        assert!((c.scale - 1.0).abs() < 1e-12);
        assert_eq!((c.width, c.height), (100, 50));
    }

    #[test]
    fn degenerate_box_still_gets_one_pixel() {
        let c = Canvas::bounding([[5.0, 5.0]], 4096).unwrap();
        assert_eq!((c.width, c.height), (1, 1));
        assert!((c.scale - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_or_non_finite_input_has_no_canvas() {
        assert!(Canvas::bounding([], 4096).is_none());
        assert!(Canvas::bounding([[0.0, 0.0], [f64::NAN, 1.0]], 4096).is_none());
        assert!(Canvas::bounding([[0.0, f64::INFINITY]], 4096).is_none());
    }
}
