//! Warps placed images onto the output canvas and feather-blends overlaps.
//!
//! Rendering runs backwards: each canvas pixel becomes a ray on the cylinder,
//! and every image that ray hits contributes a bilinear sample, weighted to fade
//! out toward that image's border.

use crate::canvas::Canvas;
use image::RgbImage;
use nalgebra::{Matrix3, Vector3};

/// Outline samples sit 1/16 of an edge apart, and the curved edge between two
/// samples bulges by under a pixel at phone focals, so this padding keeps the
/// skip box covering the whole image.
const BOUNDS_PAD_PX: f64 = 2.0;

/// Canvas pixel range `[u0, u1) x [v0, v1)` an image can touch.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bounds {
    u0: usize,
    u1: usize,
    v0: usize,
    v1: usize,
}

impl Bounds {
    fn contains(&self, u: usize, v: usize) -> bool {
        (self.u0..self.u1).contains(&u) && (self.v0..self.v1).contains(&v)
    }
}

/// One placed image, ready to be sampled by canvas rays.
pub struct Source<'a> {
    pixels: &'a RgbImage,
    /// Ray in image 0's camera coords to homogeneous pixel coords here: `K R^T`.
    ref_to_pixel: Matrix3<f64>,
    /// Pure speed-up: pixels outside are skipped without projecting.
    bounds: Bounds,
}

impl<'a> Source<'a> {
    /// `rotation` maps this camera's rays into image 0's; `outline` is in canvas pixels.
    pub fn new(
        pixels: &'a RgbImage,
        rotation: &Matrix3<f64>,
        k: &Matrix3<f64>,
        outline: &[[f64; 2]],
        canvas: &Canvas,
    ) -> Self {
        Source {
            pixels,
            ref_to_pixel: k * rotation.transpose(),
            bounds: pixel_bounds(outline, canvas),
        }
    }

    /// Pixel coords (centres at integers) where `ray` hits this image, if it does.
    fn project(&self, ray: &Vector3<f64>) -> Option<(f64, f64)> {
        let p = self.ref_to_pixel * ray;
        // Rays behind the camera also cross the image plane, mirrored.
        if p.z <= 1e-9 {
            return None;
        }
        let (x, y) = (p.x / p.z, p.y / p.z);
        let (w, h) = (self.pixels.width() as f64, self.pixels.height() as f64);
        // Outer pixel edges sit at -0.5 and size - 0.5.
        if x < -0.5 || y < -0.5 || x > w - 0.5 || y > h - 0.5 {
            return None;
        }
        Some((x, y))
    }
}

fn pixel_bounds(outline: &[[f64; 2]], canvas: &Canvas) -> Bounds {
    if outline.is_empty() {
        return Bounds { u0: 0, u1: 0, v0: 0, v1: 0 };
    }
    let fold = |axis: usize, init: f64, f: fn(f64, f64) -> f64| {
        outline.iter().map(|p| p[axis]).fold(init, f)
    };
    let lo = |v: f64| (v - BOUNDS_PAD_PX).floor().max(0.0) as usize;
    // +1: the pixel whose centre is at ceil(max) is itself inside.
    // Float-to-int casts saturate, so boxes entirely off-canvas collapse to empty ranges.
    let hi = |v: f64, size: u32| ((v + BOUNDS_PAD_PX).ceil() + 1.0).min(size as f64) as usize;
    Bounds {
        u0: lo(fold(0, f64::INFINITY, f64::min)),
        u1: hi(fold(0, f64::NEG_INFINITY, f64::max), canvas.width),
        v0: lo(fold(1, f64::INFINITY, f64::min)),
        v1: hi(fold(1, f64::NEG_INFINITY, f64::max), canvas.height),
    }
}

/// RGBA8 canvas, row-major from the top-left. Pixels no image covers are
/// transparent black; covered pixels are opaque. Overlaps are a weighted mean
/// in sRGB (exposure compensation is Phase 2).
pub fn render(canvas: &Canvas, radius: f64, sources: &[Source]) -> Vec<u8> {
    let (w, h) = (canvas.width as usize, canvas.height as usize);
    let mut out = vec![0u8; w * h * 4];

    // Inverse of Canvas::to_canvas. Yaw depends only on the column, height only on the row.
    let columns: Vec<(f64, f64)> = (0..w)
        .map(|u| ((canvas.min_x + (u as f64 + 0.5) / canvas.scale) / radius).sin_cos())
        .collect();

    for v in 0..h {
        let height = (canvas.min_y + (v as f64 + 0.5) / canvas.scale) / radius;
        for (u, &(sin, cos)) in columns.iter().enumerate() {
            let ray = Vector3::new(sin, height, cos);
            let mut acc = [0.0f64; 3];
            let mut weight_sum = 0.0;
            for s in sources {
                if !s.bounds.contains(u, v) {
                    continue;
                }
                let Some((x, y)) = s.project(&ray) else {
                    continue;
                };
                let weight = feather(x, y, s.pixels.width(), s.pixels.height());
                let c = bilinear(s.pixels, x, y);
                for i in 0..3 {
                    acc[i] += weight * c[i];
                }
                weight_sum += weight;
            }
            if weight_sum > 0.0 {
                let o = (v * w + u) * 4;
                for i in 0..3 {
                    out[o + i] = (acc[i] / weight_sum).round().clamp(0.0, 255.0) as u8;
                }
                out[o + 3] = 255;
            }
        }
    }
    out
}

/// Linear ramp: 1 at the image centre, 0 at its outer pixel edges, so each image
/// fades out toward its border where it overlaps a neighbour. Floored above 0 so
/// an edge pixel nothing else covers still takes this image's colour.
fn feather(x: f64, y: f64, width: u32, height: u32) -> f64 {
    let ramp = |t: f64, size: u32| 1.0 - ((t + 0.5) * 2.0 / size as f64 - 1.0).abs();
    (ramp(x, width) * ramp(y, height)).max(1e-6)
}

/// RGB at sub-pixel `(x, y)` (pixel centres at integers), clamped to the border pixels.
fn bilinear(img: &RgbImage, x: f64, y: f64) -> [f64; 3] {
    let (max_x, max_y) = (img.width() - 1, img.height() - 1);
    let x = x.clamp(0.0, max_x as f64);
    let y = y.clamp(0.0, max_y as f64);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(max_x), (y0 + 1).min(max_y));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);

    let px = |px: u32, py: u32| img.get_pixel(px, py).0;
    let (a, b, c, d) = (px(x0, y0), px(x1, y0), px(x0, y1), px(x1, y1));
    std::array::from_fn(|i| {
        let top = a[i] as f64 * (1.0 - fx) + b[i] as f64 * fx;
        let bottom = c[i] as f64 * (1.0 - fx) + d[i] as f64 * fx;
        top * (1.0 - fy) + bottom * fy
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cylinder;
    use image::Rgb;

    const W: u32 = 200;
    const H: u32 = 150;
    const F: f64 = 150.0;

    fn yaw_rotation(yaw: f64) -> Matrix3<f64> {
        let (s, c) = yaw.sin_cos();
        Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
    }

    fn roll_rotation(roll: f64) -> Matrix3<f64> {
        let (s, c) = roll.sin_cos();
        Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
    }

    fn solid(rgb: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(W, H, Rgb(rgb))
    }

    /// Red = x, green = y, so a sampled colour reads back as a source position.
    fn gradient() -> RgbImage {
        RgbImage::from_fn(W, H, |x, y| Rgb([x as u8, y as u8, 0]))
    }

    /// Canvas plus per-image outlines in canvas pixels, for images at `rotations`.
    fn layout(rotations: &[Matrix3<f64>]) -> (Canvas, Vec<Vec<[f64; 2]>>) {
        let k = cylinder::intrinsics(F, W, H);
        let mut near = 0.0;
        let outlines: Vec<_> = rotations
            .iter()
            .map(|r| {
                let (outline, centre) = cylinder::cylinder_outline(r, &k, W, H, F, near).unwrap();
                near = centre;
                outline
            })
            .collect();
        let canvas = Canvas::bounding(outlines.iter().flatten().copied(), 4096).unwrap();
        let outlines = outlines
            .into_iter()
            .map(|o| o.into_iter().map(|p| canvas.to_canvas(p)).collect())
            .collect();
        (canvas, outlines)
    }

    fn sources<'a>(
        images: &'a [RgbImage],
        rotations: &[Matrix3<f64>],
        outlines: &[Vec<[f64; 2]>],
        canvas: &Canvas,
    ) -> Vec<Source<'a>> {
        let k = cylinder::intrinsics(F, W, H);
        images
            .iter()
            .zip(rotations)
            .zip(outlines)
            .map(|((img, r), o)| Source::new(img, r, &k, o, canvas))
            .collect()
    }

    /// RGBA at the canvas pixel nearest cylinder coords `p`.
    fn pixel_at(out: &[u8], canvas: &Canvas, p: [f64; 2]) -> [u8; 4] {
        let [u, v] = canvas.to_canvas(p).map(|c| c.round() as usize);
        let o = (v * canvas.width as usize + u) * 4;
        [out[o], out[o + 1], out[o + 2], out[o + 3]]
    }

    #[test]
    fn reference_image_lands_upright_and_unflipped() {
        let rotations = [Matrix3::identity()];
        let (canvas, outlines) = layout(&rotations);
        let images = [gradient()];
        let out = render(&canvas, F, &sources(&images, &rotations, &outlines, &canvas));

        // Image centre (99.5, 74.5) sits at cylinder (0, 0).
        let c = pixel_at(&out, &canvas, [0.0, 0.0]);
        assert!((c[0] as f64 - 99.5).abs() <= 1.0 && (c[1] as f64 - 74.5).abs() <= 1.0, "{c:?}");
        assert_eq!(c[3], 255);

        // 20 px right on the cylinder is f * tan(20 / f) ≈ 20.1 px right in the image.
        let right = pixel_at(&out, &canvas, [20.0, 0.0]);
        assert!((right[0] as f64 - 119.6).abs() <= 1.5, "{right:?}");
        // 20 px down at yaw 0 is exactly 20 px down in the image.
        let down = pixel_at(&out, &canvas, [0.0, 20.0]);
        assert!((down[1] as f64 - 94.5).abs() <= 1.0, "{down:?}");
    }

    #[test]
    fn pixels_outside_every_image_are_transparent() {
        let rotations = [Matrix3::identity()];
        let (canvas, outlines) = layout(&rotations);
        let images = [solid([200, 100, 50])];
        let out = render(&canvas, F, &sources(&images, &rotations, &outlines, &canvas));

        assert_eq!(out.len(), canvas.width as usize * canvas.height as usize * 4);
        // The top edge bows upward on the cylinder, so the box's top-left corner is empty.
        assert_eq!(&out[0..4], &[0, 0, 0, 0]);
        assert_eq!(pixel_at(&out, &canvas, [0.0, 0.0]), [200, 100, 50, 255]);
    }

    #[test]
    fn overlap_feathers_between_neighbours() {
        let yaw = 30f64.to_radians();
        let rotations = [Matrix3::identity(), yaw_rotation(yaw)];
        let (canvas, outlines) = layout(&rotations);
        let images = [solid([255, 0, 0]), solid([0, 0, 255])];
        let out = render(&canvas, F, &sources(&images, &rotations, &outlines, &canvas));

        // Each image spans ±33.7° of yaw around its centre at f = 150, w = 200.
        assert_eq!(pixel_at(&out, &canvas, [F * (-25f64).to_radians(), 0.0]), [255, 0, 0, 255]);
        assert_eq!(pixel_at(&out, &canvas, [F * 50f64.to_radians(), 0.0]), [0, 0, 255, 255]);

        // Halfway between the centres both images are equally far from their borders.
        let mid = pixel_at(&out, &canvas, [F * yaw / 2.0, 0.0]);
        assert!((mid[0] as i32 - 128).abs() <= 4 && (mid[2] as i32 - 128).abs() <= 4, "{mid:?}");
        assert_eq!(mid[1], 0);

        // Closer to image 0's centre, image 0 dominates.
        let left = pixel_at(&out, &canvas, [F * 5f64.to_radians(), 0.0]);
        assert!(left[0] > left[2], "{left:?}");
    }

    #[test]
    fn skip_box_never_drops_pixels() {
        // Roll makes the edges diagonal and curved, the worst case for the box.
        let rotations = [
            Matrix3::identity(),
            yaw_rotation(35f64.to_radians()) * roll_rotation(25f64.to_radians()),
        ];
        let (canvas, outlines) = layout(&rotations);
        let images = [gradient(), gradient()];

        let boxed = sources(&images, &rotations, &outlines, &canvas);
        let mut unboxed = sources(&images, &rotations, &outlines, &canvas);
        for s in &mut unboxed {
            s.bounds = Bounds { u0: 0, u1: canvas.width as usize, v0: 0, v1: canvas.height as usize };
        }
        assert!(render(&canvas, F, &boxed) == render(&canvas, F, &unboxed));
    }

    #[test]
    fn pixel_bounds_pads_and_clamps_to_the_canvas() {
        let canvas = Canvas { width: 100, height: 50, min_x: 0.0, min_y: 0.0, scale: 1.0 };
        let b = pixel_bounds(&[[10.2, 5.0], [40.7, 20.5]], &canvas);
        assert_eq!(b, Bounds { u0: 8, u1: 44, v0: 3, v1: 24 });

        let edge = pixel_bounds(&[[-0.5, -0.5], [99.5, 49.5]], &canvas);
        assert_eq!(edge, Bounds { u0: 0, u1: 100, v0: 0, v1: 50 });

        assert!(!pixel_bounds(&[], &canvas).contains(0, 0));
    }

    #[test]
    fn bilinear_interpolates_and_clamps() {
        let img = RgbImage::from_fn(2, 2, |x, y| Rgb([(x * 100) as u8, (y * 200) as u8, 7]));
        let mid = bilinear(&img, 0.5, 0.5);
        assert!((mid[0] - 50.0).abs() < 1e-9 && (mid[1] - 100.0).abs() < 1e-9);
        assert!((mid[2] - 7.0).abs() < 1e-9);

        let outside = bilinear(&img, -0.4, 1.4);
        assert!(outside[0].abs() < 1e-9 && (outside[1] - 200.0).abs() < 1e-9);
    }

    #[test]
    fn feather_peaks_at_the_centre_and_fades_to_the_edges() {
        assert!((feather(99.5, 74.5, W, H) - 1.0).abs() < 1e-12);
        assert!(feather(-0.5, 74.5, W, H) <= 1e-6);
        assert!(feather(-0.5, 74.5, W, H) > 0.0);
        assert!(feather(20.0, 74.5, W, H) < feather(60.0, 74.5, W, H));
    }
}
