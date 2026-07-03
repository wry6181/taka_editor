use glam::{Mat4, Vec2, Vec3};

use super::ray::Ray;

#[derive(Clone)]
pub struct Remesh {
    pub points: Vec<Vec3>,
    pub triangles: Vec<(u32, u32, u32)>,
    pub selected: Option<usize>,
}

impl Remesh {
    pub fn new() -> Self {
        Self { points: Vec::new(), triangles: Vec::new(), selected: None }
    }

    pub fn add_point(&mut self, pos: Vec3) -> usize {
        let idx = self.points.len() as u32;
        self.points.push(pos);

        if self.points.len() >= 3 {
            let mut found_inside = None;
            for (i, &(a, b, c)) in self.triangles.iter().enumerate() {
                if point_in_triangle(pos, self.points[a as usize], self.points[b as usize], self.points[c as usize]) {
                    found_inside = Some(i);
                    break;
                }
            }

            if let Some(tri_idx) = found_inside {
                let (a, b, c) = self.triangles[tri_idx];
                self.triangles.remove(tri_idx);
                self.triangles.push((a, b, idx));
                self.triangles.push((b, c, idx));
                self.triangles.push((c, a, idx));
            } else {
                let (i1, i2) = nearest_two(&self.points, pos);
                self.triangles.push((i1, i2, idx));
            }
        }

        idx as usize
    }

    pub fn nearest_point(&self, pos: Vec3, threshold: f32) -> Option<usize> {
        self.points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, (p - pos).length_squared()))
            .filter(|&(_, d)| d < threshold * threshold)
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    pub fn select_near(&mut self, pos: Vec3) {
        self.selected = self.nearest_point(pos, 0.3);
        if self.selected.is_none() {
            // also try screen-space via pixel threshold — caller decides
        }
    }

    pub fn clear_selection(&mut self) {
        self.selected = None;
    }

    pub fn gizmo_axis_hit_screen(&self, ndc_x: f32, ndc_y: f32, view: Mat4, proj: Mat4, threshold: f32) -> Option<usize> {
        let idx = self.selected?;
        let pos = self.points[idx];
        let axis_len = 1.5;

        for axis in 0..3 {
            let dir = match axis {
                0 => Vec3::X,
                1 => Vec3::Y,
                _ => Vec3::Z,
            };
            let start_ndc = project_to_ndc(pos, view, proj);
            let end_ndc = project_to_ndc(pos + dir * axis_len, view, proj);
            let dist = ndc_segment_distance(ndc_x, ndc_y, start_ndc, end_ndc);
            if dist < threshold {
                return Some(axis);
            }
        }
        None
    }

    pub fn drag_selected(&mut self, axis: usize, ray: &Ray) {
        let Some(idx) = self.selected else { return };
        let pos = self.points[idx];
        let axis_dir = match axis {
            0 => Vec3::X,
            1 => Vec3::Y,
            _ => Vec3::Z,
        };

        // Intersect ray with a plane through the point, perpendicular to the axis.
        // Then project the hit onto the axis line.
        let plane_normal = axis_dir;
        let denom = plane_normal.dot(ray.direction);
        if denom.abs() < 1e-8 {
            return;
        }
        let t = plane_normal.dot(pos - ray.origin) / denom;
        if t < 0.0 {
            return;
        }
        let hit = ray.origin + ray.direction * t;
        let s = axis_dir.dot(hit - pos);
        self.points[idx] = pos + axis_dir * s;
    }

    pub fn hit_point_screen(
        &self,
        ndc_x: f32,
        ndc_y: f32,
        view: glam::Mat4,
        proj: glam::Mat4,
        threshold: f32,
    ) -> Option<usize> {
        for (i, &p) in self.points.iter().enumerate() {
            let clip = proj * view * p.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            let dist = ((ndc.x - ndc_x).powi(2) + (ndc.y - ndc_y).powi(2)).sqrt();
            if dist < threshold {
                return Some(i);
            }
        }
        None
    }
}

fn nearest_two(points: &[Vec3], pos: Vec3) -> (u32, u32) {
    let count = points.len() as u32;
    let mut indices: Vec<u32> = (0..count - 1).collect();
    indices.sort_by(|&a, &b| {
        let da = (points[a as usize] - pos).length_squared();
        let db = (points[b as usize] - pos).length_squared();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    (indices[0], indices[1])
}

fn point_in_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let n = ab.cross(ac);
    let n_len_sq = n.length_squared();
    if n_len_sq < 1e-12 {
        return false;
    }
    let u = n.cross(ac).dot(ap) / n_len_sq;
    let v = ab.cross(n).dot(ap) / n_len_sq;
    let w = 1.0 - u - v;
    u >= -1e-8 && u <= 1.0 + 1e-8 && v >= -1e-8 && v <= 1.0 + 1e-8 && w >= -1e-8 && w <= 1.0 + 1e-8
}

fn project_to_ndc(p: Vec3, view: Mat4, proj: Mat4) -> Vec2 {
    let clip = proj * view * p.extend(1.0);
    Vec2::new(clip.x / clip.w, clip.y / clip.w)
}

fn ndc_segment_distance(px: f32, py: f32, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-12 {
        return Vec2::new(px - a.x, py - a.y).length();
    }
    let t = ((px - a.x) * ab.x + (py - a.y) * ab.y) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let closest = a + ab * t;
    Vec2::new(px - closest.x, py - closest.y).length()
}
