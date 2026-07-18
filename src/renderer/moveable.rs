use glam::{Mat4, Vec3};

use crate::renderer::ray::Ray;

pub trait Moveable {
    fn model_matrix(&self) -> Mat4;
    fn set_model_matrix(&mut self, m: Mat4);
    fn is_selected(&self) -> bool;
    fn set_selected(&mut self, selected: bool);
    fn center(&self) -> Vec3;
    fn bounding_size(&self) -> f32;
    fn ray_intersect(&self, ray: &Ray, model: &Mat4) -> Option<Vec3>;
    fn gizmo_color(&self) -> Vec3;
    /// Return (start, end, color) lines for the selected highlight overlay.
    fn gizmo_lines(&self, model: &Mat4) -> Vec<(Vec3, Vec3, Vec3)>;
}

pub struct Gizmo {
    pub drag_axis: Option<usize>,
    pub drag_start_mouse: Option<(f64, f64)>,
    pub drag_start_pos: Option<Vec3>,
    pub drag_start_matrix: Option<Mat4>,
}

impl Gizmo {
    pub fn new() -> Self {
        Self {
            drag_axis: None,
            drag_start_mouse: None,
            drag_start_pos: None,
            drag_start_matrix: None,
        }
    }

    pub fn is_dragging(&self) -> bool {
        self.drag_axis.is_some()
    }

    pub fn hit_test(
        &self,
        movable: &dyn Moveable,
        ndc_x: f32,
        ndc_y: f32,
        view: Mat4,
        proj: Mat4,
        threshold: f32,
    ) -> Option<usize> {
        let center = movable.model_matrix().transform_point3(movable.center());
        let axis_len = movable.bounding_size().max(2.0);

        for axis in 0..3 {
            let dir = match axis {
                0 => Vec3::X,
                1 => Vec3::Y,
                _ => Vec3::Z,
            };
            let start_ndc = project_to_ndc(center, view, proj);
            let end_ndc = project_to_ndc(center + dir * axis_len, view, proj);
            let dist = ndc_segment_distance(ndc_x, ndc_y, start_ndc, end_ndc);
            if dist < threshold {
                return Some(axis);
            }
        }
        None
    }

    pub fn start_drag(&mut self, mouse: (f64, f64), start_pos: Vec3, start_matrix: Mat4) {
        self.drag_start_mouse = Some(mouse);
        self.drag_start_pos = Some(start_pos);
        self.drag_start_matrix = Some(start_matrix);
    }

    pub fn end_drag(&mut self) -> bool {
        let had = self.drag_axis.is_some();
        self.drag_axis = None;
        self.drag_start_mouse = None;
        self.drag_start_pos = None;
        self.drag_start_matrix = None;
        had
    }

    pub fn apply_drag(
        &self,
        movable: &mut dyn Moveable,
        axis: usize,
        px: f64,
        py: f64,
        config_w: u32,
        config_h: u32,
        view: Mat4,
        proj: Mat4,
    ) {
        let Some((sx, sy)) = self.drag_start_mouse else { return };
        let Some(start_pos) = self.drag_start_pos else { return };
        let Some(start_matrix) = self.drag_start_matrix else { return };

        let vp = proj * view;

        let screen_vec = |dir: Vec3| -> (f64, f64) {
            let base = vp.project_point3(start_pos);
            let tip = vp.project_point3(start_pos + dir);
            let bx = (base.x as f64 + 1.0) * 0.5 * config_w as f64;
            let by = (1.0 - base.y as f64) * 0.5 * config_h as f64;
            let tx = (tip.x as f64 + 1.0) * 0.5 * config_w as f64;
            let ty = (1.0 - tip.y as f64) * 0.5 * config_h as f64;
            (tx - bx, ty - by)
        };

        let dx = px - sx;
        let dy = py - sy;

        let axis_dir = match axis {
            0 => Vec3::X,
            1 => Vec3::Y,
            _ => Vec3::Z,
        };
        let (svx, svy) = screen_vec(axis_dir);
        let len_sq = svx * svx + svy * svy;
        if len_sq < 1.0 {
            return;
        }
        let world_offset = ((dx * svx + dy * svy) / len_sq) as f32;
        let translation = axis_dir * world_offset;
        let start_trans = start_matrix.w_axis.truncate();
        let new_matrix = Mat4::from_translation(start_trans + translation);
        movable.set_model_matrix(new_matrix);
    }
}

fn project_to_ndc(p: Vec3, view: Mat4, proj: Mat4) -> glam::Vec2 {
    let clip = proj * view * p.extend(1.0);
    glam::vec2(clip.x / clip.w, clip.y / clip.w)
}

fn ndc_segment_distance(px: f32, py: f32, a: glam::Vec2, b: glam::Vec2) -> f32 {
    let ab = b - a;
    let ap = glam::vec2(px - a.x, py - a.y);
    let t = (ap.dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
    let closest = a + ab * t;
    glam::vec2(px - closest.x, py - closest.y).length()
}
