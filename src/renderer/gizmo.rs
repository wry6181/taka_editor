use glam::{Mat4, Vec3};

use super::moveable::Moveable;

// ── Shared drag state ──────────────────────────────────────────────────

pub struct GizmoState {
    pub drag_axis: Option<usize>,
    pub drag_start_mouse: Option<(f64, f64)>,
    pub drag_start_pos: Option<Vec3>,
    pub drag_start_matrix: Option<Mat4>,
    pub hovered_axis: Option<usize>,
}

impl GizmoState {
    pub fn new() -> Self {
        Self {
            drag_axis: None,
            drag_start_mouse: None,
            drag_start_pos: None,
            drag_start_matrix: None,
            hovered_axis: None,
        }
    }

    fn store_drag(&mut self, axis: usize, mouse: (f64, f64), movable: &dyn Moveable) {
        self.drag_axis = Some(axis);
        self.drag_start_mouse = Some(mouse);
        self.drag_start_pos = Some(movable.model_matrix().transform_point3(movable.center()));
        self.drag_start_matrix = Some(movable.model_matrix());
    }

    fn clear_drag(&mut self) -> bool {
        let had = self.drag_axis.is_some();
        self.drag_axis = None;
        self.drag_start_mouse = None;
        self.drag_start_pos = None;
        self.drag_start_matrix = None;
        had
    }

    pub fn set_hovered(&mut self, axis: Option<usize>) {
        self.hovered_axis = axis;
    }

    pub fn hovered_axis(&self) -> Option<usize> {
        self.hovered_axis
    }
}

// ── Common axis helpers ────────────────────────────────────────────────

fn axis_dir(axis: usize) -> Vec3 {
    match axis { 0 => Vec3::X, 1 => Vec3::Y, _ => Vec3::Z }
}

fn axis_color(axis: usize) -> Vec3 {
    match axis { 0 => Vec3::new(1.0, 0.0, 0.0), 1 => Vec3::new(0.0, 1.0, 0.0), _ => Vec3::new(0.0, 0.0, 1.0) }
}

/// World-space scale factor extracted from a model matrix.
fn model_world_scale(model: &Mat4) -> f32 {
    let sx = model.col(0).truncate().length();
    let sy = model.col(1).truncate().length();
    let sz = model.col(2).truncate().length();
    sx.max(sy).max(sz)
}

/// Axis length in world space – proportional to object, never gets buried.
fn gizmo_axis_len(movable: &dyn Moveable, model: &Mat4) -> f32 {
    let s = model_world_scale(model).max(0.2);
    (movable.bounding_size() * s * 0.7).max(1.5)
}

/// Rotation ring radius – proportional to object, never gets buried.
fn gizmo_rotate_radius(movable: &dyn Moveable, model: &Mat4) -> f32 {
    let s = model_world_scale(model).max(0.2);
    (movable.bounding_size() * s * 0.55).max(1.5)
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

// ── The Gizmo trait ────────────────────────────────────────────────────

pub trait Gizmo {
    fn name(&self) -> &'static str;

    fn hit_test(
        &self,
        movable: &dyn Moveable,
        ndc_x: f32,
        ndc_y: f32,
        view: Mat4,
        proj: Mat4,
        threshold: f32,
    ) -> Option<usize>;

    fn start_drag(&mut self, axis: usize, movable: &dyn Moveable, mouse: (f64, f64));
    fn apply_drag(&mut self, movable: &mut dyn Moveable, px: f64, py: f64, config_w: u32, config_h: u32, view: Mat4, proj: Mat4);
    fn end_drag(&mut self) -> bool;
    fn is_dragging(&self) -> bool;

    /// Lines for the gizmo overlay (axes, handles, visual guides).
    fn axis_lines(&self, movable: &dyn Moveable, model: Mat4) -> Vec<(Vec3, Vec3, Vec3)>;
}

// ── Translate ──────────────────────────────────────────────────────────

pub struct TranslateGizmo {
    state: GizmoState,
}

impl TranslateGizmo {
    pub fn new() -> Self {
        Self { state: GizmoState::new() }
    }

    fn screen_vec(vp: Mat4, world_pos: Vec3, dir: Vec3, config_w: u32, config_h: u32) -> (f64, f64) {
        let base = vp.project_point3(world_pos);
        let tip = vp.project_point3(world_pos + dir);
        let bx = (base.x as f64 + 1.0) * 0.5 * config_w as f64;
        let by = (1.0 - base.y as f64) * 0.5 * config_h as f64;
        let tx = (tip.x as f64 + 1.0) * 0.5 * config_w as f64;
        let ty = (1.0 - tip.y as f64) * 0.5 * config_h as f64;
        (tx - bx, ty - by)
    }
}

impl Gizmo for TranslateGizmo {
    fn name(&self) -> &'static str { "translate" }

    fn hit_test(&self, movable: &dyn Moveable, ndc_x: f32, ndc_y: f32, view: Mat4, proj: Mat4, threshold: f32) -> Option<usize> {
        let center = movable.model_matrix().transform_point3(movable.center());
        let axis_len = gizmo_axis_len(movable, &movable.model_matrix());
        for axis in 0..3 {
            if !movable.gizmo_axes()[axis] { continue; }
            let dir = axis_dir(axis);
            let start_ndc = project_to_ndc(center, view, proj);
            let end_ndc = project_to_ndc(center + dir * axis_len, view, proj);
            if ndc_segment_distance(ndc_x, ndc_y, start_ndc, end_ndc) < threshold {
                return Some(axis);
            }
        }
        None
    }

    fn start_drag(&mut self, axis: usize, movable: &dyn Moveable, mouse: (f64, f64)) {
        self.state.store_drag(axis, mouse, movable);
    }

    fn apply_drag(&mut self, movable: &mut dyn Moveable, px: f64, py: f64, config_w: u32, config_h: u32, view: Mat4, proj: Mat4) {
        let Some(ref state) = self.state.drag_axis else { return };
        let axis = *state;
        let Some((sx, sy)) = self.state.drag_start_mouse else { return };
        let Some(start_pos) = self.state.drag_start_pos else { return };
        let Some(start_matrix) = self.state.drag_start_matrix else { return };

        let vp = proj * view;
        let dx = px - sx;
        let dy = py - sy;
        let dir = axis_dir(axis);
        let (svx, svy) = Self::screen_vec(vp, start_pos, dir, config_w, config_h);
        let len_sq = svx * svx + svy * svy;
        if len_sq < 1.0 { return; }
        let world_offset = ((dx * svx + dy * svy) / len_sq) as f32;
        let translation = dir * world_offset;
        let mut new_matrix = start_matrix;
        new_matrix.w_axis = (start_matrix.w_axis.truncate() + translation).extend(1.0);
        movable.set_model_matrix(new_matrix);
    }

    fn end_drag(&mut self) -> bool { self.state.clear_drag() }
    fn is_dragging(&self) -> bool { self.state.drag_axis.is_some() }

    fn axis_lines(&self, movable: &dyn Moveable, model: Mat4) -> Vec<(Vec3, Vec3, Vec3)> {
        let center = model.transform_point3(movable.center());
        let axis_len = gizmo_axis_len(movable, &model);
        let mut lines = Vec::with_capacity(16);
        for axis in 0..3 {
            if !movable.gizmo_axes()[axis] { continue; }
            let dir = axis_dir(axis);
            let tip = center + dir * axis_len;
            let c = if self.state.hovered_axis() == Some(axis) { Vec3::ONE } else { axis_color(axis) };
            lines.push((center, tip, c));
            // cone at tip
            let cone_len = (axis_len * 0.08).max(0.1);
            let cone_radius = (axis_len * 0.04).max(0.06);
            let base = tip - dir * cone_len;
            let step = std::f32::consts::TAU / 12.0;
            for i in 0..12 {
                let a = i as f32 * step;
                let b = (i + 1) as f32 * step;
                // circle at cone base in the plane perpendicular to dir
                let (p1, p2) = if dir == Vec3::X {
                    (Vec3::Y, Vec3::Z)
                } else if dir == Vec3::Y {
                    (Vec3::X, Vec3::Z)
                } else {
                    (Vec3::X, Vec3::Y)
                };
                let b0 = base + (p1 * a.cos() + p2 * a.sin()) * cone_radius;
                let b1 = base + (p1 * b.cos() + p2 * b.sin()) * cone_radius;
                lines.push((b0, b1, c)); // base ring
                lines.push((b0, tip, c)); // edge to tip
            }
        }
        lines
    }
}

// ── Rotate ─────────────────────────────────────────────────────────────

pub struct RotateGizmo {
    state: GizmoState,
}

impl RotateGizmo {
    pub fn new() -> Self {
        Self { state: GizmoState::new() }
    }
}

impl Gizmo for RotateGizmo {
    fn name(&self) -> &'static str { "rotate" }

    fn hit_test(&self, movable: &dyn Moveable, ndc_x: f32, ndc_y: f32, view: Mat4, proj: Mat4, threshold: f32) -> Option<usize> {
        let center = movable.model_matrix().transform_point3(movable.center());
        let radius = gizmo_rotate_radius(movable, &movable.model_matrix());
        let steps = 32;
        let step = std::f32::consts::TAU / steps as f32;
        for axis in 0..3 {
            if !movable.gizmo_axes()[axis] { continue; }
            let (ax, ay) = match axis { 0 => (Vec3::Y, Vec3::Z), 1 => (Vec3::X, Vec3::Z), _ => (Vec3::X, Vec3::Y) };
            for i in 0..steps {
                let a = i as f32 * step;
                let b = (i + 1) as f32 * step;
                let p0 = project_to_ndc(center + (ax * a.cos() + ay * a.sin()) * radius, view, proj);
                let p1 = project_to_ndc(center + (ax * b.cos() + ay * b.sin()) * radius, view, proj);
                if ndc_segment_distance(ndc_x, ndc_y, p0, p1) < threshold {
                    return Some(axis);
                }
            }
        }
        None
    }

    fn start_drag(&mut self, axis: usize, movable: &dyn Moveable, mouse: (f64, f64)) {
        self.state.store_drag(axis, mouse, movable);
    }

    fn apply_drag(&mut self, movable: &mut dyn Moveable, px: f64, py: f64, config_w: u32, config_h: u32, view: Mat4, proj: Mat4) {
        let Some(ref state) = self.state.drag_axis else { return };
        let axis = *state;
        let Some((sx, sy)) = self.state.drag_start_mouse else { return };
        let Some(start_matrix) = self.state.drag_start_matrix else { return };
        let Some(start_pos) = self.state.drag_start_pos else { return };

        // Project center to screen
        let vp = proj * view;
        let center_ndc = vp.project_point3(start_pos);
        let cx = (center_ndc.x as f64 + 1.0) * 0.5 * config_w as f64;
        let cy = (1.0 - center_ndc.y as f64) * 0.5 * config_h as f64;

        let angle_start = (sy - cy).atan2(sx - cx);
        let angle_cur = (py - cy).atan2(px - cx);
        let delta_angle = angle_cur - angle_start;

        let rot = Mat4::from_axis_angle(axis_dir(axis), -delta_angle as f32);
        let new_matrix = Mat4::from_translation(start_pos) * rot * Mat4::from_translation(-start_pos) * start_matrix;
        movable.set_model_matrix(new_matrix);
    }

    fn end_drag(&mut self) -> bool { self.state.clear_drag() }
    fn is_dragging(&self) -> bool { self.state.drag_axis.is_some() }

    fn axis_lines(&self, movable: &dyn Moveable, model: Mat4) -> Vec<(Vec3, Vec3, Vec3)> {
        let center = model.transform_point3(movable.center());
        let radius = gizmo_rotate_radius(movable, &model);
        let steps = 32;
        let step = std::f32::consts::TAU / steps as f32;
        let mut lines = Vec::with_capacity(3 * steps);
        for axis in 0..3 {
            if !movable.gizmo_axes()[axis] { continue; }
            let (ax, ay) = match axis { 0 => (Vec3::Y, Vec3::Z), 1 => (Vec3::X, Vec3::Z), _ => (Vec3::X, Vec3::Y) };
            let c = if self.state.hovered_axis() == Some(axis) { Vec3::ONE } else { axis_color(axis) };
            for i in 0..steps {
                let a = i as f32 * step;
                let b = (i + 1) as f32 * step;
                let p0 = center + (ax * a.cos() + ay * a.sin()) * radius;
                let p1 = center + (ax * b.cos() + ay * b.sin()) * radius;
                lines.push((p0, p1, c));
            }
        }
        lines
    }
}

// ── Scale ──────────────────────────────────────────────────────────────

pub struct ScaleGizmo {
    state: GizmoState,
}

impl ScaleGizmo {
    pub fn new() -> Self {
        Self { state: GizmoState::new() }
    }
}

impl Gizmo for ScaleGizmo {
    fn name(&self) -> &'static str { "scale" }

    fn hit_test(&self, movable: &dyn Moveable, ndc_x: f32, ndc_y: f32, view: Mat4, proj: Mat4, threshold: f32) -> Option<usize> {
        let center = movable.model_matrix().transform_point3(movable.center());
        let axis_len = gizmo_axis_len(movable, &movable.model_matrix());
        let _handle_size = (axis_len * 0.1).max(0.15);
        for axis in 0..3 {
            if !movable.gizmo_axes()[axis] { continue; }
            let dir = axis_dir(axis);
            let tip_ndc = project_to_ndc(center + dir * axis_len, view, proj);
            if (glam::vec2(ndc_x, ndc_y) - tip_ndc).length() < threshold {
                return Some(axis);
            }
        }
        None
    }

    fn start_drag(&mut self, axis: usize, movable: &dyn Moveable, mouse: (f64, f64)) {
        self.state.store_drag(axis, mouse, movable);
    }

    fn apply_drag(&mut self, movable: &mut dyn Moveable, px: f64, py: f64, config_w: u32, config_h: u32, view: Mat4, proj: Mat4) {
        let Some(ref s_axis) = self.state.drag_axis else { return };
        let axis = *s_axis;
        let Some((sx, sy)) = self.state.drag_start_mouse else { return };
        let Some(start_matrix) = self.state.drag_start_matrix else { return };
        let Some(start_pos) = self.state.drag_start_pos else { return };

        let dir = axis_dir(axis);
        let vp = proj * view;

        // Use the gizmo's on-screen axis length as reference for sensitivity
        let axis_len = gizmo_axis_len(movable, &start_matrix);
        let tip_world = start_pos + dir * axis_len;
        let base_ndc = vp.project_point3(start_pos);
        let tip_ndc = vp.project_point3(tip_world);
        let bx = (base_ndc.x as f64 + 1.0) * 0.5 * config_w as f64;
        let by = (1.0 - base_ndc.y as f64) * 0.5 * config_h as f64;
        let tx = (tip_ndc.x as f64 + 1.0) * 0.5 * config_w as f64;
        let ty = (1.0 - tip_ndc.y as f64) * 0.5 * config_h as f64;
        let svx = tx - bx;
        let svy = ty - by;
        let axis_len_pixels = (svx * svx + svy * svy).sqrt();
        if axis_len_pixels < 1.0 { return; }

        let dx = px - sx;
        let dy = py - sy;
        let pixel_delta = (dx * svx + dy * svy) / axis_len_pixels;
        let scale = 1.0 + pixel_delta / 150.0;

        let mut s = Vec3::ONE;
        s[axis] = (scale as f32).max(0.01);
        let scale_mat = Mat4::from_scale(s);
        // Scale around the object center (translate → scale → translate back)
        let t = start_pos;
        let new_matrix = Mat4::from_translation(t) * scale_mat * Mat4::from_translation(-t) * start_matrix;
        movable.set_model_matrix(new_matrix);
    }

    fn end_drag(&mut self) -> bool { self.state.clear_drag() }
    fn is_dragging(&self) -> bool { self.state.drag_axis.is_some() }

    fn axis_lines(&self, movable: &dyn Moveable, model: Mat4) -> Vec<(Vec3, Vec3, Vec3)> {
        let center = model.transform_point3(movable.center());
        let size_ref = self.state.drag_start_matrix.unwrap_or(model);
        let axis_len = gizmo_axis_len(movable, &size_ref);
        let hs = (axis_len * 0.08).max(0.12);
        let mut lines = Vec::with_capacity(12 * 3);
        for axis in 0..3 {
            if !movable.gizmo_axes()[axis] { continue; }
            let dir = axis_dir(axis);
            let tip = center + dir * axis_len;
            let c = if self.state.hovered_axis() == Some(axis) { Vec3::ONE } else { axis_color(axis) };
            lines.push((center, tip, c));
            // box at tip
            let corners = [
                Vec3::new(-hs, -hs, -hs), Vec3::new(hs, -hs, -hs),
                Vec3::new(hs, hs, -hs), Vec3::new(-hs, hs, -hs),
                Vec3::new(-hs, -hs, hs), Vec3::new(hs, -hs, hs),
                Vec3::new(hs, hs, hs), Vec3::new(-hs, hs, hs),
            ];
            let edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ];
            for (i, j) in edges {
                let p0 = tip + corners[i];
                let p1 = tip + corners[j];
                lines.push((p0, p1, c));
            }
        }
        lines
    }
}

// ── Enum over all gizmo types ──────────────────────────────────────────

pub enum GizmoMode {
    Translate(TranslateGizmo),
    Rotate(RotateGizmo),
    Scale(ScaleGizmo),
}

impl GizmoMode {
    pub fn new_translate() -> Self { Self::Translate(TranslateGizmo::new()) }
    pub fn new_rotate() -> Self { Self::Rotate(RotateGizmo::new()) }
    pub fn new_scale() -> Self { Self::Scale(ScaleGizmo::new()) }

    pub fn name(&self) -> &'static str {
        match self {
            GizmoMode::Translate(_) => "translate",
            GizmoMode::Rotate(_) => "rotate",
            GizmoMode::Scale(_) => "scale",
        }
    }

    pub fn set_hovered(&mut self, axis: Option<usize>) {
        match self {
            GizmoMode::Translate(g) => g.state.set_hovered(axis),
            GizmoMode::Rotate(g) => g.state.set_hovered(axis),
            GizmoMode::Scale(g) => g.state.set_hovered(axis),
        }
    }

    pub fn hovered_axis(&self) -> Option<usize> {
        match self {
            GizmoMode::Translate(g) => g.state.hovered_axis(),
            GizmoMode::Rotate(g) => g.state.hovered_axis(),
            GizmoMode::Scale(g) => g.state.hovered_axis(),
        }
    }
}

macro_rules! delegate_gizmo {
    ($self:ident, $method:ident, $($args:expr),*) => {
        match $self {
            GizmoMode::Translate(g) => g.$method($($args),*),
            GizmoMode::Rotate(g) => g.$method($($args),*),
            GizmoMode::Scale(g) => g.$method($($args),*),
        }
    };
}

impl Gizmo for GizmoMode {
    fn name(&self) -> &'static str {
        delegate_gizmo!(self, name,)
    }

    fn hit_test(&self, movable: &dyn Moveable, ndc_x: f32, ndc_y: f32, view: Mat4, proj: Mat4, threshold: f32) -> Option<usize> {
        delegate_gizmo!(self, hit_test, movable, ndc_x, ndc_y, view, proj, threshold)
    }

    fn start_drag(&mut self, axis: usize, movable: &dyn Moveable, mouse: (f64, f64)) {
        delegate_gizmo!(self, start_drag, axis, movable, mouse)
    }

    fn apply_drag(&mut self, movable: &mut dyn Moveable, px: f64, py: f64, config_w: u32, config_h: u32, view: Mat4, proj: Mat4) {
        delegate_gizmo!(self, apply_drag, movable, px, py, config_w, config_h, view, proj)
    }

    fn end_drag(&mut self) -> bool {
        delegate_gizmo!(self, end_drag,)
    }

    fn is_dragging(&self) -> bool {
        delegate_gizmo!(self, is_dragging,)
    }

    fn axis_lines(&self, movable: &dyn Moveable, model: Mat4) -> Vec<(Vec3, Vec3, Vec3)> {
        delegate_gizmo!(self, axis_lines, movable, model)
    }
}
