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
    /// Which axes the gizmo should show for this object (X, Y, Z).
    fn gizmo_axes(&self) -> [bool; 3] {
        [true; 3]
    }
}
