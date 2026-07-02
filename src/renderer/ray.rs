use glam::{Mat4, Vec3, Vec4};

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn from_screen(mouse_ndc_x: f32, mouse_ndc_y: f32, inv_view_proj: Mat4) -> Self {
        let near = inv_view_proj * Vec4::new(mouse_ndc_x, mouse_ndc_y, -1.0, 1.0);
        let near_ws = near.truncate() / near.w;

        let far = inv_view_proj * Vec4::new(mouse_ndc_x, mouse_ndc_y, 1.0, 1.0);
        let far_ws = far.truncate() / far.w;

        Self { origin: near_ws, direction: (far_ws - near_ws).normalize() }
    }

    pub fn intersect_triangle(&self, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = self.direction.cross(edge2);
        let a = edge1.dot(h);
        if a.abs() < 1e-8 {
            return None;
        }
        let f = 1.0 / a;
        let s = self.origin - v0;
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = s.cross(edge1);
        let v_val = f * self.direction.dot(q);
        if v_val < 0.0 || u + v_val > 1.0 {
            return None;
        }
        let t = f * edge2.dot(q);
        if t < 0.0 {
            return None;
        }
        Some(t)
    }

    pub fn intersect_mesh(&self, positions: &[[f32; 3]], indices: &[u32]) -> Option<Vec3> {
        let mut best: Option<(f32, Vec3)> = None;
        for tri in indices.chunks_exact(3) {
            let v0 = Vec3::from(positions[tri[0] as usize]);
            let v1 = Vec3::from(positions[tri[1] as usize]);
            let v2 = Vec3::from(positions[tri[2] as usize]);
            if let Some(t) = self.intersect_triangle(v0, v1, v2) {
                if best.map_or(true, |(bt, _)| t < bt) {
                    best = Some((t, self.origin + self.direction * t));
                }
            }
        }
        best.map(|(_, p)| p)
    }
}
