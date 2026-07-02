use glam::{Mat4, Vec3};

const ORBIT_SENSITIVITY: f32 = 0.3;

pub struct Camera {
    pub aspect_ratio: f32,
    pub distance: f32,
    pub target: Vec3,
    pub up: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

impl Camera {
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            aspect_ratio: screen_width as f32 / screen_height as f32,
            distance: 100.0,
            target: Vec3::ZERO,
            up: Vec3::Y,
            yaw: -90.0,
            pitch: 0.0,
            fov_y: 45.0,
        }
    }

    pub fn front(&self) -> Vec3 {
        Vec3::new(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        )
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        let position = self.target + self.distance * self.front();
        Mat4::look_at_rh(position, self.target, self.up)
    }

    pub fn get_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y.to_radians(), self.aspect_ratio, 0.1, 1000.0)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect_ratio = width as f32 / height as f32;
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * ORBIT_SENSITIVITY;
        self.pitch += dy * ORBIT_SENSITIVITY;
        self.pitch = self.pitch.clamp(-89.0, 89.0);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta).clamp(0.1, 500.0);
    }
}
