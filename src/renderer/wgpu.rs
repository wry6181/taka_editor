use bytemuck_derive::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use web_sys::HtmlCanvasElement;
use wgpu::SurfaceTarget;

use super::camera::Camera;
use super::pass::RenderPass;
use super::passes::grid_pass::GridPass;
use super::passes::line_pass::LinePass;
use super::passes::mesh_pass::{MeshPass, GpuHitResult};
use super::ray::Ray;


#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Globals {
    view: [f32; 16],
    projection: [f32; 16],
}
impl Globals {
    fn new(view: Mat4, projection: Mat4) -> Self {
        Self { view: view.to_cols_array(), projection: projection.to_cols_array() }
    }
}

#[allow(dead_code)]
pub struct WgpuRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    backend: wgpu::Backend,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    globals_ubo: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    camera: Camera,
    mesh_pass: MeshPass,
    line_pass: LinePass,
    grid_pass: GridPass,
    clear_color: wgpu::Color,
    axis_lines: Vec<(Vec3, Vec3, Vec3)>,
    debug_ray_origin: Option<Vec3>,
    debug_ray_dir: Option<Vec3>,
    debug_ray_hit: Option<Vec3>,
    gpu_raycast_active: bool,
    show_mesh: bool,
    mesh_selected: bool,
    mesh_model_matrix: Mat4,
    mesh_drag_axis: Option<usize>,
    mesh_drag_start_mouse: Option<(f64, f64)>,
    mesh_drag_start_pos: Option<Vec3>,
    mesh_drag_start_matrix: Option<Mat4>,
    mesh_highlight_color: Vec3,
}

impl WgpuRenderer {
    pub async fn new(canvas: HtmlCanvasElement, width: u32, height: u32) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let surface = instance
            .create_surface(SurfaceTarget::Canvas(canvas))
            .map_err(|e| format!("surface creation failed: {:?}", e))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("adapter request failed: {:?}", e))?;
        let info = adapter.get_info();
        let backend = info.backend;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .map_err(|e| format!("device request failed: {:?}", e))?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.first().copied().ok_or("no surface format".to_string())?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let depth_texture = Self::create_depth_texture(&device, &config);
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout::global"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Globals>() as u64),
                },
                count: None,
            }],
        });

        let globals_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ubo::globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group::globals"),
            layout: &globals_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals_ubo.as_entire_binding() }],
        });

        let mesh_pass = MeshPass::new(&device, &queue, config.format, &globals_bgl, globals_bind_group.clone());
        let line_pass = LinePass::new(&device, config.format, &globals_bgl, globals_bind_group.clone());

        let mut grid_pass = GridPass::new(&device, config.format, &globals_bgl, globals_bind_group.clone());
        grid_pass.set_grid(&device, &queue, 1000.0, 100, 5);

        let camera = Camera::new(width, height);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            backend,
            depth_texture,
            depth_view,
            globals_ubo,
            globals_bind_group,
            camera,
            mesh_pass,
            line_pass,
            grid_pass,
            clear_color: wgpu::Color { r: 0.24, g: 0.24, b: 0.24, a: 1.0 },
            axis_lines: Vec::new(),
            debug_ray_origin: None,
            debug_ray_dir: None,
            debug_ray_hit: None,
            gpu_raycast_active: false,
            show_mesh: true,
            mesh_selected: false,
            mesh_model_matrix: Mat4::IDENTITY,
            mesh_drag_axis: None,
            mesh_drag_start_mouse: None,
            mesh_drag_start_pos: None,
            mesh_drag_start_matrix: None,
            mesh_highlight_color: Vec3::new(1.0, 0.5, 0.0),
        })
    }

    fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d { width: config.width.max(1), height: config.height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    pub fn render(&mut self) {
        let globals = Globals::new(self.camera.get_view_matrix(), self.camera.get_projection_matrix());
        self.queue.write_buffer(&self.globals_ubo, 0, bytemuck::cast_slice(&[globals]));

        self.mesh_pass.prepare(&self.queue, &self.camera);
        self.line_pass.prepare(&self.queue, &self.camera);
        self.grid_pass.prepare(&self.queue, &self.camera);

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.show_mesh {
                self.mesh_pass.render(&mut rpass);
            }
            self.grid_pass.render(&mut rpass);
            self.line_pass.render(&mut rpass);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        self.check_gpu_raycast();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_texture = Self::create_depth_texture(&self.device, &self.config);
        self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.camera.resize(width, height);
    }

    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DX12",
            wgpu::Backend::Gl => "WebGL/OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            wgpu::Backend::Noop => "noop",
        }
    }

    pub fn camera_orbit(&mut self, dx: f64, dy: f64) {
        self.camera.orbit(dx as f32, dy as f32);
    }

    pub fn camera_zoom(&mut self, delta: f64) {
        self.camera.zoom(delta as f32);
    }

    pub fn set_light_direction(&mut self, direction: Vec3) {
        self.mesh_pass.set_light_direction(direction);
    }

    pub fn set_lines(&mut self, lines: &[(Vec3, Vec3, Vec3)]) {
        self.axis_lines = lines.to_vec();
        self.flush_lines();
    }

    fn wireframe_sphere(center: Vec3, radius: f32, color: Vec3, segments: u32) -> Vec<(Vec3, Vec3, Vec3)> {
        let mut lines = Vec::new();
        let step = std::f32::consts::TAU / segments as f32;
        for p in 0..3 {
            let (ax, ay) = match p {
                0 => (Vec3::X, Vec3::Y),
                1 => (Vec3::X, Vec3::Z),
                _ => (Vec3::Y, Vec3::Z),
            };
            for i in 0..segments {
                let a = i as f32 * step;
                let b = (i + 1) as f32 * step;
                let p0 = center + (ax * a.cos() + ay * a.sin()) * radius;
                let p1 = center + (ax * b.cos() + ay * b.sin()) * radius;
                lines.push((p0, p1, color));
            }
        }
        lines
    }

    fn flush_lines(&mut self) {
        let mut all = self.axis_lines.clone();

        if let (Some(origin), Some(dir)) = (self.debug_ray_origin, self.debug_ray_dir) {
            let color = Vec3::new(1.0, 0.65, 0.0);
            if let Some(hit) = self.debug_ray_hit {
                all.push((origin, hit, color));
                all.extend(Self::wireframe_sphere(hit, 0.2, color, 16));
            } else {
                let far = origin + dir * 100.0;
                all.push((origin, far, color));
            }
        }

        // Mesh highlight + gizmo when selected
        if self.mesh_selected && self.show_mesh {
            let center = self.mesh_model_matrix.transform_point3(self.mesh_pass.mesh_center());
            let (bmin, bmax) = self.mesh_pass.bounding_box();
            let size = (bmax - bmin).length();
            let axis_len = (size * 0.8).max(2.0);

            // Highlight: wireframe overlay in orange
            let highlight_color = Vec3::new(1.0, 0.5, 0.1);
            let (_, indices) = self.mesh_pass.raw_triangles();
            for tri in indices.chunks_exact(3) {
                let p0 = self.mesh_model_matrix.transform_point3(self.mesh_pass.vertex_position(tri[0]));
                let p1 = self.mesh_model_matrix.transform_point3(self.mesh_pass.vertex_position(tri[1]));
                let p2 = self.mesh_model_matrix.transform_point3(self.mesh_pass.vertex_position(tri[2]));
                all.push((p0, p1, highlight_color));
                all.push((p1, p2, highlight_color));
                all.push((p2, p0, highlight_color));
            }

            // Gizmo axes (from mesh center, scaled to mesh size)
            let gc = center;
            all.push((gc, gc + Vec3::X * axis_len, Vec3::new(1.0, 0.0, 0.0)));
            all.push((gc, gc + Vec3::Y * axis_len, Vec3::new(0.0, 1.0, 0.0)));
            all.push((gc, gc + Vec3::Z * axis_len, Vec3::new(0.0, 0.0, 1.0)));
            let r = (axis_len * 0.04).max(0.06);
            all.extend(Self::wireframe_sphere(gc + Vec3::X * axis_len, r, Vec3::new(1.0, 0.0, 0.0), 8));
            all.extend(Self::wireframe_sphere(gc + Vec3::Y * axis_len, r, Vec3::new(0.0, 1.0, 0.0), 8));
            all.extend(Self::wireframe_sphere(gc + Vec3::Z * axis_len, r, Vec3::new(0.0, 0.0, 1.0), 8));
        }

        self.line_pass.set_lines(&self.device, &self.queue, &all);
    }

    pub fn raycast_gpu(&mut self, mouse_px: f64, mouse_py: f64) {
        let view = self.camera.get_view_matrix();
        let proj = self.camera.get_projection_matrix();
        let inv_vp = (proj * view).inverse();

        let w = self.config.width as f64;
        let h = self.config.height as f64;
        let ndc_x = (2.0 * mouse_px / w - 1.0) as f32;
        let ndc_y = (1.0 - 2.0 * mouse_py / h) as f32;

        let ray = Ray::from_screen(ndc_x, ndc_y, inv_vp);

        self.debug_ray_origin = Some(ray.origin);
        self.debug_ray_dir = Some(ray.direction);
        self.debug_ray_hit = None;
        self.gpu_raycast_active = true;

        let inv_model = self.mesh_model_matrix.inverse();
        let model_ray = Ray {
            origin: inv_model.transform_point3(ray.origin),
            direction: inv_model.transform_vector3(ray.direction),
        };
        let staging = self.mesh_pass.dispatch_raycast(&self.device, &self.queue, &model_ray);
        let staged = staging.clone();
        staging.map_async(wgpu::MapMode::Read, .., move |_| {
            let data = staged.get_mapped_range(..);
            let hr: &GpuHitResult = bytemuck::from_bytes(&data);
            super::GPU_RAYCAST_RESULT.with(|r| {
                *r.borrow_mut() = Some(crate::renderer::GpuRaycastOutcome {
                    hit: hr.hit != 0,
                    t: f32::from_bits(hr.t_bits),
                });
            });
            drop(data);
            staged.unmap();
        });
    }

    pub fn check_gpu_raycast(&mut self) {
        if !self.gpu_raycast_active {
            return;
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
        let ray_origin = self.debug_ray_origin;
        let ray_dir = self.debug_ray_dir;
        super::GPU_RAYCAST_RESULT.with(|r| {
            if let Some(outcome) = r.borrow_mut().take() {
                if outcome.hit {
                    if let (Some(origin), Some(dir)) = (ray_origin, ray_dir) {
                        self.debug_ray_hit = Some(origin + dir * outcome.t);
                    }
                }
                self.gpu_raycast_active = false;
                self.flush_lines();
            }
        });
    }

    pub fn set_clear_color(&mut self, r: f64, g: f64, b: f64) {
        self.clear_color = wgpu::Color { r, g, b, a: 1.0 };
    }

    pub fn set_grid(&mut self, size: f32, divisions: u32, major_every: u32) {
        self.grid_pass.set_grid(&self.device, &self.queue, size, divisions, major_every);
    }

    pub fn canvas_width(&self) -> f64 {
        self.config.width as f64
    }

    pub fn canvas_height(&self) -> f64 {
        self.config.height as f64
    }

    pub fn get_camera_info(&self) -> Mat4 {
        self.camera.get_view_matrix()
    }

    pub fn get_inv_vp(&self) -> Mat4 {
        let view = self.camera.get_view_matrix();
        let proj = self.camera.get_projection_matrix();
        (proj * view).inverse()
    }

    pub fn get_view_proj(&self) -> (Mat4, Mat4) {
        (self.camera.get_view_matrix(), self.camera.get_projection_matrix())
    }

    pub fn mesh_is_dragging(&self) -> bool {
        self.mesh_drag_axis.is_some()
    }

    pub fn handle_mousedown(&mut self, px: f64, py: f64) -> bool {
        // Mesh gizmo hit test
        if self.mesh_selected {
            let w = self.config.width as f64;
            let h = self.config.height as f64;
            let ndc_x = (2.0 * px / w - 1.0) as f32;
            let ndc_y = (1.0 - 2.0 * py / h) as f32;
            let (view, proj) = self.get_view_proj();
            if let Some(axis) = self.mesh_gizmo_hit_screen(ndc_x, ndc_y, view, proj, 0.04) {
                self.mesh_drag_axis = Some(axis);
                self.mesh_drag_start_mouse = Some((px, py));
                let center = self.mesh_model_matrix.transform_point3(self.mesh_pass.mesh_center());
                self.mesh_drag_start_pos = Some(center);
                self.mesh_drag_start_matrix = Some(self.mesh_model_matrix);
                return true;
            }
        }
        false
    }

    pub fn handle_mousemove(&mut self, px: f64, py: f64) {
        if let Some(axis) = self.mesh_drag_axis {
            self.mesh_drag_mousemove(axis, px, py);
        }
    }

    pub fn handle_mouseup(&mut self) -> bool {
        let had_drag = self.mesh_drag_axis.is_some();
        self.mesh_drag_axis = None;
        had_drag
    }

    fn mesh_gizmo_hit_screen(&self, ndc_x: f32, ndc_y: f32, view: Mat4, proj: Mat4, threshold: f32) -> Option<usize> {
        let center = self.mesh_model_matrix.transform_point3(self.mesh_pass.mesh_center());
        let (bmin, bmax) = self.mesh_pass.bounding_box();
        let size = (bmax - bmin).length();
        let axis_len = (size * 0.8).max(2.0);

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

    fn mesh_drag_mousemove(&mut self, axis: usize, px: f64, py: f64) {
        let Some((sx, sy)) = self.mesh_drag_start_mouse else { return };
        let Some(start_pos) = self.mesh_drag_start_pos else { return };
        let Some(start_matrix) = self.mesh_drag_start_matrix else { return };

        self.debug_ray_hit = None;

        let w = self.config.width;
        let h = self.config.height;
        let view = self.camera.get_view_matrix();
        let proj = self.camera.get_projection_matrix();
        let vp = proj * view;

        let screen_vec = |dir: Vec3| -> (f64, f64) {
            let base = vp.project_point3(start_pos);
            let tip = vp.project_point3(start_pos + dir);
            let bx = (base.x as f64 + 1.0) * 0.5 * w as f64;
            let by = (1.0 - base.y as f64) * 0.5 * h as f64;
            let tx = (tip.x as f64 + 1.0) * 0.5 * w as f64;
            let ty = (1.0 - tip.y as f64) * 0.5 * h as f64;
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
        if len_sq < 1.0 { return; }
        let world_offset = ((dx * svx + dy * svy) / len_sq) as f32;
        let translation = axis_dir * world_offset;
        let start_trans = start_matrix.w_axis.truncate();
        self.mesh_model_matrix = Mat4::from_translation(start_trans + translation);
        self.mesh_pass.set_model_matrix(self.mesh_model_matrix);
        self.flush_lines();
    }

    pub fn toggle_select_mesh(&mut self) {
        self.mesh_selected = !self.mesh_selected;
        self.flush_lines();
    }

    pub fn mesh_is_selected(&self) -> bool {
        self.mesh_selected
    }

    pub fn select_mesh_at_screen(&mut self, px: f64, py: f64) -> bool {
        let w = self.config.width as f64;
        let h = self.config.height as f64;
        let ndc_x = (2.0 * px / w - 1.0) as f32;
        let ndc_y = (1.0 - 2.0 * py / h) as f32;
        let inv_vp = self.get_inv_vp();
        let ray = Ray::from_screen(ndc_x, ndc_y, inv_vp);
        if self.mesh_pass.ray_intersect_with_model(&ray, &self.mesh_model_matrix).is_some() {
            self.mesh_selected = true;
            self.flush_lines();
            true
        } else {
            false
        }
    }

    pub fn deselect_mesh(&mut self) {
        self.mesh_selected = false;
        self.flush_lines();
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
