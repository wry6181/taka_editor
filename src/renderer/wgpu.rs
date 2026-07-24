use bytemuck_derive::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use web_sys::HtmlCanvasElement;
use wgpu::SurfaceTarget;

use super::camera::Camera;
use super::gizmo::{Gizmo, GizmoMode};
use super::moveable::Moveable;
use super::pass::RenderPass;
use super::passes::grid_pass::GridPass;
use super::passes::line_pass::LinePass;
use super::passes::image_pass::ImagePass;
use super::passes::mesh_pass::{MeshPass, GpuHitResult};
use super::ray::Ray;

enum Selection {
    None,
    Mesh,
    Image,
}


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
    image_pass: ImagePass,
    grid_pass: GridPass,
    clear_color: wgpu::Color,
    axis_lines: Vec<(Vec3, Vec3, Vec3)>,
    debug_ray_origin: Option<Vec3>,
    debug_ray_dir: Option<Vec3>,
    debug_ray_hit: Option<Vec3>,
    gpu_raycast_active: bool,
    show_mesh: bool,
    selection: Selection,
    gizmo: GizmoMode,
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
        let image_pass = ImagePass::new(&device, config.format, &globals_bgl, globals_bind_group.clone());

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
            image_pass,
            grid_pass,
            clear_color: wgpu::Color { r: 0.24, g: 0.24, b: 0.24, a: 1.0 },
            axis_lines: Vec::new(),
            debug_ray_origin: None,
            debug_ray_dir: None,
            debug_ray_hit: None,
            gpu_raycast_active: false,
            show_mesh: true,
            selection: Selection::None,
            gizmo: GizmoMode::new_translate(),
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
        self.image_pass.prepare(&self.queue, &self.camera);
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
            self.image_pass.render(&mut rpass);
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

    pub fn set_image(&mut self, colors: &[[u8; 4]], positions: &[(u32, u32)]) {
        self.image_pass.set_image(&self.device, &self.queue, colors, positions);
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

        // Highlight + gizmo for selected object
        let selected: Option<&dyn Moveable> = match &self.selection {
            Selection::Mesh => Some(&self.mesh_pass as &dyn Moveable),
            Selection::Image => Some(&self.image_pass as &dyn Moveable),
            Selection::None => None,
        };
        if let Some(movable) = selected {
            let model = movable.model_matrix();
            all.extend(movable.gizmo_lines(&model));
            all.extend(self.gizmo.axis_lines(movable, model));
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

        let inv_model = self.mesh_pass.model_matrix().inverse();
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
        self.gizmo.is_dragging()
    }

    pub fn handle_mousedown(&mut self, px: f64, py: f64) -> bool {
        let w = self.config.width as f64;
        let h = self.config.height as f64;
        let ndc_x = (2.0 * px / w - 1.0) as f32;
        let ndc_y = (1.0 - 2.0 * py / h) as f32;
        let (view, proj) = self.get_view_proj();

        let movable: Option<&mut dyn Moveable> = match &mut self.selection {
            Selection::Mesh => Some(&mut self.mesh_pass as &mut dyn Moveable),
            Selection::Image => Some(&mut self.image_pass as &mut dyn Moveable),
            Selection::None => None,
        };
        let Some(movable) = movable else { return false };

        if let Some(axis) = self.gizmo.hit_test(movable, ndc_x, ndc_y, view, proj, 0.04) {
            self.gizmo.start_drag(axis, movable, (px, py));
            return true;
        }
        false
    }

    pub fn handle_mousemove(&mut self, px: f64, py: f64) {
        self.debug_ray_hit = None;

        let w = self.config.width as f64;
        let h = self.config.height as f64;
        let ndc_x = (2.0 * px / w - 1.0) as f32;
        let ndc_y = (1.0 - 2.0 * py / h) as f32;
        let view = self.camera.get_view_matrix();
        let proj = self.camera.get_projection_matrix();

        let movable: Option<&mut dyn Moveable> = match &mut self.selection {
            Selection::Mesh => Some(&mut self.mesh_pass as &mut dyn Moveable),
            Selection::Image => Some(&mut self.image_pass as &mut dyn Moveable),
            Selection::None => None,
        };
        let Some(movable) = movable else {
            self.gizmo.set_hovered(None);
            return;
        };

        if self.gizmo.is_dragging() {
            let (vw, vh) = (self.config.width, self.config.height);
            self.gizmo.apply_drag(movable, px, py, vw, vh, view, proj);
        } else {
            let hover = self.gizmo.hit_test(movable, ndc_x, ndc_y, view, proj, 0.04);
            self.gizmo.set_hovered(hover);
        }
        self.flush_lines();
    }

    pub fn handle_mouseup(&mut self) -> bool {
        self.gizmo.end_drag()
    }


    pub fn toggle_select_mesh(&mut self) {
        self.selection = match self.selection {
            Selection::Mesh => Selection::None,
            _ => Selection::Mesh,
        };
        self.mesh_pass.set_selected(matches!(self.selection, Selection::Mesh));
        self.image_pass.set_selected(false);
        self.flush_lines();
    }

    pub fn mesh_is_selected(&self) -> bool {
        matches!(self.selection, Selection::Mesh)
    }

    pub fn select_mesh_at_screen(&mut self, px: f64, py: f64) -> bool {
        let w = self.config.width as f64;
        let h = self.config.height as f64;
        let ndc_x = (2.0 * px / w - 1.0) as f32;
        let ndc_y = (1.0 - 2.0 * py / h) as f32;
        let inv_vp = self.get_inv_vp();
        let ray = Ray::from_screen(ndc_x, ndc_y, inv_vp);
        let model = self.mesh_pass.model_matrix();
        if self.mesh_pass.ray_intersect_with_model(&ray, &model).is_some() {
            self.selection = Selection::Mesh;
            self.mesh_pass.set_selected(true);
            self.image_pass.set_selected(false);
            self.flush_lines();
            true
        } else {
            false
        }
    }

    pub fn deselect_mesh(&mut self) {
        self.selection = Selection::None;
        self.mesh_pass.set_selected(false);
        self.image_pass.set_selected(false);
        self.flush_lines();
    }

    pub fn select_image_at_screen(&mut self, px: f64, py: f64) -> bool {
        let w = self.config.width as f64;
        let h = self.config.height as f64;
        let ndc_x = (2.0 * px / w - 1.0) as f32;
        let ndc_y = (1.0 - 2.0 * py / h) as f32;
        let inv_vp = self.get_inv_vp();
        let ray = Ray::from_screen(ndc_x, ndc_y, inv_vp);
        let model = self.image_pass.model_matrix();
        if self.image_pass.ray_intersect(&ray, &model).is_some() {
            self.selection = Selection::Image;
            self.mesh_pass.set_selected(false);
            self.image_pass.set_selected(true);
            self.flush_lines();
            true
        } else {
            false
        }
    }

    pub fn toggle_select_image(&mut self) {
        self.selection = match self.selection {
            Selection::Image => Selection::None,
            _ => Selection::Image,
        };
        self.mesh_pass.set_selected(false);
        self.image_pass.set_selected(matches!(self.selection, Selection::Image));
        self.flush_lines();
    }

    pub fn set_gizmo_mode(&mut self, mode: GizmoMode) {
        self.gizmo = mode;
        self.flush_lines();
    }

    pub fn gizmo_name(&self) -> &'static str {
        self.gizmo.name()
    }
}

