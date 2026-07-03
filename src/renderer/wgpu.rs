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
use super::remesh::Remesh;

const REMESH_FILL_SOURCE: &str = r#"
struct Globals {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> globals: Globals;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return globals.projection * globals.view * vec4(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4(0.2, 0.8, 0.6, 0.8);
}
"#;

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
    remesh: Option<Remesh>,
    remesh_active: bool,
    remesh_drag_axis: Option<usize>,
    remesh_drag_start_mouse: Option<(f64, f64)>,
    remesh_drag_start_pos: Option<Vec3>,
    show_mesh: bool,
    remesh_fill_pipeline: wgpu::RenderPipeline,
    remesh_fill_buffer: wgpu::Buffer,
    remesh_fill_capacity: u32,
    remesh_fill_count: u32,
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

        let remesh_fill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("remesh_fill"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(REMESH_FILL_SOURCE)),
        });
        let remesh_fill_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("remesh_fill_layout"),
                bind_group_layouts: &[Some(&globals_bgl)],
                immediate_size: 0,
            });
        let remesh_fill_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("remesh_fill"),
                layout: Some(&remesh_fill_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &remesh_fill_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &remesh_fill_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let remesh_fill_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("remesh_fill"),
            size: 4096 * 12,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
            remesh: None,
            remesh_active: false,
            remesh_drag_axis: None,
            remesh_drag_start_mouse: None,
            remesh_drag_start_pos: None,
            show_mesh: true,
            remesh_fill_pipeline,
            remesh_fill_buffer,
            remesh_fill_capacity: 4096,
            remesh_fill_count: 0,
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
            if self.remesh_fill_count > 0 {
                rpass.set_pipeline(&self.remesh_fill_pipeline);
                rpass.set_bind_group(0, &self.globals_bind_group, &[]);
                rpass.set_vertex_buffer(0, self.remesh_fill_buffer.slice(..));
                rpass.draw(0..self.remesh_fill_count, 0..1);
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

    fn rebuild_remesh_buffer(&mut self) {
        let Some(ref remesh) = self.remesh else {
            self.remesh_fill_count = 0;
            return;
        };
        if !self.remesh_active || remesh.triangles.is_empty() {
            self.remesh_fill_count = 0;
            return;
        }
        let vert_count = (remesh.triangles.len() * 3) as u32;
        let needed = vert_count.max(1) as u64 * 12;
        if needed > self.remesh_fill_capacity as u64 * 12 {
            self.remesh_fill_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("remesh_fill"),
                size: needed,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.remesh_fill_capacity = vert_count;
        }
        let mut verts: Vec<[f32; 3]> = Vec::with_capacity(vert_count as usize);
        for &(a, b, c) in &remesh.triangles {
            verts.push(remesh.points[a as usize].to_array());
            verts.push(remesh.points[b as usize].to_array());
            verts.push(remesh.points[c as usize].to_array());
        }
        self.queue.write_buffer(&self.remesh_fill_buffer, 0, bytemuck::cast_slice(&verts));
        self.remesh_fill_count = vert_count;
    }

    fn flush_lines(&mut self) {
        let mut all = self.axis_lines.clone();

        if !self.remesh_active {
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
        }

        if let Some(ref remesh) = self.remesh {
            if self.remesh_active {
                for &(a, b, c) in &remesh.triangles {
                    let pa = remesh.points[a as usize];
                    let pb = remesh.points[b as usize];
                    let pc = remesh.points[c as usize];
                    all.push((pa, pb, Vec3::new(0.2, 1.0, 0.2)));
                    all.push((pb, pc, Vec3::new(0.2, 1.0, 0.2)));
                    all.push((pc, pa, Vec3::new(0.2, 1.0, 0.2)));
                }
                if let Some(idx) = remesh.selected {
                    let pos = remesh.points[idx];
                    let axis_len = 1.5;
                    all.push((pos, pos + Vec3::X * axis_len, Vec3::new(1.0, 0.0, 0.0)));
                    all.push((pos, pos + Vec3::Y * axis_len, Vec3::new(0.0, 1.0, 0.0)));
                    all.push((pos, pos + Vec3::Z * axis_len, Vec3::new(0.0, 0.0, 1.0)));
                    all.extend(Self::wireframe_sphere(pos + Vec3::X * axis_len, 0.08, Vec3::new(1.0, 0.0, 0.0), 8));
                    all.extend(Self::wireframe_sphere(pos + Vec3::Y * axis_len, 0.08, Vec3::new(0.0, 1.0, 0.0), 8));
                    all.extend(Self::wireframe_sphere(pos + Vec3::Z * axis_len, 0.08, Vec3::new(0.0, 0.0, 1.0), 8));
                }
                for (i, &pos) in remesh.points.iter().enumerate() {
                    let c = if remesh.selected == Some(i) { Vec3::new(1.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.5, 0.0) };
                    all.extend(Self::wireframe_sphere(pos, 0.1, c, 12));
                }
            }
        }

        self.line_pass.set_lines(&self.device, &self.queue, &all);
        self.rebuild_remesh_buffer();
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

        let staging = self.mesh_pass.dispatch_raycast(&self.device, &self.queue, &ray);
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
                if self.remesh_active {
                    if outcome.hit {
                        if let (Some(origin), Some(dir)) = (ray_origin, ray_dir) {
                            if let Some(ref mut remesh) = self.remesh {
                                remesh.add_point(origin + dir * outcome.t);
                            }
                        }
                    }
                    self.debug_ray_hit = None;
                } else {
                    if outcome.hit {
                        if let (Some(origin), Some(dir)) = (ray_origin, ray_dir) {
                            self.debug_ray_hit = Some(origin + dir * outcome.t);
                        }
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

    pub fn remesh_toggle(&mut self) {
        self.remesh_active = !self.remesh_active;
        if self.remesh_active && self.remesh.is_none() {
            self.remesh = Some(Remesh::new());
        }
        if !self.remesh_active {
            self.remesh_drag_axis = None;
            self.remesh_drag_start_mouse = None;
            self.remesh_drag_start_pos = None;
        }
        self.flush_lines();
    }

    pub fn remesh_toggle_mesh(&mut self) {
        self.show_mesh = !self.show_mesh;
    }

    pub fn remesh_show_mesh(&self) -> bool {
        self.show_mesh
    }

    pub fn remesh_is_active(&self) -> bool {
        self.remesh_active
    }

    pub fn remesh_is_dragging(&self) -> bool {
        self.remesh_drag_axis.is_some()
    }

    pub fn remesh_handle_mousedown(&mut self, px: f64, py: f64) -> bool {
        if !self.remesh_active {
            return false;
        }
        let Some(ref remesh) = self.remesh else { return false };
        let Some(sel) = remesh.selected else { return false };
        let w = self.config.width as f64;
        let h = self.config.height as f64;
        let ndc_x = (2.0 * px / w - 1.0) as f32;
        let ndc_y = (1.0 - 2.0 * py / h) as f32;
        let (view, proj) = self.get_view_proj();
        let axis = remesh.gizmo_axis_hit_screen(ndc_x, ndc_y, view, proj, 0.04);
        let on_point = remesh.hit_point_screen(ndc_x, ndc_y, view, proj, 0.04)
            .map_or(false, |i| Some(i) == remesh.selected);
        if axis.is_none() && !on_point {
            return false;
        }
        self.remesh_drag_axis = axis;
        self.remesh_drag_start_mouse = Some((px, py));
        self.remesh_drag_start_pos = Some(remesh.points[sel]);
        true
    }

    pub fn remesh_handle_mousemove(&mut self, px: f64, py: f64) {
        let Some(axis) = self.remesh_drag_axis else { return };
        let Some((sx, sy)) = self.remesh_drag_start_mouse else { return };
        let Some(start_pos) = self.remesh_drag_start_pos else { return };
        let Some(idx) = self.remesh.as_ref().and_then(|r| r.selected) else { return };

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

        if axis < 3 {
            let axis_dir = match axis {
                0 => Vec3::X,
                1 => Vec3::Y,
                _ => Vec3::Z,
            };
            let (svx, svy) = screen_vec(axis_dir);
            let len_sq = svx * svx + svy * svy;
            if len_sq < 1.0 { return; }
            let world_offset = ((dx * svx + dy * svy) / len_sq) as f32;
            if let Some(ref mut remesh) = self.remesh {
                remesh.points[idx] = start_pos + axis_dir * world_offset;
                self.flush_lines();
            }
        } else {
            let right = view.x_axis.truncate().normalize();
            let up = view.y_axis.truncate().normalize();
            let (rx, ry) = screen_vec(right);
            let (ux, uy) = screen_vec(up);
            let det = rx * uy - ry * ux;
            if det.abs() < 1.0 { return; }
            let a = (dx * uy - dy * ux) / det;
            let b = (ry * dx - rx * dy) / det;
            if let Some(ref mut remesh) = self.remesh {
                remesh.points[idx] = start_pos + right * a as f32 + up * b as f32;
                self.flush_lines();
            }
        }
    }

    pub fn remesh_handle_mouseup(&mut self) {
        self.remesh_drag_axis = None;
        self.remesh_drag_start_mouse = None;
        self.remesh_drag_start_pos = None;
    }

    pub fn remesh_handle_click(&mut self, px: f64, py: f64) -> bool {
        if !self.remesh_active {
            return false;
        }
        let w = self.config.width as f64;
        let h = self.config.height as f64;
        let ndc_x = (2.0 * px / w - 1.0) as f32;
        let ndc_y = (1.0 - 2.0 * py / h) as f32;

        // 1. Try selecting a control point
        if let Some(ref remesh) = self.remesh {
            if !remesh.points.is_empty() {
                let (view, proj) = self.get_view_proj();
                let hit_idx = remesh.hit_point_screen(ndc_x, ndc_y, view, proj, 0.03);
                if let Some(idx) = hit_idx {
                    let remesh = self.remesh.as_mut().unwrap();
                    remesh.selected = if remesh.selected == Some(idx) { None } else { Some(idx) };
                    self.flush_lines();
                    return true;
                }
            }
        }

        // 2. Deselect and add a new point via CPU raycast
        if let Some(ref mut remesh) = self.remesh {
            remesh.selected = None;
        }

        let inv_vp = self.get_inv_vp();
        let ray = Ray::from_screen(ndc_x, ndc_y, inv_vp);
        if let Some(hit) = self.mesh_pass.ray_intersect(&ray) {
            if let Some(ref mut remesh) = self.remesh {
                remesh.add_point(hit);
            }
        }
        self.flush_lines();
        true
    }
}
