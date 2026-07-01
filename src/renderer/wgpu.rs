use bytemuck_derive::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use web_sys::HtmlCanvasElement;
use wgpu::SurfaceTarget;
use wgpu::util::DeviceExt;

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.0, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

const WGSL_SHADER: &str = include_str!("shaders/triangle.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Model {
    model_matrix: [f32; 16],
}

impl Model {
    fn new(model_matrix: Mat4) -> Self {
        Self {
            model_matrix: model_matrix.to_cols_array(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Globals {
    view: [f32; 16],
    projection: [f32; 16],
}

impl Globals {
    fn new(view: Mat4, projection: Mat4) -> Self {
        Self {
            view: view.to_cols_array(),
            projection: projection.to_cols_array(),
        }
    }
}

const ORBIT_SENSITIVITY: f32 = 0.3;

struct Camera {
    aspect_ratio: f32,
    distance: f32,
    target: Vec3,
    up: Vec3,
    yaw: f32,
    pitch: f32,
    fov_y: f32,
}

impl Camera {
    fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            aspect_ratio: screen_width as f32 / screen_height as f32,
            distance: 5.0,
            target: Vec3::ZERO,
            up: Vec3::Y,
            yaw: -90.0_f32,
            pitch: 0.0_f32,
            fov_y: 45.0_f32,
        }
    }

    fn front(&self) -> Vec3 {
        Vec3::new(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        )
    }

    fn get_view_matrix(&self) -> Mat4 {
        let position = self.target + self.distance * self.front();
        Mat4::look_at_rh(position, self.target, self.up)
    }

    fn get_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y.to_radians(), self.aspect_ratio, 0.1, 100.0)
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.aspect_ratio = width as f32 / height as f32;
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * ORBIT_SENSITIVITY;
        self.pitch += dy * ORBIT_SENSITIVITY;

        if self.pitch > 89.0 {
            self.pitch = 89.0;
        }
        if self.pitch < -89.0 {
            self.pitch = -89.0;
        }
    }

    fn zoom(&mut self, delta: f32) {
        self.distance -= delta;
        if self.distance < 0.1 {
            self.distance = 0.1;
        }
        if self.distance > 100.0 {
            self.distance = 100.0;
        }
    }
}

#[allow(dead_code)]
pub struct WgpuRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    backend: wgpu::Backend,
    globals_ubo: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    model_ubo: wgpu::Buffer,
    model_bind_group: wgpu::BindGroup,
    camera: Camera,
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
        let format = caps
            .formats
            .first()
            .copied()
            .ok_or("no surface format".to_string())?;

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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL_SHADER)),
        });

        let global_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let model_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout::model"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Model>() as u64),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&global_bgl), Some(&model_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let globals_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ubo::globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let model_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ubo::model"),
            size: std::mem::size_of::<Model>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group::globals"),
            layout: &global_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_ubo.as_entire_binding(),
            }],
        });

        let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group::model"),
            layout: &model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: model_ubo.as_entire_binding(),
            }],
        });

        let camera = Camera::new(width, height);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            num_vertices: VERTICES.len() as u32,
            backend,
            globals_ubo,
            globals_bind_group,
            model_ubo,
            model_bind_group,
            camera,
        })
    }

    pub fn render(&self) {
        let model = Model::new(Mat4::IDENTITY);
        self.queue
            .write_buffer(&self.model_ubo, 0, bytemuck::cast_slice(&[model]));

        let globals = Globals::new(
            self.camera.get_view_matrix(),
            self.camera.get_projection_matrix(),
        );
        self.queue
            .write_buffer(&self.globals_ubo, 0, bytemuck::cast_slice(&[globals]));

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Validation => {
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.globals_bind_group, &[]);
            rpass.set_bind_group(1, &self.model_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.draw(0..self.num_vertices, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);

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
}
