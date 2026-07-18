use bytemuck_derive::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::renderer::camera::Camera;
use crate::renderer::moveable::Moveable;
use crate::renderer::pass::RenderPass;
use crate::renderer::ray::Ray;

const SHADER: &str = include_str!("../shaders/image.wgsl");

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ImageVertex {
    position: [f32; 3],
    color: [u8; 4],
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ModelUniform {
    matrix: [f32; 16],
}

pub struct ImagePass {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    vertex_count: u32,
    globals_bind_group: wgpu::BindGroup,
    model_ubo: wgpu::Buffer,
    model_bind_group: wgpu::BindGroup,
    model_matrix: Mat4,
    pixel_count: usize,
    selected: bool,
}

impl ImagePass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        globals_bgl: &wgpu::BindGroupLayout,
        globals_bind_group: wgpu::BindGroup,
    ) -> Self {
        let model_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image_model_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ModelUniform>() as u64),
                },
                count: None,
            }],
        });

        let model_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image_model_ubo"),
            size: std::mem::size_of::<ModelUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("image_model_bind_group"),
            layout: &model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: model_ubo.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("image_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("image_pipeline_layout"),
            bind_group_layouts: &[Some(globals_bgl), Some(&model_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("image_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ImageVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Uint8x4,
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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

        let initial_capacity = 4096;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image_vertex_buffer"),
            size: (initial_capacity * std::mem::size_of::<ImageVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buffer,
            vertex_capacity: initial_capacity,
            vertex_count: 0,
            globals_bind_group,
            model_ubo,
            model_bind_group,
            model_matrix: Mat4::IDENTITY,
            pixel_count: 0,
            selected: false,
        }
    }

    pub fn set_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        colors: &[[u8; 4]],
        positions: &[(u32, u32)],
    ) {
        let count = colors.len().min(positions.len());
        self.pixel_count = count;

        if count == 0 {
            self.vertex_count = 0;
            return;
        }

        let scale = 0.15;
        let max_x = positions.iter().map(|p| p.0).max().unwrap_or(0);
        let max_y = positions.iter().map(|p| p.1).max().unwrap_or(0);
        let cx = max_x as f32 / 2.0;
        let cy = max_y as f32 / 2.0;
        let half = 0.5 * scale;

        let mut verts = Vec::with_capacity(count * 6);
        for i in 0..count {
            let px = (positions[i].0 as f32 - cx) * scale;
            let py = (positions[i].1 as f32 - cy) * scale;
            let c = colors[i];

            verts.push(ImageVertex { position: [px - half, py - half, 0.0], color: c });
            verts.push(ImageVertex { position: [px + half, py - half, 0.0], color: c });
            verts.push(ImageVertex { position: [px - half, py + half, 0.0], color: c });
            verts.push(ImageVertex { position: [px + half, py - half, 0.0], color: c });
            verts.push(ImageVertex { position: [px + half, py + half, 0.0], color: c });
            verts.push(ImageVertex { position: [px - half, py + half, 0.0], color: c });
        }

        let needed = verts.len();
        if needed > self.vertex_capacity {
            self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("image_vertex_buffer"),
                size: (needed * std::mem::size_of::<ImageVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = needed;
        }
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&verts));
        self.vertex_count = needed as u32;
    }

    fn image_bounds(&self) -> (f32, f32) {
        let side = (self.pixel_count as f32).sqrt().max(1.0);
        (side * 0.15, side * 0.15)
    }
}

impl RenderPass for ImagePass {
    fn prepare(&mut self, queue: &wgpu::Queue, _camera: &Camera) {
        let uniform = ModelUniform { matrix: self.model_matrix.to_cols_array() };
        queue.write_buffer(&self.model_ubo, 0, bytemuck::cast_slice(&[uniform]));
    }

    fn render<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>) {
        if self.vertex_count == 0 {
            return;
        }
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.globals_bind_group, &[]);
        rpass.set_bind_group(1, &self.model_bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.draw(0..self.vertex_count, 0..1);
    }
}

impl Moveable for ImagePass {
    fn model_matrix(&self) -> Mat4 {
        self.model_matrix
    }

    fn set_model_matrix(&mut self, m: Mat4) {
        self.model_matrix = m;
    }

    fn is_selected(&self) -> bool {
        self.selected
    }

    fn set_selected(&mut self, selected: bool) {
        self.selected = selected;
    }

    fn center(&self) -> Vec3 {
        Vec3::ZERO
    }

    fn bounding_size(&self) -> f32 {
        let (w, h) = self.image_bounds();
        w.max(h)
    }

    fn ray_intersect(&self, ray: &Ray, model: &Mat4) -> Option<Vec3> {
        let inv_model = model.inverse();
        let local_ray = Ray {
            origin: inv_model.transform_point3(ray.origin),
            direction: inv_model.transform_vector3(ray.direction),
        };
        let (w, h) = self.image_bounds();
        let half_w = w / 2.0;
        let half_h = h / 2.0;
        local_ray
            .intersect_aabb(Vec3::new(-half_w, -half_h, 0.0), Vec3::new(half_w, half_h, 0.0))
            .map(|t| {
                let local_hit = local_ray.origin + local_ray.direction * t;
                model.transform_point3(local_hit)
            })
    }

    fn gizmo_color(&self) -> Vec3 {
        Vec3::new(1.0, 0.5, 0.1)
    }

    fn gizmo_lines(&self, model: &Mat4) -> Vec<(Vec3, Vec3, Vec3)> {
        let (w, h) = self.image_bounds();
        let half_w = w / 2.0;
        let half_h = h / 2.0;
        let local = [
            Vec3::new(-half_w, -half_h, 0.0),
            Vec3::new(half_w, -half_h, 0.0),
            Vec3::new(half_w, half_h, 0.0),
            Vec3::new(-half_w, half_h, 0.0),
        ];
        let world: Vec<Vec3> = local.iter().map(|p| model.transform_point3(*p)).collect();
        let color = Vec3::new(1.0, 0.5, 0.1);
        vec![
            (world[0], world[1], color),
            (world[1], world[2], color),
            (world[2], world[3], color),
            (world[3], world[0], color),
        ]
    }
}
