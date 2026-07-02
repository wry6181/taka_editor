use bytemuck_derive::{Pod, Zeroable};
use glam::Vec3;

use crate::renderer::camera::Camera;
use crate::renderer::pass::RenderPass;

const SHADER: &str = include_str!("../shaders/line.wgsl");

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct LineVertex {
    position: [f32; 3],
    color: [f32; 3],
}

pub struct LinePass {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    vertex_count: u32,
    globals_bind_group: wgpu::BindGroup,
}

impl LinePass {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        globals_bgl: &wgpu::BindGroupLayout,
        globals_bind_group: wgpu::BindGroup,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("line_pipeline_layout"),
            bind_group_layouts: &[Some(globals_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
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
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
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
            label: Some("line_vertex_buffer"),
            size: (initial_capacity * std::mem::size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buffer,
            vertex_capacity: initial_capacity,
            vertex_count: 0,
            globals_bind_group,
        }
    }

    pub fn set_lines(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, lines: &[(Vec3, Vec3, Vec3)]) {
        let mut verts = Vec::with_capacity(lines.len() * 2);
        for (start, end, color) in lines {
            verts.push(LineVertex { position: (*start).into(), color: (*color).into() });
            verts.push(LineVertex { position: (*end).into(), color: (*color).into() });
        }
        if verts.len() > self.vertex_capacity {
            self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("line_vertex_buffer"),
                size: (verts.len() * std::mem::size_of::<LineVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = verts.len();
        }
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&verts));
        self.vertex_count = verts.len() as u32;
    }
}

impl RenderPass for LinePass {
    fn prepare(&mut self, _queue: &wgpu::Queue, _camera: &Camera) {
    }

    fn render<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>) {
        if self.vertex_count == 0 {
            return;
        }
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.globals_bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.draw(0..self.vertex_count, 0..1);
    }
}
