// ── Compute kernel ──────────────────────────────────────────────────────

pub struct ComputeKernel {
    pub pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
}

impl ComputeKernel {
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        wgsl: &str,
        entry: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label}_shader")),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_bgl")),
            entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}_layout")),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}_pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bgl }
    }

    pub fn bind(&self, device: &wgpu::Device, resources: &[wgpu::BindGroupEntry]) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group::compute"),
            layout: &self.bgl,
            entries: resources,
        })
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg: &wgpu::BindGroup,
        groups: (u32, u32, u32),
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bg, &[]);
        cpass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }
}

// ── Render-pipeline builder ─────────────────────────────────────────────

pub struct RenderKernelBuilder<'a> {
    device: &'a wgpu::Device,
    label: &'a str,
    shader: &'a wgpu::ShaderModule,
    bgls: &'a [&'a wgpu::BindGroupLayout],
    vertex_attributes: Vec<wgpu::VertexAttribute>,
    vertex_stride: wgpu::BufferAddress,
    vertex_step: wgpu::VertexStepMode,
    has_vertex_layout: bool,
    vs_entry: &'a str,
    fs_entry: &'a str,
    color_format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
    depth_format: Option<wgpu::TextureFormat>,
    depth_write: Option<bool>,
    depth_compare: Option<wgpu::CompareFunction>,
    cull_mode: Option<wgpu::Face>,
    topology: wgpu::PrimitiveTopology,
}

impl<'a> RenderKernelBuilder<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        label: &'a str,
        shader: &'a wgpu::ShaderModule,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            device,
            label,
            shader,
            bgls: &[],
            vertex_attributes: Vec::new(),
            vertex_stride: 0,
            vertex_step: wgpu::VertexStepMode::Vertex,
            has_vertex_layout: false,
            vs_entry: "vs_main",
            fs_entry: "fs_main",
            color_format,
            blend: wgpu::BlendState::REPLACE,
            depth_format: None,
            depth_write: None,
            depth_compare: None,
            cull_mode: None,
            topology: wgpu::PrimitiveTopology::TriangleList,
        }
    }

    pub fn bind_group_layouts(mut self, bgls: &'a [&'a wgpu::BindGroupLayout]) -> Self {
        self.bgls = bgls;
        self
    }

    pub fn vertex_layout(mut self, layout: wgpu::VertexBufferLayout<'_>) -> Self {
        self.has_vertex_layout = true;
        self.vertex_stride = layout.array_stride;
        self.vertex_step = layout.step_mode;
        self.vertex_attributes = layout.attributes.to_vec();
        self
    }

    #[allow(dead_code)]
    pub fn entry_points(mut self, vs: &'a str, fs: &'a str) -> Self {
        self.vs_entry = vs;
        self.fs_entry = fs;
        self
    }

    pub fn depth(
        mut self,
        format: wgpu::TextureFormat,
        write: bool,
        cmp: wgpu::CompareFunction,
    ) -> Self {
        self.depth_format = Some(format);
        self.depth_write = Some(write);
        self.depth_compare = Some(cmp);
        self
    }

    #[allow(dead_code)]
    pub fn cull_mode(mut self, mode: Option<wgpu::Face>) -> Self {
        self.cull_mode = mode;
        self
    }

    pub fn blend(mut self, blend: wgpu::BlendState) -> Self {
        self.blend = blend;
        self
    }

    pub fn topology(mut self, top: wgpu::PrimitiveTopology) -> Self {
        self.topology = top;
        self
    }

    pub fn build(self) -> wgpu::RenderPipeline {
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_layout", self.label)),
            bind_group_layouts: &self
                .bgls
                .iter()
                .map(|bgl| Some(*bgl))
                .collect::<Vec<_>>(),
            immediate_size: 0,
        });

        let buffers: &[wgpu::VertexBufferLayout] = if self.has_vertex_layout {
            &[wgpu::VertexBufferLayout {
                array_stride: self.vertex_stride,
                step_mode: self.vertex_step,
                attributes: &self.vertex_attributes,
            }]
        } else {
            &[]
        };

        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{}_pipeline", self.label)),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: self.shader,
                entry_point: Some(self.vs_entry),
                compilation_options: Default::default(),
                buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: self.shader,
                entry_point: Some(self.fs_entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.color_format,
                    blend: Some(self.blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: self.topology,
                cull_mode: self.cull_mode,
                ..Default::default()
            },
            depth_stencil: self.depth_format.map(|fmt| wgpu::DepthStencilState {
                format: fmt,
                depth_write_enabled: self.depth_write,
                depth_compare: self.depth_compare,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }
}
