use wgpu::BindingType;

/// A uniform-buffer binding sized for `T`.
pub fn uniform<T: bytemuck::Pod>() -> BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<T>() as u64),
    }
}

/// A read-only storage-buffer binding (no minimum size — size is open-ended).
pub fn storage_ro() -> BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

/// A read-write storage-buffer binding.
pub fn storage_rw() -> BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

/// A sampled 2D texture with float samples.
pub fn texture2d() -> BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

/// A filtering sampler.
pub fn sampler() -> BindingType {
    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
}
