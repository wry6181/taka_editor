use std::marker::PhantomData;
use wgpu::util::DeviceExt;

/// Typed wrapper around a wgpu buffer — the only way to construct one is through
/// the factory methods below so the usage flags are always correct for the role.
pub struct GpuBuffer<T> {
    raw: wgpu::Buffer,
    _phantom: PhantomData<T>,
}

#[allow(dead_code)]
impl<T: bytemuck::Pod> GpuBuffer<T> {
    /// Create a device-local STORAGE buffer initialised from `data`.
    pub fn storage_init(device: &wgpu::Device, label: &str, data: &[T]) -> Self {
        let raw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Self { raw, _phantom: PhantomData }
    }

    /// Create a host-visible UNIFORM buffer (one element, zeroed).
    pub fn uniform(device: &wgpu::Device, label: &str) -> Self {
        let size = std::mem::size_of::<T>() as u64;
        let raw = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { raw, _phantom: PhantomData }
    }

    /// Create a STORAGE buffer that can be read and written by the GPU, and used
    /// as a COPY_SRC for readback.
    pub fn storage_rw(device: &wgpu::Device, label: &str, len: usize) -> Self {
        let size = (std::mem::size_of::<T>() * len) as u64;
        let raw = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { raw, _phantom: PhantomData }
    }

    /// Write a single element into the buffer (for uniforms / UBO updates).
    pub fn write(&self, queue: &wgpu::Queue, data: &T) {
        queue.write_buffer(&self.raw, 0, bytemuck::cast_slice(std::slice::from_ref(data)));
    }

    pub fn write_slice(&self, queue: &wgpu::Queue, offset: u64, data: &[T]) {
        queue.write_buffer(&self.raw, offset, bytemuck::cast_slice(data));
    }

    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        self.raw.as_entire_binding()
    }

    pub fn raw(&self) -> &wgpu::Buffer {
        &self.raw
    }
}

// ── Readback ────────────────────────────────────────────────────────────

/// A staging buffer for non-blocking GPU→CPU readback.
///
/// Usage pattern (same as the existing raycast code):
///
/// ```ignore
/// staging.copy_from(encoder, &gpu_buffer);
/// queue.submit(Some(encoder.finish()));
/// staging.request(|value| { /* called when mapping completes */ });
/// // in a *separate* per-frame poll call:
/// device.poll(PollType::Poll);
/// ```
pub struct Readback<T> {
    raw: wgpu::Buffer,
    size: u64,
    _phantom: PhantomData<T>,
}

impl<T: bytemuck::Pod + bytemuck::Zeroable + Copy + 'static> Readback<T> {
    pub fn new(device: &wgpu::Device, label: &str) -> Self {
        let size = std::mem::size_of::<T>() as u64;
        let raw = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { raw, size, _phantom: PhantomData }
    }

    /// Record a buffer-to-buffer copy from `src` into this staging buffer.
    pub fn copy_from(&self, encoder: &mut wgpu::CommandEncoder, src: &GpuBuffer<T>) {
        encoder.copy_buffer_to_buffer(&src.raw, 0, &self.raw, 0, self.size);
    }

    /// Non-blocking `map_async` — the callback receives the read-back value (or
    /// a zeroed default if the mapping fails).  Does NOT call `device.poll` —
    /// the caller must do that in a separate per-frame tick.
    pub fn request(&self, on_ready: impl FnOnce(T) + Send + 'static) {
        let staged = self.raw.clone();
        self.raw.map_async(wgpu::MapMode::Read, .., move |result| {
            let value = if result.is_ok() {
                let data = staged.get_mapped_range(..);
                let v = *bytemuck::from_bytes(&data);
                drop(data);
                staged.unmap();
                v
            } else {
                bytemuck::Zeroable::zeroed()
            };
            on_ready(value);
        });
    }
}
