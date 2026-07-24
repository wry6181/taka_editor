use std::collections::HashMap;

use bytemuck_derive::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::renderer::camera::Camera;
use crate::renderer::gpu::buffer::{GpuBuffer, Readback};
use crate::renderer::gpu::kernel::ComputeKernel;
use crate::renderer::moveable::Moveable;
use crate::renderer::pass::RenderPass;
use crate::renderer::ray::Ray;
use crate::{bgl, bind_group};

const SHADER: &str = include_str!("../shaders/editor.wgsl");
const GLB_DATA: &[u8] = include_bytes!("../../../data/gameboy.glb");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coord: [f32; 2],
    normal: [f32; 3],
    tangent: [f32; 4],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuRayUniform {
    origin: [f32; 4],
    direction: [f32; 4],
}

impl From<&Ray> for GpuRayUniform {
    fn from(ray: &Ray) -> Self {
        Self {
            origin: [ray.origin.x, ray.origin.y, ray.origin.z, 0.0],
            direction: [ray.direction.x, ray.direction.y, ray.direction.z, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuHitResult {
    pub hit: u32,
    pub t_bits: u32,
}

#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Model {
    model_matrix: [f32; 16],
}
impl Model {
    fn new(model_matrix: Mat4) -> Self {
        Self { model_matrix: model_matrix.to_cols_array() }
    }
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct MaterialFactors {
    base_color_factor: [f32; 4],
    metallic_factor: f32,
    roughness_factor: f32,
    _padding: [f32; 2],
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Light {
    direction: [f32; 3],
    _padding: f32,
}
impl Light {
    fn new(direction: Vec3) -> Self {
        Self { direction: direction.into(), _padding: 0.0 }
    }
}

struct Defaults {
    base_color: wgpu::TextureView,
    normal: wgpu::TextureView,
    white_linear: wgpu::TextureView,
}

struct Primitive {
    index_start: u32,
    index_count: u32,
    base_vertex: i32,
    bind_group: wgpu::BindGroup,
}

fn create_solid_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color: [u8; 4],
    srgb: bool,
) -> wgpu::TextureView {
    let format = if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("default_texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &color,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn to_rgba8(img: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;
    match img.format {
        Format::R8G8B8A8 => img.pixels.clone(),
        Format::R8G8B8 => img.pixels.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        Format::R8 => img.pixels.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        Format::R8G8 => img.pixels.chunks_exact(2).flat_map(|c| [c[0], c[1], 0, 255]).collect(),
        other => {
            web_sys::console::log_1(
                &format!("unsupported glTF image format {:?}, substituting white", other).into(),
            );
            vec![255u8; (img.width * img.height * 4) as usize]
        }
    }
}

fn get_or_create_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut HashMap<(usize, bool), wgpu::TextureView>,
    images: &[gltf::image::Data],
    image_index: usize,
    srgb: bool,
) -> wgpu::TextureView {
    if let Some(view) = cache.get(&(image_index, srgb)) {
        return view.clone();
    }
    let img = &images[image_index];
    let rgba = to_rgba8(img);
    let format = if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("glb_texture"),
            size: wgpu::Extent3d { width: img.width, height: img.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &rgba,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    cache.insert((image_index, srgb), view.clone());
    view
}

fn compute_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let mut tan1 = vec![Vec3::ZERO; positions.len()];
    let mut tan2 = vec![Vec3::ZERO; positions.len()];

    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vec3::from(positions[i0]);
        let p1 = Vec3::from(positions[i1]);
        let p2 = Vec3::from(positions[i2]);
        let uv0 = uvs[i0];
        let uv1 = uvs[i1];
        let uv2 = uvs[i2];

        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let du1 = uv1[0] - uv0[0];
        let dv1 = uv1[1] - uv0[1];
        let du2 = uv2[0] - uv0[0];
        let dv2 = uv2[1] - uv0[1];

        let det = du1 * dv2 - du2 * dv1;
        if det.abs() < 1e-8 {
            continue;
        }
        let r = 1.0 / det;
        let tangent = (edge1 * dv2 - edge2 * dv1) * r;
        let bitangent = (edge2 * du1 - edge1 * du2) * r;

        for &i in &[i0, i1, i2] {
            tan1[i] += tangent;
            tan2[i] += bitangent;
        }
    }

    (0..positions.len())
        .map(|i| {
            let n = Vec3::from(normals[i]);
            let t = tan1[i];
            let t_ortho = (t - n * n.dot(t)).normalize_or_zero();
            let t_final = if t_ortho.length_squared() > 0.0 { t_ortho } else { Vec3::X };
            let w = if n.cross(t_final).dot(tan2[i]) < 0.0 { -1.0 } else { 1.0 };
            [t_final.x, t_final.y, t_final.z, w]
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn visit_node(
    node: &gltf::Node,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material_bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    images: &[gltf::image::Data],
    texture_cache: &mut HashMap<(usize, bool), wgpu::TextureView>,
    defaults: &Defaults,
    all_vertices: &mut Vec<Vertex>,
    all_indices: &mut Vec<u32>,
    primitives: &mut Vec<Primitive>,
) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_transform * local;
    let normal_matrix = world.inverse().transpose();

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|b| Some(&buffers[b.index()]));

            let positions: Vec<[f32; 3]> =
                reader.read_positions().expect("mesh has no positions").collect();
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|it| it.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|it| it.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
            let colors: Vec<[f32; 3]> = reader
                .read_colors(0)
                .map(|it| it.into_rgba_f32().map(|c| [c[0], c[1], c[2]]).collect())
                .unwrap_or_else(|| vec![[1.0, 1.0, 1.0]; positions.len()]);

            let indices: Vec<u32> = match reader.read_indices() {
                Some(it) => it.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            let tangents: Vec<[f32; 4]> = match reader.read_tangents() {
                Some(it) => it.collect(),
                None => compute_tangents(&positions, &normals, &uvs, &indices),
            };

            let base_vertex = all_vertices.len() as u32;
            for i in 0..positions.len() {
                let world_pos = world.transform_point3(Vec3::from(positions[i]));
                let world_normal =
                    normal_matrix.transform_vector3(Vec3::from(normals[i])).normalize_or_zero();
                let t = tangents[i];
                let world_tangent = normal_matrix
                    .transform_vector3(Vec3::new(t[0], t[1], t[2]))
                    .normalize_or_zero();

                all_vertices.push(Vertex {
                    position: world_pos.into(),
                    tex_coord: uvs[i],
                    normal: world_normal.into(),
                    tangent: [world_tangent.x, world_tangent.y, world_tangent.z, t[3]],
                    color: colors[i],
                });
            }

            let index_start = all_indices.len() as u32;
            all_indices.extend_from_slice(&indices);
            let index_count = indices.len() as u32;

            let material = primitive.material();
            let pbr = material.pbr_metallic_roughness();

            let base_color_view = pbr
                .base_color_texture()
                .map(|info| {
                    get_or_create_texture(
                        device, queue, texture_cache, images, info.texture().source().index(), true,
                    )
                })
                .unwrap_or_else(|| defaults.base_color.clone());
            let normal_view = material
                .normal_texture()
                .map(|info| {
                    get_or_create_texture(
                        device, queue, texture_cache, images, info.texture().source().index(), false,
                    )
                })
                .unwrap_or_else(|| defaults.normal.clone());
            let mr_view = pbr
                .metallic_roughness_texture()
                .map(|info| {
                    get_or_create_texture(
                        device, queue, texture_cache, images, info.texture().source().index(), false,
                    )
                })
                .unwrap_or_else(|| defaults.white_linear.clone());

            let factors = MaterialFactors {
                base_color_factor: pbr.base_color_factor(),
                metallic_factor: pbr.metallic_factor(),
                roughness_factor: pbr.roughness_factor(),
                _padding: [0.0; 2],
            };
            let factors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ubo::material_factors"),
                contents: bytemuck::cast_slice(&[factors]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind_group::material"),
                layout: material_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: factors_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&base_color_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&normal_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&mr_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            });

            primitives.push(Primitive {
                index_start,
                index_count,
                base_vertex: base_vertex as i32,
                bind_group,
            });
        }
    }

    for child in node.children() {
        visit_node(
            &child, world, buffers, device, queue, material_bgl, sampler, images,
            texture_cache, defaults, all_vertices, all_indices, primitives,
        );
    }
}

fn load_glb_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material_bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
) -> (Vec<Vertex>, Vec<u32>, Vec<Primitive>) {
    let (document, buffers, images) =
        gltf::import_slice(GLB_DATA).expect("failed to parse GLB");

    let defaults = Defaults {
        base_color: create_solid_texture(device, queue, [255, 255, 255, 255], true),
        normal: create_solid_texture(device, queue, [128, 128, 255, 255], false),
        white_linear: create_solid_texture(device, queue, [255, 255, 255, 255], false),
    };

    let mut texture_cache: HashMap<(usize, bool), wgpu::TextureView> = HashMap::new();
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();
    let mut primitives = Vec::new();

    let scene = document
        .default_scene()
        .unwrap_or_else(|| document.scenes().next().expect("glb has no scenes"));

    for node in scene.nodes() {
        visit_node(
            &node, Mat4::IDENTITY, &buffers, device, queue, material_bgl, sampler, &images,
            &mut texture_cache, &defaults, &mut all_vertices, &mut all_indices, &mut primitives,
        );
    }

    (all_vertices, all_indices, primitives)
}

pub struct MeshPass {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    primitives: Vec<Primitive>,
    globals_bind_group: wgpu::BindGroup,
    model_ubo: wgpu::Buffer,
    model_bind_group: wgpu::BindGroup,
    light_ubo: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_direction: Vec3,
    mesh_positions: Vec<[f32; 3]>,
    mesh_indices: Vec<u32>,
    position_storage: GpuBuffer<[f32; 4]>,
    index_storage: GpuBuffer<u32>,
    ray_ubo: GpuBuffer<GpuRayUniform>,
    result_ssbo: GpuBuffer<GpuHitResult>,
    staging: Readback<GpuHitResult>,
    raycast_bind_group: wgpu::BindGroup,
    raycast_kernel: ComputeKernel,
    model_matrix: Mat4,
    selected: bool,
}

impl MeshPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        globals_bgl: &wgpu::BindGroupLayout,
        globals_bind_group: wgpu::BindGroup,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });

        use crate::renderer::gpu::binding_types;
        use wgpu::ShaderStages;

        let model_bgl = bgl!(device, "bind_group_layout::model", [
            0 => ShaderStages::VERTEX, binding_types::uniform::<Model>(),
        ]);

        let light_bgl = bgl!(device, "bind_group_layout::light", [
            0 => ShaderStages::FRAGMENT, binding_types::uniform::<Light>(),
        ]);

        let material_bgl = bgl!(device, "bind_group_layout::material", [
            0 => ShaderStages::FRAGMENT, binding_types::uniform::<MaterialFactors>(),
            1 => ShaderStages::FRAGMENT, binding_types::texture2d(),
            2 => ShaderStages::FRAGMENT, binding_types::sampler(),
            3 => ShaderStages::FRAGMENT, binding_types::texture2d(),
            4 => ShaderStages::FRAGMENT, binding_types::sampler(),
            5 => ShaderStages::FRAGMENT, binding_types::texture2d(),
            6 => ShaderStages::FRAGMENT, binding_types::sampler(),
        ]);

        let vertex_attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3, 3 => Float32x4, 4 => Float32x3];

        let pipeline = crate::renderer::gpu::kernel::RenderKernelBuilder::new(
            device, "mesh", &shader, format,
        )
            .bind_group_layouts(&[globals_bgl, &model_bgl, &material_bgl, &light_bgl])
            .vertex_layout(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &vertex_attributes,
            })
            .depth(wgpu::TextureFormat::Depth32Float, true, wgpu::CompareFunction::Less)
            .build();

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler::default"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let (mesh_vertices, mesh_indices, primitives) =
            load_glb_mesh(device, queue, &material_bgl, &sampler);

        let mesh_positions: Vec<[f32; 3]> = mesh_vertices.iter().map(|v| v.position).collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&mesh_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&mesh_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let model_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ubo::model"),
            size: std::mem::size_of::<Model>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let model_bind_group = bind_group!(device, "bind_group::model", &model_bgl, [
            0 => model_ubo.as_entire_binding(),
        ]);

        let light_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ubo::light"),
            size: std::mem::size_of::<Light>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_bind_group = bind_group!(device, "bind_group::light", &light_bgl, [
            0 => light_ubo.as_entire_binding(),
        ]);

        let padded_positions: Vec<[f32; 4]> =
            mesh_positions.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let position_storage = GpuBuffer::<[f32; 4]>::storage_init(device, "buffer::positions_storage", &padded_positions);

        let index_storage = GpuBuffer::<u32>::storage_init(device, "buffer::indices_storage", &mesh_indices);

        let ray_ubo = GpuBuffer::<GpuRayUniform>::uniform(device, "ubo::ray");

        let result_ssbo = GpuBuffer::<GpuHitResult>::storage_rw(device, "ssbo::raycast_result", 1);

        let staging = Readback::<GpuHitResult>::new(device, "buffer::raycast_staging");

        let raycast_kernel = ComputeKernel::new(
            device,
            "raycast",
            include_str!("../shaders/raycast.wgsl"),
            "cs_main",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: binding_types::storage_ro(),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: binding_types::storage_ro(),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: binding_types::uniform::<GpuRayUniform>(),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: binding_types::storage_rw(),
                    count: None,
                },
            ],
        );

        let raycast_bind_group = raycast_kernel.bind(device, &[
            wgpu::BindGroupEntry { binding: 0, resource: position_storage.binding() },
            wgpu::BindGroupEntry { binding: 1, resource: index_storage.binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ray_ubo.binding() },
            wgpu::BindGroupEntry { binding: 3, resource: result_ssbo.binding() },
        ]);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            primitives,
            globals_bind_group,
            model_ubo,
            model_bind_group,
            light_ubo,
            light_bind_group,
            light_direction: Vec3::new(-0.25, 0.5, -0.5),
            mesh_positions,
            mesh_indices,
            position_storage,
            index_storage,
            ray_ubo,
            result_ssbo,
            staging,
            raycast_bind_group,
            raycast_kernel,
            model_matrix: Mat4::IDENTITY,
            selected: false,
        }
    }

    pub fn set_light_direction(&mut self, direction: Vec3) {
        self.light_direction = direction;
    }

    pub fn ray_intersect(&self, ray: &Ray) -> Option<Vec3> {
        ray.intersect_mesh(&self.mesh_positions, &self.mesh_indices)
    }

    pub fn ray_intersect_with_model(&self, ray: &Ray, model: &Mat4) -> Option<Vec3> {
        let inv = model.inverse();
        let orig = inv.transform_point3(ray.origin);
        let dir = inv.transform_vector3(ray.direction);
        let local_ray = Ray { origin: orig, direction: dir };
        local_ray.intersect_mesh(&self.mesh_positions, &self.mesh_indices).map(|p| model.transform_point3(p))
    }

    pub fn set_model_matrix(&mut self, mat: Mat4) {
        self.model_matrix = mat;
    }

    pub fn mesh_center(&self) -> Vec3 {
        if self.mesh_positions.is_empty() {
            return Vec3::ZERO;
        }
        let mut sum = Vec3::ZERO;
        for p in &self.mesh_positions {
            sum += Vec3::from(*p);
        }
        sum / self.mesh_positions.len() as f32
    }

    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        if self.mesh_positions.is_empty() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let mut bmin = Vec3::splat(f32::MAX);
        let mut bmax = Vec3::splat(f32::MIN);
        for p in &self.mesh_positions {
            let v = Vec3::from(*p);
            bmin = bmin.min(v);
            bmax = bmax.max(v);
        }
        (bmin, bmax)
    }

    pub fn raw_triangles(&self) -> (&[[f32; 3]], &[u32]) {
        (&self.mesh_positions, &self.mesh_indices)
    }

    pub fn vertex_position(&self, idx: u32) -> Vec3 {
        Vec3::from(self.mesh_positions[idx as usize])
    }

    pub fn select_mesh_action(&self) -> bool {
        true
    }

    pub fn dispatch_raycast(&self, device: &wgpu::Device, queue: &wgpu::Queue, ray: &Ray) {
        let ray_uniform = GpuRayUniform::from(ray);
        self.ray_ubo.write(queue, &ray_uniform);

        let reset = GpuHitResult { hit: 0, t_bits: f32::INFINITY.to_bits() };
        self.result_ssbo.write(queue, &reset);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder::raycast"),
        });

        let num_tris = self.mesh_indices.len() / 3;
        self.raycast_kernel.dispatch(
            &mut encoder,
            &self.raycast_bind_group,
            ((num_tris as u32 + 63) / 64, 1, 1),
        );

        self.staging.copy_from(&mut encoder, &self.result_ssbo);

        queue.submit(Some(encoder.finish()));
    }

    pub fn staging(&self) -> &Readback<GpuHitResult> {
        &self.staging
    }
}

impl Moveable for MeshPass {
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
        self.mesh_center()
    }

    fn bounding_size(&self) -> f32 {
        let (bmin, bmax) = self.bounding_box();
        (bmax - bmin).length()
    }

    fn ray_intersect(&self, ray: &Ray, model: &Mat4) -> Option<Vec3> {
        self.ray_intersect_with_model(ray, model)
    }

    fn gizmo_color(&self) -> Vec3 {
        Vec3::new(1.0, 0.5, 0.1)
    }

    fn gizmo_lines(&self, model: &Mat4) -> Vec<(Vec3, Vec3, Vec3)> {
        let highlight_color = Vec3::new(1.0, 0.5, 0.1);
        let (_, indices) = self.raw_triangles();
        let mut lines = Vec::with_capacity(indices.len());
        for tri in indices.chunks_exact(3) {
            let p0 = model.transform_point3(self.vertex_position(tri[0]));
            let p1 = model.transform_point3(self.vertex_position(tri[1]));
            let p2 = model.transform_point3(self.vertex_position(tri[2]));
            lines.push((p0, p1, highlight_color));
            lines.push((p1, p2, highlight_color));
            lines.push((p2, p0, highlight_color));
        }
        lines
    }
}

impl RenderPass for MeshPass {
    fn prepare(&mut self, queue: &wgpu::Queue, _camera: &Camera) {
        let model = Model::new(self.model_matrix);
        queue.write_buffer(&self.model_ubo, 0, bytemuck::cast_slice(&[model]));
        let light = Light::new(self.light_direction.normalize());
        queue.write_buffer(&self.light_ubo, 0, bytemuck::cast_slice(&[light]));
    }

    fn render<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.globals_bind_group, &[]);
        rpass.set_bind_group(1, &self.model_bind_group, &[]);
        rpass.set_bind_group(3, &self.light_bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        for primitive in &self.primitives {
            rpass.set_bind_group(2, &primitive.bind_group, &[]);
            let start = primitive.index_start;
            let end = start + primitive.index_count;
            rpass.draw_indexed(start..end, primitive.base_vertex, 0..1);
        }
    }
}
