struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coord: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec3<f32>,
    @location(3) world_bitangent: vec3<f32>,
    @location(4) color: vec3<f32>,
};

struct CameraBinding {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
}
struct ModelBinding {
    model_matrix: mat4x4<f32>,
}

struct LightBinding {
    direction: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraBinding;
@group(1) @binding(0) var<uniform> model: ModelBinding;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let world_position = model.model_matrix * vec4<f32>(input.position, 1.0);
    output.position = camera.projection_matrix * camera.view_matrix * world_position;

    // NOTE: assumes uniform scale (fine here since world transforms are
    // baked into vertex data on the CPU and model_matrix stays identity).
    let world_normal = normalize((model.model_matrix * vec4<f32>(input.normal, 0.0)).xyz);
    let world_tangent = normalize((model.model_matrix * vec4<f32>(input.tangent.xyz, 0.0)).xyz);
    let world_bitangent = cross(world_normal, world_tangent) * input.tangent.w;

    output.tex_coord = input.tex_coord;
    output.world_normal = world_normal;
    output.world_tangent = world_tangent;
    output.world_bitangent = world_bitangent;
    output.color = input.color;

    return output;
}

struct MaterialFactorsBinding {
    base_color_factor: vec4<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
}

@group(2) @binding(0) var<uniform> factors: MaterialFactorsBinding;
@group(2) @binding(1) var base_color_texture: texture_2d<f32>;
@group(2) @binding(2) var base_color_sampler: sampler;
@group(2) @binding(3) var normal_texture: texture_2d<f32>;
@group(2) @binding(4) var normal_sampler: sampler;
@group(2) @binding(5) var metallic_roughness_texture: texture_2d<f32>;
@group(2) @binding(6) var metallic_roughness_sampler: sampler;

@group(3) @binding(0) var<uniform> light: LightBinding;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var base_color = factors.base_color_factor
        * textureSample(base_color_texture, base_color_sampler, input.tex_coord);
    base_color = vec4<f32>(base_color.rgb * input.color, base_color.a);

    let metallic_roughness = textureSample(metallic_roughness_texture, metallic_roughness_sampler, input.tex_coord);
    let metallic = factors.metallic_factor * metallic_roughness.b;
    let roughness = factors.roughness_factor * metallic_roughness.g;

    var normal_sample = textureSample(normal_texture, normal_sampler, input.tex_coord).xyz;
    normal_sample = normal_sample * 2.0 - 1.0;
    let normal = normalize(
        input.world_tangent * normal_sample.x
        + input.world_bitangent * normal_sample.y
        + input.world_normal * normal_sample.z
    );

    let light_direction = normalize(light.direction);
    let normal_dot_light = max(dot(normal, light_direction), 0.0);
    // Cheap non-PBR shading, just enough to read shape/material correctly.
    let lit_color = base_color.rgb * (0.15 + 0.85 * normal_dot_light) * (1.0 - metallic * 0.5);

    return vec4<f32>(lit_color, base_color.a);
}
