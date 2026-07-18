struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<u32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct CameraBinding {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraBinding;
@group(1) @binding(0) var<uniform> model: mat4x4<f32>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = model * vec4<f32>(input.position, 1.0);
    output.clip_position = camera.projection_matrix * camera.view_matrix * world_pos;
    output.color = vec4<f32>(
        f32(input.color.x) / 255.0,
        f32(input.color.y) / 255.0,
        f32(input.color.z) / 255.0,
        f32(input.color.w) / 255.0,
    );
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
