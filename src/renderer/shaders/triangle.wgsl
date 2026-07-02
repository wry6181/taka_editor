struct Globals {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) frag_color: vec3<f32>,
};

struct Model {
    model_matrix: mat4x4<f32>,
};

@group(0)
@binding(0)
var<uniform> globals: Globals;

@group(1)
@binding(0)
var<uniform> model: Model;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = model.model_matrix * vec4<f32>(input.position, 1.0);
    output.position = globals.projection * globals.view * world_pos;
    let normal_mat = mat3x3<f32>(
        model.model_matrix[0].xyz,
        model.model_matrix[1].xyz,
        model.model_matrix[2].xyz,
    );
    output.world_normal = normalize(normal_mat * input.normal);
    output.frag_color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let normal = normalize(input.world_normal);
    let diffuse = max(dot(normal, light_dir), 0.0) * 0.8;
    let ambient = 0.2;
    let intensity = ambient + diffuse;
    return vec4<f32>(input.frag_color * intensity, 1.0);
}
