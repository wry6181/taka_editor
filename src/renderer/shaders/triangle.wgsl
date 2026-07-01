struct Globals {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
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
    output.position = globals.projection * globals.view * model.model_matrix * vec4<f32>(input.position, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
