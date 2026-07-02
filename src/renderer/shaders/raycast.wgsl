struct Ray {
    origin: vec4<f32>,
    direction: vec4<f32>,
}

struct HitResult {
    hit: atomic<u32>,
    t_bits: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<uniform> ray: Ray;
@group(0) @binding(3) var<storage, read_write> result: HitResult;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tri_index = id.x;
    let num_indices = arrayLength(&indices);
    let num_tris = num_indices / 3u;
    if tri_index >= num_tris {
        return;
    }

    let i0 = indices[tri_index * 3u];
    let i1 = indices[tri_index * 3u + 1u];
    let i2 = indices[tri_index * 3u + 2u];

    let v0 = positions[i0];
    let v1 = positions[i1];
    let v2 = positions[i2];

    let orig = ray.origin.xyz;
    let dir = ray.direction.xyz;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(dir, edge2);
    let a = dot(edge1, h);
    if abs(a) < 1e-8 { return; }
    let f = 1.0 / a;
    let s = orig - v0;
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return; }
    let q = cross(s, edge1);
    let v = f * dot(dir, q);
    if v < 0.0 || u + v > 1.0 { return; }
    let t = f * dot(edge2, q);
    if t < 0.0 { return; }

    let t_u32 = bitcast<u32>(t);
    atomicMin(&result.t_bits, t_u32);
    atomicStore(&result.hit, 1u);
}
