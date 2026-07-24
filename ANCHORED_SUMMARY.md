# Anchored Summary

## Goal
Build a thin GPU-resource API over wgpu to eliminate repeated descriptor ceremony in pass constructors. All four passes (mesh, grid, line, image) and the raycast compute path are now ported.

## Constraints & Preferences
- WASM target (`wasm32-unknown-unknown`), WebGPU backend
- `wgpu` 0.29, `glam` 0.29, `bytemuck` 1.25
- Never call `device.poll(PollType::Wait)` or block on GPU work — use non-blocking `map_async` with callback + per-frame `PollType::Poll`
- No `async fn` / `.await` in the GPU readback path
- WGSL shaders stay as `.wgsl` string files (`include_str!`)
- `Gizmo` / `Moveable` trait refactoring from previous sessions remains unchanged

## Progress

### Completed
1. Created `src/renderer/gpu/` module with `mod.rs`, `buffer.rs`, `binding_types.rs`, `macros.rs`, `kernel.rs`
2. `GpuBuffer<T>` — typed wgpu buffer wrapping, factory methods per role
3. `Readback<T>` — non-blocking GPU→CPU readback via `map_async` + callback
4. `binding_types.rs` — `uniform::<T>()`, `storage_ro()`, `storage_rw()`, `texture2d()`, `sampler()`
5. `macros.rs` — `bgl!` and `bind_group!` macros
6. `kernel.rs` — `ComputeKernel` (compute pipeline + bind + dispatch) and `RenderKernelBuilder` (render pipeline builder)
7. **Phase 2**: Ported raycast compute path in `mesh_pass.rs` to `ComputeKernel`/`GpuBuffer`/`Readback`; updated `wgpu.rs::raycast_gpu`
8. **Phase 3**: Ported mesh render pipeline to `RenderKernelBuilder` + `bgl!` macro
9. **Phase 4**: Ported `grid_pass.rs`, `line_pass.rs`, `image_pass.rs` to `RenderKernelBuilder`
10. Fixed build errors related to `Send`, `Zeroable`, owned vertex attributes, duplicate imports/fields, temporary lifetimes

### Remaining warnings (all pre-existing)
- `half_edge.rs` — unused types
- `gizmo.rs` — unused methods
- `moveable.rs` — unused trait methods
- `mesh_pass.rs` — unused fields/methods
- `wgpu.rs` / `mod.rs` — unused methods

## Key Decisions
- `#[macro_export]` for macros — used as `crate::bgl!()` / `crate::bind_group!()`
- `RenderKernelBuilder` stores vertex attributes as owned `Vec<wgpu::VertexAttribute>` to avoid temporary lifetime issues
- `ComputeKernel` and `RenderKernelBuilder` are separate types
- `GpuBuffer` factory methods encode usage flags by role
- `Readback::request` hides `map_async` — caller only provides callback

## Relevant Files
- `src/renderer/gpu/mod.rs`, `buffer.rs`, `binding_types.rs`, `macros.rs`, `kernel.rs`
- `src/renderer/passes/mesh_pass.rs` — fully ported (raycast + render)
- `src/renderer/passes/grid_pass.rs` — ported
- `src/renderer/passes/line_pass.rs` — ported
- `src/renderer/passes/image_pass.rs` — ported
- `src/renderer/wgpu.rs` — uses `staging().request()`
