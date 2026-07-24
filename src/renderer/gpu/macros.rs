/// Create a `BindGroupLayout` from a compact list of `(binding, visibility, ty)` entries.
///
/// Each entry is written as:
/// ```ignore
/// binding_number => visibility, type_expression
/// ```
/// For example:
/// ```ignore
/// bgl!(device, "my_layout", [
///     0 => VERTEX, uniform::<Model>(),
///     1 => FRAGMENT, texture2d(),
///     2 => FRAGMENT, sampler(),
/// ])
/// ```
#[macro_export]
macro_rules! bgl {
    ($device:expr, $label:expr, [ $( $binding:expr => $vis:expr, $ty:expr ),* $(,)? ]) => {{
        let entries = &[
            $(
                wgpu::BindGroupLayoutEntry {
                    binding: $binding,
                    visibility: $vis,
                    ty: $ty,
                    count: None,
                },
            )*
        ];
        $device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some($label),
            entries,
        })
    }};
}

/// Create a `BindGroup` from a list of `(binding, resource)` pairs.
///
/// ```ignore
/// bind_group!(device, "my_bg", &layout, [
///     0 => model_ubo.as_entire_binding(),
///     1 => wgpu::BindingResource::TextureView(&texture_view),
/// ])
/// ```
#[macro_export]
macro_rules! bind_group {
    ($device:expr, $label:expr, $layout:expr, [ $( $binding:expr => $res:expr ),* $(,)? ]) => {{
        let entries = &[
            $(
                wgpu::BindGroupEntry {
                    binding: $binding,
                    resource: $res,
                },
            )*
        ];
        $device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some($label),
            layout: $layout,
            entries,
        })
    }};
}
