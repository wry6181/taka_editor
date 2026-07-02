pub trait RenderPass {
    fn prepare(&mut self, queue: &wgpu::Queue, camera: &crate::renderer::camera::Camera);
    fn render<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>);
}
