mod wgpu;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

pub struct Renderer {
    inner: wgpu::WgpuRenderer,
}

impl Renderer {
    pub fn render(&mut self) {
        self.inner.render();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.inner.resize(width, height);
    }
}

thread_local! {
    pub static RENDERER: RefCell<Option<Renderer>> = const { RefCell::new(None) };
}

pub fn resize_renderer(width: u32, height: u32) {
    RENDERER.with(|rc| {
        if let Some(r) = rc.borrow_mut().as_mut() {
            r.resize(width, height);
        }
    });
}

pub fn init_renderer(canvas: HtmlCanvasElement) {
    let width = canvas.width();
    let height = canvas.height();

    web_sys::console::log_1(
        &"Initializing wgpu (auto-selects WebGPU or WebGL)".into(),
    );

    wasm_bindgen_futures::spawn_local(async move {
        match wgpu::WgpuRenderer::new(canvas, width, height).await {
            Ok(r) => {
                let backend = r.backend_name();
                web_sys::console::log_1(&format!("wgpu ready (backend: {})", backend).into());
                RENDERER.with(|rc| {
                    *rc.borrow_mut() = Some(Renderer { inner: r });
                });
            }
            Err(e) => {
                web_sys::console::log_1(&format!("wgpu init failed: {}", e).into());
            }
        }
    });
}

pub fn start_render_loop() {
    let f = Rc::new(RefCell::new(None::<Closure<dyn FnMut()>>));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        RENDERER.with(|rc| {
            let mut renderer = rc.borrow_mut();
            if let Some(r) = renderer.as_mut() {
                r.render();
            }
        });
        let window = web_sys::window().expect("no window");
        if let Some(cb) = f.borrow().as_ref() {
            window
                .request_animation_frame(cb.as_ref().unchecked_ref())
                .expect("request_animation_frame failed");
        }
    }) as Box<dyn FnMut()>));

    let window = web_sys::window().expect("no window");
    if let Some(cb) = g.borrow().as_ref() {
        window
            .request_animation_frame(cb.as_ref().unchecked_ref())
            .expect("request_animation_frame failed");
    }
}
