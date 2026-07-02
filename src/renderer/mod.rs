mod wgpu;

use std::cell::RefCell;
use std::rc::Rc;
use glam::Mat4;
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

    pub fn camera_orbit(&mut self, dx: f64, dy: f64) {
        self.inner.camera_orbit(dx, dy);
    }

    pub fn camera_zoom(&mut self, delta: f64) {
        self.inner.camera_zoom(delta);
    }

    pub fn set_light_direction(&mut self, x: f64, y: f64, z: f64) {
        self.inner.set_light_direction(glam::Vec3::new(x as f32, y as f32, z as f32));
    }

    pub fn set_lines(&mut self, lines: &[(glam::Vec3, glam::Vec3, glam::Vec3)]) {
        self.inner.set_lines(lines);
    }

    pub fn get_camera_info(&self) -> Mat4 {
        self.inner.get_camera_info()
    }

}

thread_local! {
    pub static RENDERER: RefCell<Option<Renderer>> = const { RefCell::new(None) };
    pub static ORBIT_DELTA: RefCell<(f64, f64)> = RefCell::new((0.0, 0.0));
    pub static DEBUG_OVERLAY: RefCell<Option<web_sys::Element>> = RefCell::new(None);
    static GIZMO_ELEMENT: std::cell::RefCell<Option<web_sys::Element>> = const { std::cell::RefCell::new(None) };
}

pub fn resize_renderer(width: u32, height: u32) {
    RENDERER.with(|rc| {
        if let Some(r) = rc.borrow_mut().as_mut() {
            r.resize(width, height);
        }
    });
}

fn get_debug_overlay() -> Option<web_sys::Element> {
    DEBUG_OVERLAY.with(|cell| {
        if cell.borrow().is_none() {
            let el = web_sys::window()?
                .document()?
                .get_element_by_id("debug-overlay");
            *cell.borrow_mut() = el;
        }
        cell.borrow().clone()
    })
}

fn format_mat4(label: &str, m: glam::Mat4) -> String {
    let c = m.to_cols_array();
    format!(
        "{label}:\n  [{:>6.2} {:>6.2} {:>6.2} {:>6.2}]\n  [{:>6.2} {:>6.2} {:>6.2} {:>6.2}]\n  [{:>6.2} {:>6.2} {:>6.2} {:>6.2}]\n  [{:>6.2} {:>6.2} {:>6.2} {:>6.2}]",
        c[0], c[4], c[8], c[12],
        c[1], c[5], c[9], c[13],
        c[2], c[6], c[10], c[14],
        c[3], c[7], c[11], c[15],
    )
}

fn update_debug_overlay(r: &Renderer) {
    let Some(el) = get_debug_overlay() else { return };
    let view = r.get_camera_info();
    let text = format!(
        "{}\n",
        format_mat4("View", view),
    );
    el.set_text_content(Some(&text));
    // 2. Drive the 3D CSS Gizmo
        GIZMO_ELEMENT.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = web_sys::window()
                    .and_then(|w| w.document())
                    .and_then(|d| d.get_element_by_id("gizmo-space"));
            }

            if let Some(gizmo_el) = cell.borrow().as_ref() {
                // Extract the top-left 3x3 rotation columns from your glam::Mat4
                let rot = glam::Mat3::from_mat4(view);
                let r = rot.to_cols_array();

                // Format into CSS matrix3d layout.
                // We use the rotation columns and keep scale at 1.0, translations at 0.0
                let css_matrix = format!(
                    "matrix3d({}, {}, {}, 0, {}, {}, {}, 0, {}, {}, {}, 0, 0, 0, 0, 1)",
                    r[0], r[1], r[2],
                    r[3], r[4], r[5],
                    r[6], r[7], r[8]
                );

                let _ = gizmo_el.set_attribute("style", &format!("transform: {};", css_matrix));
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
                RENDERER.with(|rc| {
                    if let Some(r) = rc.borrow_mut().as_mut() {
                        r.set_light_direction(1.25, 2.5, 10.5);
                        let coord_pos = glam::Vec3::new(0.0, 0.0, 5.5);
                        r.set_lines(&[
                          (coord_pos, coord_pos + glam::Vec3::X * 2.0, glam::Vec3::new(1.0, 0.0, 0.0)), // red X
                          (coord_pos, coord_pos + glam::Vec3::Y * 2.0, glam::Vec3::new(0.0, 1.0, 0.0)), // green Y
                          (coord_pos, coord_pos + glam::Vec3::Z * 2.0, glam::Vec3::new(0.0, 0.0, 1.0)), // blue Z
                        ]);
                    }
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
                ORBIT_DELTA.with(|m| {
                    let mut delta = m.borrow_mut();
                    if delta.0 != 0.0 || delta.1 != 0.0 {
                        r.camera_orbit(delta.0, delta.1);
                        *delta = (0.0, 0.0);
                    }
                });

                r.render();
                update_debug_overlay(r);
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
