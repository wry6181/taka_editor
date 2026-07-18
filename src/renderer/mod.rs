mod camera;
mod half_edge;
mod moveable;
mod pass;
mod passes;
mod ray;
mod wgpu;


use crate::file_loader::TakaImage;
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

    pub fn set_clear_color(&mut self, r: f64, g: f64, b: f64) {
        self.inner.set_clear_color(r, g, b);
    }

    pub fn canvas_width(&self) -> f64 {
        self.inner.canvas_width()
    }

    pub fn canvas_height(&self) -> f64 {
        self.inner.canvas_height()
    }

    pub fn raycast(&mut self, px: f64, py: f64) {
        self.inner.raycast_gpu(px, py);
    }

    pub fn set_lines(&mut self, lines: &[(glam::Vec3, glam::Vec3, glam::Vec3)]) {
        self.inner.set_lines(lines);
    }

    pub fn get_camera_info(&self) -> Mat4 {
        self.inner.get_camera_info()
    }

    pub fn handle_mousedown(&mut self, px: f64, py: f64) -> bool {
        self.inner.handle_mousedown(px, py)
    }

    pub fn handle_mousemove(&mut self, px: f64, py: f64) {
        self.inner.handle_mousemove(px, py);
    }

    pub fn handle_mouseup(&mut self) -> bool {
        self.inner.handle_mouseup()
    }

    pub fn mesh_is_dragging(&self) -> bool {
        self.inner.mesh_is_dragging()
    }

    pub fn toggle_select_mesh(&mut self) {
        self.inner.toggle_select_mesh();
    }

    pub fn toggle_select_image(&mut self) {
        self.inner.toggle_select_image();
    }

    pub fn mesh_is_selected(&self) -> bool {
        self.inner.mesh_is_selected()
    }

    pub fn select_mesh_at_screen(&mut self, px: f64, py: f64) -> bool {
        self.inner.select_mesh_at_screen(px, py)
    }

    pub fn select_image_at_screen(&mut self, px: f64, py: f64) -> bool {
        self.inner.select_image_at_screen(px, py)
    }

    pub fn deselect_mesh(&mut self) {
        self.inner.deselect_mesh();
    }

}

pub struct GpuRaycastOutcome {
    pub hit: bool,
    pub t: f32,
}

thread_local! {
    pub static RENDERER: RefCell<Option<Renderer>> = const { RefCell::new(None) };
    pub static ORBIT_DELTA: RefCell<(f64, f64)> = RefCell::new((0.0, 0.0));
    pub static DEBUG_OVERLAY: RefCell<Option<web_sys::Element>> = RefCell::new(None);
    static GIZMO_ELEMENT: std::cell::RefCell<Option<web_sys::Element>> = const { std::cell::RefCell::new(None) };
    static FPS_STATE: RefCell<FpsCounter> = RefCell::new(FpsCounter::new());
    pub static RAYCAST_PENDING: RefCell<Option<(f64, f64)>> = const { RefCell::new(None) };
    pub static GPU_RAYCAST_RESULT: RefCell<Option<GpuRaycastOutcome>> = const { RefCell::new(None) };
    static PENDING_IMAGE: RefCell<Option<TakaImage>> = const { RefCell::new(None) };
}

struct FpsCounter {
    last_time: f64,
    frame_count: u32,
    accumulator: f64,
    fps: f64,
}

impl FpsCounter {
    fn new() -> Self {
        Self { last_time: 0.0, frame_count: 0, accumulator: 0.0, fps: 0.0 }
    }

    fn tick(&mut self, now: f64) -> f64 {
        if self.last_time == 0.0 {
            self.last_time = now;
            return 0.0;
        }
        let dt = now - self.last_time;
        self.last_time = now;
        self.frame_count += 1;
        self.accumulator += dt;
        if self.accumulator >= 500.0 {
            self.fps = self.frame_count as f64 / self.accumulator * 1000.0;
            self.frame_count = 0;
            self.accumulator = 0.0;
        }
        self.fps
    }
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

    let fps = FPS_STATE.with(|f| {
        let now = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
        f.borrow_mut().tick(now)
    });

    let text = format!(
        "FPS: {:.0}\n{}\n",
        fps,
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
                PENDING_IMAGE.with(|p| {
                    if let Some(img) = p.borrow_mut().take() {
                        add_image(img);
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

                RAYCAST_PENDING.with(|pending| {
                    if let Some((px, py)) = pending.borrow_mut().take() {
                        let w = r.canvas_width();
                        let h = r.canvas_height();
                        if w > 0.0 && h > 0.0 {
                            r.raycast(px, py);
                        }
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

pub fn add_image(image: TakaImage) {
    let applied = RENDERER.with(|rc| {
        if let Some(r) = rc.borrow_mut().as_mut() {
            r.inner.set_image(&image.color, &image.position);
            true
        } else {
            false
        }
    });
    if !applied {
        PENDING_IMAGE.with(|p| *p.borrow_mut() = Some(image));
    }
}
