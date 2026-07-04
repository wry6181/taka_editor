mod renderer;

use std::cell::RefCell;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

thread_local! {
    static DRAGGING: RefCell<bool> = RefCell::new(false);
    static LAST_MOUSE: RefCell<(i32, i32)> = RefCell::new((0, 0));
    static CLICK_POS: RefCell<(i32, i32)> = RefCell::new((0, 0));
}

fn resize_canvas(canvas: &web_sys::HtmlCanvasElement) {
    let w = canvas.client_width().max(1) as u32;
    let h = canvas.client_height().max(1) as u32;

    canvas.set_width(w);
    canvas.set_height(h);
    renderer::resize_renderer(w, h);
}

#[wasm_bindgen(start)]
pub fn run() {
    let window = web_sys::window().expect("no window");
    let document = window.document().expect("no document");
    let canvas = document
        .get_element_by_id("viewport")
        .expect("no canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("not a canvas");

    resize_canvas(&canvas);
    renderer::init_renderer(canvas.clone());

    let canvas_cb = canvas.clone();
    let cb = Closure::wrap(Box::new(move || {
        resize_canvas(&canvas_cb);
    }) as Box<dyn FnMut()>);
    window
        .add_event_listener_with_callback("resize", cb.as_ref().unchecked_ref())
        .expect("failed to add resize listener");
    cb.forget();

    // Mousedown → start orbit drag, or remesh/gizmo drag
    let canvas_md = canvas.clone();
    let on_mousedown = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let cx = event.client_x();
        let cy = event.client_y();
        let gizmo_drag = renderer::RENDERER.with(|rc| {
            rc.borrow_mut().as_mut().map_or(false, |r| r.handle_mousedown(cx as f64, cy as f64))
        });
        if !gizmo_drag {
            DRAGGING.with(|d| *d.borrow_mut() = true);
        }
        LAST_MOUSE.with(|p| *p.borrow_mut() = (cx, cy));
        CLICK_POS.with(|p| *p.borrow_mut() = (cx, cy));
    }) as Box<dyn FnMut(_)>);
    canvas_md
        .add_event_listener_with_callback("mousedown", on_mousedown.as_ref().unchecked_ref())
        .expect("failed to add mousedown listener");
    on_mousedown.forget();

    // Mousemove → orbit if dragging, or gizmo drag (remesh/mesh)
    let on_mousemove = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let cx = event.client_x() as f64;
        let cy = event.client_y() as f64;
        let is_dragging = renderer::RENDERER.with(|rc| {
            rc.borrow_mut().as_mut().map_or(false, |r| {
                if r.remesh_is_dragging() {
                    r.handle_mousemove(cx, cy);
                    true
                } else {
                    false
                }
            })
        });
        if !is_dragging {
            DRAGGING.with(|d| {
                if *d.borrow() {
                    LAST_MOUSE.with(|p| {
                        let (lx, ly) = *p.borrow();
                        let dx = (cy as i32 - ly) as f64;
                        let dy = (cx as i32 - lx) as f64;
                        if dx != 0.0 || dy != 0.0 {
                            renderer::ORBIT_DELTA.with(|m| {
                                let mut delta = m.borrow_mut();
                                delta.0 += dy;
                                delta.1 += dx;
                            });
                            *p.borrow_mut() = (cx as i32, cy as i32);
                        }
                    });
                }
            });
        }
    }) as Box<dyn FnMut(_)>);
    canvas
        .add_event_listener_with_callback("mousemove", on_mousemove.as_ref().unchecked_ref())
        .expect("failed to add mousemove listener");
    on_mousemove.forget();

    // Mouseup → stop orbit drag, end gizmo drag, detect click
    let on_mouseup = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let was_drag = renderer::RENDERER.with(|rc| {
            rc.borrow_mut().as_mut().map_or(false, |r| {
                if r.remesh_is_dragging() {
                    r.handle_mouseup();
                    true
                } else {
                    false
                }
            })
        });
        DRAGGING.with(|d| *d.borrow_mut() = false);
        if !was_drag {
            CLICK_POS.with(|cp| {
                let (sx, sy) = *cp.borrow();
                let dx = (event.client_x() - sx).abs();
                let dy = (event.client_y() - sy).abs();
                if dx < 4 && dy < 4 {
                    let px = event.client_x() as f64;
                    let py = event.client_y() as f64;
                    let consumed = renderer::RENDERER.with(|rc| {
                        rc.borrow_mut().as_mut().map_or(false, |r| r.remesh_handle_click(px, py))
                    });
                    if !consumed {
                        // Try CPU raycast to select mesh (only when remesh is inactive)
                        let mesh_hit = renderer::RENDERER.with(|rc| {
                            rc.borrow_mut().as_mut().map_or(false, |r| r.select_mesh_at_screen(px, py))
                        });
                        if !mesh_hit {
                            // Deselect mesh on empty space click
                            renderer::RENDERER.with(|rc| {
                                if let Some(r) = rc.borrow_mut().as_mut() {
                                    if r.mesh_is_selected() {
                                        r.deselect_mesh();
                                    }
                                }
                            });
                            renderer::RAYCAST_PENDING.with(|p| {
                                *p.borrow_mut() = Some((px, py));
                            });
                        }
                    }
                }
            });
        }
    }) as Box<dyn FnMut(_)>);
    canvas
        .add_event_listener_with_callback("mouseup", on_mouseup.as_ref().unchecked_ref())
        .expect("failed to add mouseup listener");
    on_mouseup.forget();

    // Mouseleave → stop orbit drag
    let on_mouseleave = Closure::wrap(Box::new(move |_: web_sys::MouseEvent| {
        DRAGGING.with(|d| *d.borrow_mut() = false);
    }) as Box<dyn FnMut(_)>);
    canvas
        .add_event_listener_with_callback("mouseleave", on_mouseleave.as_ref().unchecked_ref())
        .expect("failed to add mouseleave listener");
    on_mouseleave.forget();

    // Scroll wheel → zoom (dolly)
    let on_wheel = Closure::wrap(Box::new(move |event: web_sys::WheelEvent| {
        let dy = event.delta_y() as f64;
        if dy != 0.0 {
            renderer::RENDERER.with(|rc| {
                if let Some(r) = rc.borrow_mut().as_mut() {
                    r.camera_zoom(-dy * 0.05);
                }
            });
        }
        event.prevent_default();
    }) as Box<dyn FnMut(_)>);
    canvas
        .add_event_listener_with_callback("wheel", on_wheel.as_ref().unchecked_ref())
        .expect("failed to add wheel listener");
    on_wheel.forget();

    // Keydown → 'R' toggles remesh mode, 'M' toggles mesh visibility, 'G' selects mesh
    let on_keydown = Closure::wrap(Box::new(move |event: web_sys::KeyboardEvent| {
        let key = event.key();
        if key == "r" || key == "R" {
            renderer::RENDERER.with(|rc| {
                if let Some(r) = rc.borrow_mut().as_mut() {
                    r.remesh_toggle();
                }
            });
        } else if key == "m" || key == "M" {
            renderer::RENDERER.with(|rc| {
                if let Some(r) = rc.borrow_mut().as_mut() {
                    r.remesh_toggle_mesh();
                }
            });
        } else if key == "g" || key == "G" {
            renderer::RENDERER.with(|rc| {
                if let Some(r) = rc.borrow_mut().as_mut() {
                    r.toggle_select_mesh();
                }
            });
        }
    }) as Box<dyn FnMut(_)>);
    window
        .add_event_listener_with_callback("keydown", on_keydown.as_ref().unchecked_ref())
        .expect("failed to add keydown listener");
    on_keydown.forget();

    renderer::start_render_loop();
}
