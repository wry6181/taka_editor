mod renderer;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

fn resize_canvas(canvas: &web_sys::HtmlCanvasElement) {
    let window = web_sys::window().expect("no window");
    let w = window.inner_width().unwrap().as_f64().unwrap() as u32;
    let h = window.inner_height().unwrap().as_f64().unwrap() as u32;

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

    let cb = Closure::wrap(Box::new(move || {
        resize_canvas(&canvas);
    }) as Box<dyn FnMut()>);
    window
        .add_event_listener_with_callback("resize", cb.as_ref().unchecked_ref())
        .expect("failed to add resize listener");
    cb.forget();

    renderer::start_render_loop();
}
