use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::JsCast;

thread_local! {
    static CACHE: RefCell<HashMap<&'static str, web_sys::Element>> = RefCell::new(HashMap::new());
    static BTNS: RefCell<Vec<(&'static str, web_sys::Element)>> = const { RefCell::new(Vec::new()) };
    static PREV_GIZMO: RefCell<String> = const { RefCell::new(String::new()) };
}

fn elem(id: &'static str) -> Option<web_sys::Element> {
    CACHE.with(|cache| {
        let mut map = cache.borrow_mut();
        if let Some(el) = map.get(id) {
            return Some(el.clone());
        }
        if let Some(el) = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id(id))
        {
            map.insert(id, el.clone());
            Some(el)
        } else {
            None
        }
    })
}

pub fn init() {
    elem("sel-name");
    elem("sel-pos");
    elem("sel-size");
    elem("gizmo-translate");
    elem("gizmo-rotate");
    elem("gizmo-scale");

    let mut btns = Vec::new();
    btns.push(init_btn("gizmo-translate", "translate", super::renderer::set_gizmo_translate));
    btns.push(init_btn("gizmo-rotate", "rotate", super::renderer::set_gizmo_rotate));
    btns.push(init_btn("gizmo-scale", "scale", super::renderer::set_gizmo_scale));
    BTNS.with(|cell| *cell.borrow_mut() = btns);
    // Force the first frame to apply the correct button state
    PREV_GIZMO.with(|p| *p.borrow_mut() = String::from("__init__"));
}

fn init_btn(id: &'static str, mode: &'static str, action: fn()) -> (&'static str, web_sys::Element) {
    let el = elem(id).expect("gizmo button not found");

    let cb = wasm_bindgen::closure::Closure::wrap(Box::new(move || {
        action();
        BTNS.with(|btns| {
            for (m, e) in btns.borrow().iter() {
                let cls = if *m == mode { "gizmo-btn active" } else { "gizmo-btn" };
                let _ = e.set_attribute("class", cls);
            }
        });
    }) as Box<dyn FnMut()>);

    let _ = el.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref());
    cb.forget();

    (mode, el)
}

pub fn update(selection: Option<SelectionInfo>, gizmo_mode: &str) {
    match selection {
        Some(ref info) => {
            set_text("sel-name", &info.name);
            set_text("sel-pos", &format!("{:.2}, {:.2}, {:.2}", info.pos_x, info.pos_y, info.pos_z));
            set_text("sel-size", &format!("{:.2}", info.size));
        }
        None => {
            set_text("sel-name", "—");
            set_text("sel-pos", "—");
            set_text("sel-size", "—");
        }
    }

    let target = if selection.is_some() { gizmo_mode } else { "" };

    PREV_GIZMO.with(|prev| {
        if *prev.borrow() != target {
            *prev.borrow_mut() = target.to_string();
            BTNS.with(|btns| {
                for (m, el) in btns.borrow().iter() {
                    let cls = if *m == target { "gizmo-btn active" } else { "gizmo-btn" };
                    let _ = el.set_attribute("class", cls);
                }
            });
        }
    });
}

fn set_text(id: &'static str, text: &str) {
    if let Some(el) = elem(id) {
        el.set_text_content(Some(text));
    }
}

pub struct SelectionInfo {
    pub name: String,
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub size: f32,
}
