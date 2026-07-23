/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Native Mersey engine hosted inside Servo — the "native" leg of the
//! `bench/web` benchmark, the Servo counterpart of the Gecko fork's
//! `dom/mersey/MerseyScriptRunner` and the Chromium fork's `//components/mersey`.
//!
//! A `<script type="text/mersey">` runs in the engine directly (Rust
//! interpreter + Cranelift JIT), *not* as WASM and *not* as JS. Web-API calls
//! prefer the direct-Rust tier: hot methods classified at intern time dispatch
//! straight to Servo's own DOM implementations (`NodeMethods::AppendChild`,
//! `StorageMethods::SetItem`, `CryptoMethods::GetRandomValues`,
//! `TextEncoderMethods::Encode`, and the canvas draw loop through `web_bind` to
//! `CanvasRenderingContext2DMethods::FillRect`) — the Rust API, not JS. Only
//! what falls outside the hot set crosses through the reflective bridge
//! (`web/mersey-bridge.js`, embedded as `bridge.js`) into the SpiderMonkey
//! realm, which also owns the handle table for object identity. Same results
//! on every path, verified by matching workload checksums.
//!
//! Threading/re-entrancy: one engine context per script thread, always called
//! from that thread; a bridge call the engine makes can re-enter (a JS callback
//! invoking a Mersey closure), so the runner is reached through a raw pointer,
//! not a `RefCell` (which would panic on the legitimate re-entrant stack — the
//! same discipline `mersey_capi`'s own `MsyContext` uses).

#![allow(unsafe_code)]

use std::borrow::Cow;
use std::cell::Cell;
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;
use std::ptr::{self, NonNull};
use std::time::Instant;

use js::context::JSContext;
use js::conversions::{jsstr_to_string, Utf8Chars};
use js::jsapi::{CallArgs, Heap, JSObject, Value};
use js::typedarray::{ArrayBufferView, ArrayBufferViewU8, TypedArray};
use js::jsval::{BooleanValue, DoubleValue, NullValue, StringValue, UndefinedValue};
use js::rust::wrappers2::{
    JS_CallFunctionName, JS_DefineFunction, JS_GetProperty, JS_NewStringCopyUTF8N,
};
use js::rust::HandleObject;
use script_bindings::reflector::DomObject;
use script_bindings::settings_stack::run_a_script;

use mersey_capi::{
    msy_context_invoke_args, msy_context_new, msy_context_repl_turn, msy_context_run, MsyArg16,
    MsyContext, MsyHostTable,
    MsyReply, MsyScalar,
};

use std::collections::HashMap;

use js::jsval::ObjectValue;
use stylo_atoms::Atom;

use script_bindings::trace::RootedTraceableBox;

use crate::dom::bindings::codegen::Bindings::CSSStyleDeclarationBinding::CSSStyleDeclarationMethods;
use crate::dom::bindings::codegen::Bindings::CanvasRenderingContext2DBinding::CanvasRenderingContext2DMethods;
use crate::dom::bindings::codegen::Bindings::CryptoBinding::CryptoMethods;
use crate::dom::bindings::codegen::Bindings::DOMTokenListBinding::DOMTokenListMethods;
use crate::dom::bindings::codegen::Bindings::StorageBinding::StorageMethods;
use crate::dom::bindings::codegen::Bindings::TextDecoderBinding::{TextDecodeOptions, TextDecoderMethods};
use crate::dom::bindings::codegen::Bindings::TextEncoderBinding::TextEncoderMethods;
use crate::dom::bindings::codegen::UnionTypes::ArrayBufferViewOrArrayBuffer;
use crate::dom::bindings::codegen::Bindings::DocumentBinding::DocumentMethods;
use crate::dom::bindings::codegen::Bindings::ElementBinding::ElementMethods;
use crate::dom::bindings::codegen::Bindings::EventTargetBinding::EventTargetMethods;
use crate::dom::bindings::codegen::Bindings::HTMLElementBinding::HTMLElementMethods;
use crate::dom::bindings::codegen::Bindings::NodeBinding::NodeMethods;
use crate::dom::bindings::codegen::Bindings::NodeListBinding::NodeListMethods;
use crate::dom::bindings::codegen::Bindings::URLBinding::URLMethods;
use crate::dom::bindings::codegen::UnionTypes::StringOrElementCreationOptions;
use crate::dom::bindings::conversions::root_from_object;
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::root::DomRoot;
use crate::dom::bindings::str::{DOMString, USVString};
use crate::dom::canvasrenderingcontext2d::CanvasRenderingContext2D;
use crate::dom::crypto::Crypto;
use crate::dom::cssstyledeclaration::CSSStyleDeclaration;
use crate::dom::document::Document;
use crate::dom::domtokenlist::DOMTokenList;
use crate::dom::element::Element;
use crate::dom::event::{Event, EventBubbles, EventCancelable};
use crate::dom::eventtarget::EventTarget;
use crate::dom::globalscope::GlobalScope;
use crate::dom::htmlelement::HTMLElement;
use crate::dom::node::Node;
use crate::dom::nodelist::NodeList;
use crate::dom::storage::Storage;
use crate::dom::textdecoder::TextDecoder;
use crate::dom::textencoder::TextEncoder;
use crate::dom::url::URL;
use crate::realms::enter_auto_realm;

/// The reflective bridge JS, generated from `web/mersey-bridge.js` (import
/// stripped, `globalThis.__merseyBridge = makeBridge(...)` epilogue appended).
const BRIDGE_JS: &str = include_str!("bridge.js");

/// Capabilities granted to the engine (spec §5.4). Matches the Gecko fork:
/// the whole web surface is reachable, the engine still gates each API by import.
const CAPS: &str = "[\"dom\",\"web\",\"time\",\"random\",\"net\",\"storage\"]";

/// Hot methods with a direct-Rust path (the direct-DOM tier, ported from the
/// Ladybird fork): the engine interns a name once, we classify it then, and the
/// wide-path hooks switch on the id straight to the Servo DOM method — skipping
/// the reflective bridge into SpiderMonkey entirely. `Index` is an indexed
/// access (`nodes[i]` crosses as a digit-only property name).
#[derive(Clone, Copy, PartialEq)]
enum Hot {
    None,
    CtorUrl,          // new URL(str)              -> URL::Constructor
    Pathname,         // url.pathname              -> URLMethods::Pathname
    Search,           // url.search                -> URLMethods::Search
    CreateElement,    // document.createElement(t) -> DocumentMethods::CreateElement
    AppendChild,      // node.appendChild(child)   -> NodeMethods::AppendChild
    TextContent,      // el.textContent (get+set)  -> NodeMethods::{Get,Set}TextContent
    CtorEvent,        // new Event(type)           -> Event::new
    DispatchEvent,    // el.dispatchEvent(ev)      -> EventTargetMethods::DispatchEvent
    ClassName,        // el.className = s          -> ElementMethods::SetClassName
    ClassList,        // el.classList              -> ElementMethods::ClassList
    Contains,         // tokens.contains(s)        -> DOMTokenListMethods::Contains
    Style,            // el.style                  -> HTMLElementMethods::Style
    SetProperty,      // style.setProperty(p, v)   -> CSSStyleDeclarationMethods::SetProperty
    GetPropertyValue, // style.getPropertyValue(p) -> CSSStyleDeclarationMethods::GetPropertyValue
    QuerySelectorAll, // doc.querySelectorAll(sel) -> DocumentMethods::QuerySelectorAll
    Length,           // nodes.length              -> NodeListMethods::Length
    Index(u32),       // nodes[i]                  -> NodeListMethods::Item
    GetRandomValues,  // crypto.getRandomValues(b) -> CryptoMethods::GetRandomValues
    Encode,           // enc.encode(s)             -> TextEncoderMethods::Encode
    Decode,           // dec.decode(bytes)         -> TextDecoderMethods::Decode
    GetItem,          // storage.getItem(k)        -> StorageMethods::GetItem
    SetItem,          // storage.setItem(k, v)     -> StorageMethods::SetItem
    RemoveItem,       // storage.removeItem(k)     -> StorageMethods::RemoveItem
}

/// Cached native pointers for a bridge handle, one slot per role. Servo DOM
/// natives are boxed and never move (only the JS reflector does), and the
/// bridge's handle table keeps the reflector — and so the native — alive until
/// `web_release`, which invalidates this cache. Null = not resolved yet.
#[derive(Clone, Copy)]
struct Natives {
    url: *const URL,
    document: *const Document,
    element: *const Element,
    html: *const HTMLElement,
    node: *const Node,
    node_list: *const NodeList,
    token_list: *const DOMTokenList,
    style: *const CSSStyleDeclaration,
    event: *const Event,
    target: *const EventTarget,
    storage: *const Storage,
    crypto: *const Crypto,
    encoder: *const TextEncoder,
    decoder: *const TextDecoder,
    canvas2d: *const CanvasRenderingContext2D,
}

impl Default for Natives {
    fn default() -> Self {
        Natives {
            url: ptr::null(),
            document: ptr::null(),
            element: ptr::null(),
            html: ptr::null(),
            node: ptr::null(),
            node_list: ptr::null(),
            token_list: ptr::null(),
            style: ptr::null(),
            event: ptr::null(),
            target: ptr::null(),
            storage: ptr::null(),
            crypto: ptr::null(),
            encoder: ptr::null(),
            decoder: ptr::null(),
            canvas2d: ptr::null(),
        }
    }
}

/// Per-thread engine runner. Reached through a raw pointer (see module doc).
struct Runner {
    ctx: *mut MsyContext,
    /// The page's global object (kept alive by the realm for the page lifetime).
    global: *mut JSObject,
    bridge_ready: bool,
    /// Backing store for a reply the engine reads — valid until the next host
    /// call on this runner, exactly the C-ABI contract.
    scratch: String,
    /// Backing store for a typed UTF-16 reply (MsyReply::str16) on the wide path.
    reply16: Vec<u16>,
    /// Direct-DOM tier: interned-name id → hot method, and handle → cached
    /// native pointers (invalidated on web_release).
    hot: Vec<Hot>,
    natives: HashMap<i64, Natives>,
    start: Instant,
}

thread_local! {
    static RUNNER: Cell<*mut Runner> = const { Cell::new(ptr::null_mut()) };
}

fn runner_ptr() -> *mut Runner {
    RUNNER.with(|c| c.get())
}

// ---- host table shims -----------------------------------------------------

extern "C" fn host_print(_data: *mut c_void, utf8: *const c_char, len: usize) {
    use std::io::Write;
    let bytes = unsafe { std::slice::from_raw_parts(utf8 as *const u8, len) };
    let out = std::io::stdout();
    let mut h = out.lock();
    let _ = h.write_all(bytes);
    let _ = h.write_all(b"\n");
    let _ = h.flush();
}

extern "C" fn host_caps(_data: *mut c_void, out_len: *mut usize) -> *const c_char {
    unsafe { *out_len = CAPS.len() };
    CAPS.as_ptr() as *const c_char
}

extern "C" fn host_time_ms(_data: *mut c_void, epoch: i32) -> f64 {
    if epoch != 0 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    } else {
        let r = runner_ptr();
        if r.is_null() {
            0.0
        } else {
            unsafe {
                let start = &(*r).start;
                start.elapsed().as_secs_f64() * 1000.0
            }
        }
    }
}

/// Call `__merseyBridge[method](args...)` with string args; the reply string
/// lands in the runner's scratch buffer (valid until the next host call).
unsafe fn call_bridge_str(method: &CStr, args: &[&str], out_len: *mut usize) -> *const c_char {
    let r = runner_ptr();
    if r.is_null() {
        *out_len = 0;
        return ptr::null();
    }
    let reply = call_bridge_string(method, args);
    (*r).scratch = reply;
    // Explicit reference so the method calls below don't autoref off a raw deref.
    let scratch: &String = &(*r).scratch;
    *out_len = scratch.len();
    scratch.as_ptr() as *const c_char
}

/// The shared body: fetch `__merseyBridge`, call `method`, return its string.
unsafe fn call_bridge_string(method: &CStr, args: &[&str]) -> String {
    let r = runner_ptr();
    if r.is_null() {
        return String::new();
    }
    let Some(mut cx) = JSContext::get_from_thread() else {
        return String::new();
    };
    let cx = &mut cx;
    let global = HandleObject::from_marked_location(&(*r).global);

    rooted!(&in(cx) let mut bridge_val = UndefinedValue());
    if !JS_GetProperty(cx, global, c"__merseyBridge".as_ptr(), bridge_val.handle_mut())
        || !bridge_val.is_object()
    {
        return String::new();
    }
    rooted!(&in(cx) let bridge_obj = bridge_val.to_object());

    rooted_vec!(let mut argv);
    for a in args {
        let chars = Utf8Chars::from(*a);
        let js = JS_NewStringCopyUTF8N(cx, &*chars as *const _);
        if js.is_null() {
            return String::new();
        }
        rooted!(&in(cx) let v = StringValue(&*js));
        argv.push(v.get());
    }

    rooted!(&in(cx) let mut rval = UndefinedValue());
    let hva = js::jsapi::HandleValueArray::from(&argv);
    let ok = JS_CallFunctionName(cx, bridge_obj.handle(), method.as_ptr(), &hva, rval.handle_mut());
    if ok && rval.is_string() {
        jsstr_to_string(cx, NonNull::new(rval.to_string()).unwrap())
    } else {
        String::new()
    }
}

/// Number-returning bridge call (`global`, `instanceOf`).
unsafe fn call_bridge_int(method: &CStr, args: &[&str]) -> i64 {
    let r = runner_ptr();
    if r.is_null() {
        return -1;
    }
    let Some(mut cx) = JSContext::get_from_thread() else {
        return -1;
    };
    let cx = &mut cx;
    let global = HandleObject::from_marked_location(&(*r).global);
    rooted!(&in(cx) let mut bridge_val = UndefinedValue());
    if !JS_GetProperty(cx, global, c"__merseyBridge".as_ptr(), bridge_val.handle_mut())
        || !bridge_val.is_object()
    {
        return -1;
    }
    rooted!(&in(cx) let bridge_obj = bridge_val.to_object());
    rooted_vec!(let mut argv);
    for a in args {
        let chars = Utf8Chars::from(*a);
        let js = JS_NewStringCopyUTF8N(cx, &*chars as *const _);
        if js.is_null() {
            return -1;
        }
        rooted!(&in(cx) let v = StringValue(&*js));
        argv.push(v.get());
    }
    rooted!(&in(cx) let mut rval = UndefinedValue());
    let hva = js::jsapi::HandleValueArray::from(&argv);
    let ok = JS_CallFunctionName(cx, bridge_obj.handle(), method.as_ptr(), &hva, rval.handle_mut());
    if ok && rval.is_number() {
        rval.to_number() as i64
    } else {
        -1
    }
}

// ---- interned + scalar fast paths (ABI v3) --------------------------------
// A member name crosses the boundary once (web_intern), then only its integer
// id does, and scalar arguments cross as JS values — no per-call args JSON to
// build and parse. The bridge already implements the matching methods
// (intern / getId / callStr / callScalars / newScalars); these just forward,
// passing numbers as numbers and strings as strings instead of a JSON blob.

enum JsArg<'a> {
    Num(f64),
    Str(&'a str),
}

/// Call `__merseyBridge[method](args...)` with mixed number/string args; return
/// the reply string (empty on failure). Shared by the string-returning and
/// number-returning wrappers below.
unsafe fn call_bridge_vals(method: &CStr, args: &[JsArg]) -> Option<String> {
    let r = runner_ptr();
    if r.is_null() {
        return None;
    }
    let Some(mut cx) = JSContext::get_from_thread() else {
        return None;
    };
    let cx = &mut cx;
    let global = HandleObject::from_marked_location(&(*r).global);
    rooted!(&in(cx) let mut bridge_val = UndefinedValue());
    if !JS_GetProperty(cx, global, c"__merseyBridge".as_ptr(), bridge_val.handle_mut())
        || !bridge_val.is_object()
    {
        return None;
    }
    rooted!(&in(cx) let bridge_obj = bridge_val.to_object());
    rooted_vec!(let mut argv);
    for a in args {
        match a {
            JsArg::Num(n) => {
                rooted!(&in(cx) let v = DoubleValue(*n));
                argv.push(v.get());
            },
            JsArg::Str(s) => {
                let chars = Utf8Chars::from(*s);
                let js = JS_NewStringCopyUTF8N(cx, &*chars as *const _);
                if js.is_null() {
                    return None;
                }
                rooted!(&in(cx) let v = StringValue(&*js));
                argv.push(v.get());
            },
        }
    }
    rooted!(&in(cx) let mut rval = UndefinedValue());
    let hva = js::jsapi::HandleValueArray::from(&argv);
    if !JS_CallFunctionName(cx, bridge_obj.handle(), method.as_ptr(), &hva, rval.handle_mut()) {
        return None;
    }
    if rval.is_string() {
        Some(jsstr_to_string(cx, NonNull::new(rval.to_string()).unwrap()))
    } else if rval.is_number() {
        // Numeric reply (intern) — encode so the number-returning wrapper reads it.
        Some(rval.to_number().to_string())
    } else {
        Some(String::new())
    }
}

/// Store a reply into the runner scratch and return its pointer (ABI lifetime).
unsafe fn reply(method: &CStr, args: &[JsArg], out_len: *mut usize) -> *const c_char {
    let r = runner_ptr();
    if r.is_null() {
        *out_len = 0;
        return ptr::null();
    }
    (*r).scratch = call_bridge_vals(method, args).unwrap_or_default();
    let scratch: &String = &(*r).scratch;
    *out_len = scratch.len();
    scratch.as_ptr() as *const c_char
}

extern "C" fn host_web_intern(_data: *mut c_void, name: *const c_char, len: usize) -> u32 {
    let name = unsafe { str_from(name, len) };
    let id = match unsafe { call_bridge_vals(c"intern", &[JsArg::Str(name)]) } {
        Some(s) => s.parse::<f64>().map(|n| n as u32).unwrap_or(u32::MAX),
        None => u32::MAX,
    };
    // Classify the name once, so the wide-path hooks can dispatch a hot method
    // to its direct-Rust path by id (the engine interns before it ever calls).
    let r = runner_ptr();
    if !r.is_null() && id != u32::MAX {
        let hot = match name {
            "URL" => Hot::CtorUrl,
            "pathname" => Hot::Pathname,
            "search" => Hot::Search,
            "createElement" => Hot::CreateElement,
            "appendChild" => Hot::AppendChild,
            "textContent" => Hot::TextContent,
            "Event" => Hot::CtorEvent,
            "dispatchEvent" => Hot::DispatchEvent,
            "className" => Hot::ClassName,
            "classList" => Hot::ClassList,
            "contains" => Hot::Contains,
            "style" => Hot::Style,
            "setProperty" => Hot::SetProperty,
            "getPropertyValue" => Hot::GetPropertyValue,
            "querySelectorAll" => Hot::QuerySelectorAll,
            "length" => Hot::Length,
            "getRandomValues" => Hot::GetRandomValues,
            "encode" => Hot::Encode,
            "decode" => Hot::Decode,
            "getItem" => Hot::GetItem,
            "setItem" => Hot::SetItem,
            "removeItem" => Hot::RemoveItem,
            // A digit-only name is an indexed access (`nodes[i]`) crossing as a
            // property read; dispatch it to NodeList::Item by value.
            n if !n.is_empty() && n.len() <= 9 && n.bytes().all(|b| b.is_ascii_digit()) => {
                Hot::Index(n.parse::<u32>().unwrap_or(0))
            },
            _ => Hot::None,
        };
        unsafe {
            let hots = &mut (*r).hot;
            while hots.len() <= id as usize {
                hots.push(Hot::None);
            }
            hots[id as usize] = hot;
        }
    }
    id
}

extern "C" fn host_web_get_id(
    _data: *mut c_void,
    target: i64,
    name_id: u32,
    out_len: *mut usize,
) -> *const c_char {
    unsafe { reply(c"getId", &[JsArg::Num(target as f64), JsArg::Num(name_id as f64)], out_len) }
}

extern "C" fn host_web_call_str(
    _data: *mut c_void,
    target: i64,
    name_id: u32,
    arg: *const c_char,
    arg_len: usize,
    out_len: *mut usize,
) -> *const c_char {
    let arg = unsafe { str_from(arg, arg_len) };
    unsafe {
        reply(
            c"callStr",
            &[JsArg::Num(target as f64), JsArg::Num(name_id as f64), JsArg::Str(arg)],
            out_len,
        )
    }
}

unsafe fn scalar_args<'a>(lead: &[JsArg<'a>], scalars: &'a [MsyScalar]) -> Vec<JsArg<'a>> {
    let mut v: Vec<JsArg> = lead.iter().map(|a| match a {
        JsArg::Num(n) => JsArg::Num(*n),
        JsArg::Str(s) => JsArg::Str(s),
    }).collect();
    for s in scalars {
        if s.is_num != 0 {
            v.push(JsArg::Num(s.num));
        } else {
            v.push(JsArg::Str(str_from(s.str_ptr, s.str_len)));
        }
    }
    v
}

extern "C" fn host_web_call_scalars(
    _data: *mut c_void,
    target: i64,
    name_id: u32,
    args: *const MsyScalar,
    argc: usize,
    out_len: *mut usize,
) -> *const c_char {
    unsafe {
        let scalars = if args.is_null() { &[][..] } else { std::slice::from_raw_parts(args, argc) };
        let v = scalar_args(&[JsArg::Num(target as f64), JsArg::Num(name_id as f64)], scalars);
        reply(c"callScalars", &v, out_len)
    }
}

extern "C" fn host_web_new_scalars(
    _data: *mut c_void,
    ctor_id: u32,
    args: *const MsyScalar,
    argc: usize,
    out_len: *mut usize,
) -> *const c_char {
    unsafe {
        let scalars = if args.is_null() { &[][..] } else { std::slice::from_raw_parts(args, argc) };
        let v = scalar_args(&[JsArg::Num(ctor_id as f64)], scalars);
        reply(c"newScalars", &v, out_len)
    }
}

extern "C" fn host_web_global(_data: *mut c_void, name: *const c_char, len: usize) -> i64 {
    let name = unsafe { str_from(name, len) };
    unsafe { call_bridge_int(c"global", &[name]) }
}

extern "C" fn host_web_get(
    _data: *mut c_void,
    target: i64,
    prop: *const c_char,
    prop_len: usize,
    out_len: *mut usize,
) -> *const c_char {
    let prop = unsafe { str_from(prop, prop_len) };
    let t = target.to_string();
    unsafe { call_bridge_str(c"get", &[&t, prop], out_len) }
}

extern "C" fn host_web_set(
    _data: *mut c_void,
    target: i64,
    prop: *const c_char,
    prop_len: usize,
    value_json: *const c_char,
    value_len: usize,
    out_len: *mut usize,
) -> *const c_char {
    let prop = unsafe { str_from(prop, prop_len) };
    let value = unsafe { str_from(value_json, value_len) };
    let t = target.to_string();
    unsafe { call_bridge_str(c"set", &[&t, prop, value], out_len) }
}

extern "C" fn host_web_call(
    _data: *mut c_void,
    target: i64,
    method: *const c_char,
    method_len: usize,
    args_json: *const c_char,
    args_len: usize,
    out_len: *mut usize,
) -> *const c_char {
    let method = unsafe { str_from(method, method_len) };
    let args = unsafe { str_from(args_json, args_len) };
    let t = target.to_string();
    unsafe { call_bridge_str(c"call", &[&t, method, args], out_len) }
}

extern "C" fn host_web_new(
    _data: *mut c_void,
    ctor: *const c_char,
    ctor_len: usize,
    args_json: *const c_char,
    args_len: usize,
    out_len: *mut usize,
) -> *const c_char {
    let ctor = unsafe { str_from(ctor, ctor_len) };
    let args = unsafe { str_from(args_json, args_len) };
    unsafe { call_bridge_str(c"construct", &[ctor, args], out_len) }
}

extern "C" fn host_web_iterate(
    _data: *mut c_void,
    target: i64,
    out_len: *mut usize,
) -> *const c_char {
    let t = target.to_string();
    unsafe { call_bridge_str(c"iterate", &[&t], out_len) }
}

extern "C" fn host_web_instanceof(_data: *mut c_void, target: i64, ctor: i64) -> i32 {
    let t = target.to_string();
    let c = ctor.to_string();
    unsafe { call_bridge_int(c"instanceOf", &[&t, &c]) as i32 }
}

extern "C" fn host_web_release(_data: *mut c_void, target: i64) {
    // Drop any cached native for this handle before the bridge forgets it.
    let r = runner_ptr();
    if !r.is_null() {
        unsafe { (&mut (*r).natives).remove(&target) };
    }
    let t = target.to_string();
    let mut dummy: usize = 0;
    unsafe { call_bridge_str(c"release", &[&t], &mut dummy) };
}

unsafe fn str_from<'a>(p: *const c_char, len: usize) -> &'a str {
    if p.is_null() || len == 0 {
        return "";
    }
    let bytes = std::slice::from_raw_parts(p as *const u8, len);
    std::str::from_utf8(bytes).unwrap_or("")
}

/// `mersey(source)` — the browser-console REPL: one growing, always-typechecked
/// module against this page's engine (see msy_context_repl_turn). Echoes a
/// trailing bare expression; a rejected turn's diagnostics throw.
unsafe extern "C" fn mersey_repl(cx: *mut js::jsapi::JSContext, argc: u32, vp: *mut Value) -> bool {
    let args = CallArgs::from_vp(vp, argc);
    let r = runner_ptr();
    if r.is_null() || argc < 1 || !args.get(0).is_string() {
        args.rval().set(UndefinedValue());
        return true;
    }
    let mut cx_safe = JSContext::from_ptr(NonNull::new(cx).unwrap());
    let source = jsstr_to_string(&cx_safe, NonNull::new(args.get(0).to_string()).unwrap());
    let mut out_len = 0usize;
    let reply = msy_context_repl_turn(
        (*r).ctx,
        source.as_ptr() as *const c_char,
        source.len(),
        &mut out_len,
    );
    let text = if reply.is_null() {
        String::new()
    } else {
        String::from_utf8_lossy(std::slice::from_raw_parts(reply as *const u8, out_len)).to_string()
    };
    if let Some(diags) = text.strip_prefix('!') {
        let msg = std::ffi::CString::new(diags.replace('%', "%%")).unwrap_or_default();
        js::jsapi::JS_ReportErrorUTF8(cx, msg.as_ptr());
        return false;
    }
    if text.is_empty() {
        args.rval().set(UndefinedValue());
        return true;
    }
    let chars = js::conversions::Utf8Chars::from(text.as_str());
    let jsstr = JS_NewStringCopyUTF8N(&mut cx_safe, &*chars as *const _);
    if jsstr.is_null() {
        args.rval().set(UndefinedValue());
    } else {
        args.rval().set(js::jsval::StringValue(&*jsstr));
    }
    true
}

/// `__merseyInvoke(cb, argsJson)` — the hook the bridge calls when JS invokes a
/// Mersey closure (a promise reaction, an event listener). Forwards into the
/// engine via `msy_context_invoke_args`.
unsafe extern "C" fn mersey_invoke(cx: *mut js::jsapi::JSContext, argc: u32, vp: *mut Value) -> bool {
    let args = CallArgs::from_vp(vp, argc);
    let r = runner_ptr();
    if !r.is_null() && argc >= 2 {
        let cb = args.get(0).to_number() as u32;
        let arg1 = args.get(1);
        let args_json = if arg1.is_string() {
            let cx = JSContext::from_ptr(NonNull::new(cx).unwrap());
            jsstr_to_string(&cx, NonNull::new(arg1.to_string()).unwrap())
        } else {
            String::from("[]")
        };
        let bytes = args_json.as_bytes();
        msy_context_invoke_args((*r).ctx, cb, bytes.as_ptr() as *const c_char, bytes.len());
    }
    args.rval().set(UndefinedValue());
    true
}

// ---- wide-string fast paths (ABI v5) --------------------------------------
// The fastest reflective tier: the bridge's *Wide methods return the raw value
// (a scalar as itself, a host object as {r: handle}, an array as {j: json}), so
// there is NO JSON on args or replies, and object arguments (appendChild(el),
// getRandomValues(buf)) stay on this path via a refs bitmask instead of falling
// back to the JSON `call`. Reply strings are re-encoded to UTF-16 for the typed
// MsyReply; the args reuse the UTF-8 string plumbing above (SpiderMonkey does not
// care what encoding the JS string was built from) — same lever as the Ladybird
// fork's web_*_u16, whose LibJS is UTF-16 end to end.

/// Stash a reply string as UTF-16 in the runner buffer and point `out` at it.
unsafe fn fill_str16(r: *mut Runner, out: *mut MsyReply, tag: i32, s: &str) {
    (*r).reply16 = s.encode_utf16().collect();
    (*out).tag = tag;
    (*out).str16 = (*r).reply16.as_ptr();
    (*out).str16_len = (*r).reply16.len();
}

/// Call `__merseyBridge[method](lead…, args…)` and type the raw reply into `out`.
unsafe fn bridge_wide(method: &CStr, lead: &[f64], args: &[MsyArg16], out: *mut MsyReply) {
    *out = MsyReply::default();
    let r = runner_ptr();
    if r.is_null() {
        return;
    }
    let Some(mut cx) = JSContext::get_from_thread() else {
        return;
    };
    let cx = &mut cx;
    let global = HandleObject::from_marked_location(&(*r).global);

    rooted!(&in(cx) let mut bridge_val = UndefinedValue());
    if !JS_GetProperty(cx, global, c"__merseyBridge".as_ptr(), bridge_val.handle_mut())
        || !bridge_val.is_object()
    {
        return;
    }
    rooted!(&in(cx) let bridge_obj = bridge_val.to_object());

    // Own any UTF-16 string args as Rust strings for the duration of the call.
    let mut owned: Vec<String> = Vec::new();
    for a in args {
        if a.kind == 0 {
            owned.push(String::from_utf16_lossy(std::slice::from_raw_parts(a.str16, a.str16_len)));
        }
    }

    rooted_vec!(let mut argv);
    for n in lead {
        rooted!(&in(cx) let v = DoubleValue(*n));
        argv.push(v.get());
    }
    let mut oi = 0usize;
    for a in args {
        match a.kind {
            0 => {
                let s = &owned[oi];
                oi += 1;
                let chars = Utf8Chars::from(s.as_str());
                let js = JS_NewStringCopyUTF8N(cx, &*chars as *const _);
                if js.is_null() {
                    return;
                }
                rooted!(&in(cx) let v = StringValue(&*js));
                argv.push(v.get());
            },
            3 => {
                rooted!(&in(cx) let v = BooleanValue(a.num != 0.0));
                argv.push(v.get());
            },
            4 => {
                rooted!(&in(cx) let v = NullValue());
                argv.push(v.get());
            },
            // number (1) or host-object handle (2): both cross as a number.
            _ => {
                rooted!(&in(cx) let v = DoubleValue(a.num));
                argv.push(v.get());
            },
        }
    }

    rooted!(&in(cx) let mut rval = UndefinedValue());
    let hva = js::jsapi::HandleValueArray::from(&argv);
    if !JS_CallFunctionName(cx, bridge_obj.handle(), method.as_ptr(), &hva, rval.handle_mut()) {
        (*out).tag = 5; // the *Wide methods throw on error
        return;
    }

    if rval.is_null() || rval.is_undefined() {
        (*out).tag = 0;
    } else if rval.is_boolean() {
        (*out).tag = 4;
        (*out).num = if rval.to_boolean() { 1.0 } else { 0.0 };
    } else if rval.is_number() {
        (*out).tag = 1;
        (*out).num = rval.to_number();
    } else if rval.is_string() {
        let s = jsstr_to_string(cx, NonNull::new(rval.to_string()).unwrap());
        fill_str16(r, out, 2, &s);
    } else if rval.is_object() {
        rooted!(&in(cx) let obj = rval.to_object());
        rooted!(&in(cx) let mut refv = UndefinedValue());
        rooted!(&in(cx) let mut jsonv = UndefinedValue());
        if JS_GetProperty(cx, obj.handle(), c"r".as_ptr(), refv.handle_mut()) && refv.is_number() {
            (*out).tag = 3;
            (*out).num = refv.to_number();
        } else if JS_GetProperty(cx, obj.handle(), c"j".as_ptr(), jsonv.handle_mut())
            && jsonv.is_string()
        {
            let s = jsstr_to_string(cx, NonNull::new(jsonv.to_string()).unwrap());
            fill_str16(r, out, 7, &s);
        } else {
            (*out).tag = 0;
        }
    } else {
        (*out).tag = 0;
    }
}

fn refs_mask(args: &[MsyArg16]) -> f64 {
    let mut mask = 0u32;
    for (i, a) in args.iter().enumerate() {
        if a.kind == 2 && i < 32 {
            mask |= 1u32 << i;
        }
    }
    mask as f64
}

/// Which args are stable callback ids (kind 5, ABI v8) — the bridge resolves
/// them to its cached wrapper functions.
fn cb_mask(args: &[MsyArg16]) -> f64 {
    let mut mask = 0u32;
    for (i, a) in args.iter().enumerate() {
        if a.kind == 5 && i < 32 {
            mask |= 1u32 << i;
        }
    }
    mask as f64
}

// ---- direct-DOM tier (the Ladybird fork's HotMethod dispatch, in Rust) ------
// A hot method unwraps its receiver (and object args) to the Servo DOM native
// once — the native is boxed and never moves; the bridge's handle table keeps
// it alive until web_release — and calls the DOM method directly, with the
// reflective wide path as fall-back on any type mismatch.

/// A UTF-16 string argument as an owned Rust string.
unsafe fn arg16_str(a: &MsyArg16) -> String {
    String::from_utf16_lossy(std::slice::from_raw_parts(a.str16, a.str16_len))
}

/// `__merseyBridge.handleObj(handle)` — the JS object a handle names.
unsafe fn bridge_handle_obj(cx: &mut JSContext, handle: i64) -> Option<*mut JSObject> {
    let r = runner_ptr();
    if r.is_null() {
        return None;
    }
    let global = HandleObject::from_marked_location(&(*r).global);
    rooted!(&in(cx) let mut bridge_val = UndefinedValue());
    if !JS_GetProperty(cx, global, c"__merseyBridge".as_ptr(), bridge_val.handle_mut())
        || !bridge_val.is_object()
    {
        return None;
    }
    rooted!(&in(cx) let bridge_obj = bridge_val.to_object());
    rooted_vec!(let mut argv);
    rooted!(&in(cx) let v = DoubleValue(handle as f64));
    argv.push(v.get());
    rooted!(&in(cx) let mut rval = UndefinedValue());
    let hva = js::jsapi::HandleValueArray::from(&argv);
    if !JS_CallFunctionName(cx, bridge_obj.handle(), c"handleObj".as_ptr(), &hva, rval.handle_mut())
        || !rval.is_object()
    {
        return None;
    }
    Some(rval.to_object())
}

/// `__merseyBridge.register(obj)` — keep a host-created object alive in the
/// handle table and get its (deduped) handle back.
unsafe fn bridge_register_object(cx: &mut JSContext, obj: *mut JSObject) -> i64 {
    let r = runner_ptr();
    if r.is_null() {
        return -1;
    }
    let global = HandleObject::from_marked_location(&(*r).global);
    rooted!(&in(cx) let mut bridge_val = UndefinedValue());
    if !JS_GetProperty(cx, global, c"__merseyBridge".as_ptr(), bridge_val.handle_mut())
        || !bridge_val.is_object()
    {
        return -1;
    }
    rooted!(&in(cx) let bridge_obj = bridge_val.to_object());
    rooted_vec!(let mut argv);
    rooted!(&in(cx) let v = ObjectValue(obj));
    argv.push(v.get());
    rooted!(&in(cx) let mut rval = UndefinedValue());
    let hva = js::jsapi::HandleValueArray::from(&argv);
    if !JS_CallFunctionName(cx, bridge_obj.handle(), c"register".as_ptr(), &hva, rval.handle_mut())
        || !rval.is_number()
    {
        return -1;
    }
    rval.to_number() as i64
}

/// Register a host-created native and reply with its ref; also pre-cache the
/// native pointer under the new handle via `fill`.
unsafe fn reply_ref<T: DomObject>(
    cx: &mut JSContext,
    root: &T,
    fill: impl FnOnce(&mut Natives, *const T),
    out: *mut MsyReply,
) -> bool {
    let obj = root.reflector().get_jsobject().get();
    let h = bridge_register_object(cx, obj);
    if h < 0 {
        return false;
    }
    let r = runner_ptr();
    fill((&mut (*r).natives).entry(h).or_default(), root as *const T);
    *out = MsyReply::default();
    (*out).tag = 3; // ref
    (*out).num = h as f64;
    true
}

macro_rules! resolver {
    ($fn_name:ident, $ty:ty, $field:ident) => {
        /// Resolve a handle to its native, cached per handle (invalidated on
        /// web_release). The returned reference is valid for the current host
        /// call: the handle table keeps the native alive.
        unsafe fn $fn_name<'a>(cx: &mut JSContext, handle: i64) -> Option<&'a $ty> {
            let r = runner_ptr();
            if r.is_null() {
                return None;
            }
            if let Some(n) = (&(*r).natives).get(&handle) {
                if !n.$field.is_null() {
                    return Some(&*n.$field);
                }
            }
            let obj = bridge_handle_obj(cx, handle)?;
            let root: DomRoot<$ty> = root_from_object::<$ty>(cx, obj).ok()?;
            let p: *const $ty = &*root;
            (&mut (*r).natives).entry(handle).or_default().$field = p;
            Some(&*p)
        }
    };
}

resolver!(resolve_url, URL, url);
resolver!(resolve_document, Document, document);
resolver!(resolve_element, Element, element);
resolver!(resolve_html, HTMLElement, html);
resolver!(resolve_node, Node, node);
resolver!(resolve_node_list, NodeList, node_list);
resolver!(resolve_token_list, DOMTokenList, token_list);
resolver!(resolve_style, CSSStyleDeclaration, style);
resolver!(resolve_event, Event, event);
resolver!(resolve_target, EventTarget, target);
resolver!(resolve_storage, Storage, storage);
resolver!(resolve_crypto, Crypto, crypto);
resolver!(resolve_encoder, TextEncoder, encoder);
resolver!(resolve_decoder, TextDecoder, decoder);
resolver!(resolve_canvas2d, CanvasRenderingContext2D, canvas2d);

unsafe fn reply_none(out: *mut MsyReply) {
    *out = MsyReply::default();
    (*out).tag = 0;
}

unsafe fn reply_bool(out: *mut MsyReply, b: bool) {
    *out = MsyReply::default();
    (*out).tag = 4;
    (*out).num = if b { 1.0 } else { 0.0 };
}

// new URL(str): parse and build the DOM URL directly, register it, and hand
// back a ref — the pathname/search reads then hit the native cache.
unsafe fn try_ctor_url(cx: &mut JSContext, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 0 {
        return false;
    }
    let r = runner_ptr();
    let global = GlobalScope::from_object((*r).global);
    let Ok(url) = <URL as URLMethods<crate::DomTypeHolder>>::Constructor(
        cx,
        &global,
        None,
        USVString(arg16_str(&args[0])),
        None,
    ) else {
        return false;
    };
    reply_ref(cx, &*url, |n, p| n.url = p, out)
}

unsafe fn try_url_get(cx: &mut JSContext, target: i64, which: Hot, out: *mut MsyReply) -> bool {
    let Some(url) = resolve_url(cx, target) else {
        return false;
    };
    let s = if which == Hot::Pathname { url.Pathname() } else { url.Search() };
    let r = runner_ptr();
    fill_str16(r, out, 2, &s.0);
    true
}

unsafe fn try_create_element(cx: &mut JSContext, target: i64, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 0 {
        return false;
    }
    let Some(doc) = resolve_document(cx, target) else {
        return false;
    };
    let options = StringOrElementCreationOptions::String(DOMString::new());
    let Ok(el) = doc.CreateElement(cx, DOMString::from(arg16_str(&args[0])), options) else {
        return false;
    };
    let node_ptr = el.upcast::<Node>() as *const Node;
    reply_ref(
        cx,
        &*el,
        |n, p| {
            n.element = p;
            n.node = node_ptr;
        },
        out,
    )
}

unsafe fn try_append_child(cx: &mut JSContext, target: i64, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 2 {
        return false;
    }
    let Some(parent) = resolve_node(cx, target) else {
        return false;
    };
    let Some(child) = resolve_node(cx, args[0].num as i64) else {
        return false;
    };
    if parent.AppendChild(cx, child).is_err() {
        return false;
    }
    reply_none(out);
    true
}

unsafe fn try_text_content_set(cx: &mut JSContext, target: i64, value: &MsyArg16, out: *mut MsyReply) -> bool {
    if value.kind != 0 {
        return false;
    }
    let Some(node) = resolve_node(cx, target) else {
        return false;
    };
    if node.SetTextContent(cx, Some(DOMString::from(arg16_str(value)))).is_err() {
        return false;
    }
    reply_none(out);
    true
}

unsafe fn try_text_content_get(cx: &mut JSContext, target: i64, out: *mut MsyReply) -> bool {
    let Some(node) = resolve_node(cx, target) else {
        return false;
    };
    match node.GetTextContent() {
        Some(s) => {
            let r = runner_ptr();
            fill_str16(r, out, 2, &s.str());
        },
        None => reply_none(out),
    }
    true
}

unsafe fn try_set_class_name(cx: &mut JSContext, target: i64, value: &MsyArg16, out: *mut MsyReply) -> bool {
    if value.kind != 0 {
        return false;
    }
    let Some(el) = resolve_element(cx, target) else {
        return false;
    };
    el.SetClassName(cx, DOMString::from(arg16_str(value)));
    reply_none(out);
    true
}

unsafe fn try_class_list(cx: &mut JSContext, target: i64, out: *mut MsyReply) -> bool {
    let Some(el) = resolve_element(cx, target) else {
        return false;
    };
    let list = el.ClassList(cx);
    reply_ref(cx, &*list, |n, p| n.token_list = p, out)
}

unsafe fn try_contains(cx: &mut JSContext, target: i64, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 0 {
        return false;
    }
    let Some(list) = resolve_token_list(cx, target) else {
        return false;
    };
    reply_bool(out, list.Contains(DOMString::from(arg16_str(&args[0]))));
    true
}

unsafe fn try_style(cx: &mut JSContext, target: i64, out: *mut MsyReply) -> bool {
    let Some(el) = resolve_html(cx, target) else {
        return false;
    };
    let style = el.Style(cx);
    reply_ref(cx, &*style, |n, p| n.style = p, out)
}

unsafe fn try_style_property(cx: &mut JSContext, target: i64, which: Hot, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 0 {
        return false;
    }
    let Some(style) = resolve_style(cx, target) else {
        return false;
    };
    let prop = DOMString::from(arg16_str(&args[0]));
    if which == Hot::SetProperty {
        if args.len() < 2 || args[1].kind != 0 {
            return false;
        }
        let value = DOMString::from(arg16_str(&args[1]));
        if style.SetProperty(cx, prop, value, DOMString::new()).is_err() {
            return false;
        }
        reply_none(out);
    } else {
        let s = style.GetPropertyValue(prop);
        let r = runner_ptr();
        fill_str16(r, out, 2, &s.str());
    }
    true
}

unsafe fn try_query_selector_all(cx: &mut JSContext, target: i64, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 0 {
        return false;
    }
    let Some(doc) = resolve_document(cx, target) else {
        return false;
    };
    let Ok(list) = doc.QuerySelectorAll(cx, DOMString::from(arg16_str(&args[0]))) else {
        return false;
    };
    reply_ref(cx, &*list, |n, p| n.node_list = p, out)
}

unsafe fn try_length(cx: &mut JSContext, target: i64, out: *mut MsyReply) -> bool {
    let Some(list) = resolve_node_list(cx, target) else {
        return false;
    };
    *out = MsyReply::default();
    (*out).tag = 1;
    (*out).num = list.Length() as f64;
    true
}

unsafe fn try_index(cx: &mut JSContext, target: i64, index: u32, out: *mut MsyReply) -> bool {
    let Some(list) = resolve_node_list(cx, target) else {
        return false;
    };
    match list.Item(index) {
        Some(node) => reply_ref(cx, &*node, |n, p| n.node = p, out),
        None => {
            reply_none(out);
            true
        },
    }
}

unsafe fn try_ctor_event(cx: &mut JSContext, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 0 {
        return false;
    }
    let r = runner_ptr();
    let global = GlobalScope::from_object((*r).global);
    let ev = Event::new(
        cx,
        &global,
        Atom::from(arg16_str(&args[0])),
        EventBubbles::DoesNotBubble,
        EventCancelable::NotCancelable,
    );
    reply_ref(cx, &*ev, |n, p| n.event = p, out)
}

unsafe fn try_dispatch_event(cx: &mut JSContext, target: i64, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 2 {
        return false;
    }
    let Some(t) = resolve_target(cx, target) else {
        return false;
    };
    let Some(ev) = resolve_event(cx, args[0].num as i64) else {
        return false;
    };
    let Ok(not_cancelled) = t.DispatchEvent(cx, ev) else {
        return false;
    };
    reply_bool(out, not_cancelled);
    true
}

// storage.getItem(k) / setItem(k, v) / removeItem(k): straight to Servo's
// Storage implementation — no realm entry beyond the one-time receiver unwrap.
unsafe fn try_storage_get_item(cx: &mut JSContext, target: i64, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 0 {
        return false;
    }
    let Some(st) = resolve_storage(cx, target) else {
        return false;
    };
    match st.GetItem(DOMString::from(arg16_str(&args[0]))) {
        Some(v) => {
            let r = runner_ptr();
            fill_str16(r, out, 2, &v.str());
        },
        None => reply_none(out),
    }
    true
}

unsafe fn try_storage_set_item(cx: &mut JSContext, target: i64, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.len() < 2 || args[0].kind != 0 || args[1].kind != 0 {
        return false;
    }
    let Some(st) = resolve_storage(cx, target) else {
        return false;
    };
    // A quota error falls back to the reflective path, which throws it properly.
    if st
        .SetItem(DOMString::from(arg16_str(&args[0])), DOMString::from(arg16_str(&args[1])))
        .is_err()
    {
        return false;
    }
    reply_none(out);
    true
}

unsafe fn try_storage_remove_item(cx: &mut JSContext, target: i64, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 0 {
        return false;
    }
    let Some(st) = resolve_storage(cx, target) else {
        return false;
    };
    st.RemoveItem(DOMString::from(arg16_str(&args[0])));
    reply_none(out);
    true
}

// crypto.getRandomValues(buf): unwrap the Crypto receiver and the buffer, fill
// the bytes directly in Rust.
unsafe fn try_get_random_values(cx: &mut JSContext, target: i64, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 2 {
        return false;
    }
    let Some(crypto) = resolve_crypto(cx, target) else {
        return false;
    };
    let Some(obj) = bridge_handle_obj(cx, args[0].num as i64) else {
        return false;
    };
    let Ok(view) = ArrayBufferView::from(obj) else {
        return false;
    };
    auto_root!(&in(cx) let guard = view);
    // CryptoMethods::GetRandomValues takes a &NoGC first; JSContext derefs to
    // NoGC, so pass cx straight through as the generated binding itself does.
    if crypto.GetRandomValues(cx, guard).is_err() {
        return false;
    }
    reply_none(out);
    true
}

// enc.encode(s): direct UTF-8 encode; the fresh Uint8Array is registered and
// crosses as a ref.
unsafe fn try_encode(cx: &mut JSContext, target: i64, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 0 {
        return false;
    }
    let Some(enc) = resolve_encoder(cx, target) else {
        return false;
    };
    let bytes = enc.Encode(cx, USVString(arg16_str(&args[0])));
    let obj = bytes.underlying_object().get();
    if obj.is_null() {
        return false;
    }
    let h = bridge_register_object(cx, obj);
    if h < 0 {
        return false;
    }
    *out = MsyReply::default();
    (*out).tag = 3; // ref
    (*out).num = h as f64;
    true
}

// dec.decode(bytes) with a typed-array handle: direct decode.
unsafe fn try_decode(cx: &mut JSContext, target: i64, args: &[MsyArg16], out: *mut MsyReply) -> bool {
    if args.is_empty() || args[0].kind != 2 {
        return false;
    }
    let Some(dec) = resolve_decoder(cx, target) else {
        return false;
    };
    let Some(obj) = bridge_handle_obj(cx, args[0].num as i64) else {
        return false;
    };
    let Ok(heap_view) = TypedArray::<ArrayBufferViewU8, Box<Heap<*mut JSObject>>>::from(obj) else {
        return false;
    };
    let input = ArrayBufferViewOrArrayBuffer::ArrayBufferView(RootedTraceableBox::new(heap_view));
    let Ok(text) = dec.Decode(cx, Some(input), &TextDecodeOptions::empty()) else {
        return false;
    };
    let r = runner_ptr();
    fill_str16(r, out, 2, &text.0);
    true
}

/// The hot classification of an interned id, if any.
unsafe fn hot_of(name_id: u32) -> Hot {
    let r = runner_ptr();
    if r.is_null() {
        return Hot::None;
    }
    (&(*r).hot).get(name_id as usize).copied().unwrap_or(Hot::None)
}

extern "C" fn host_web_get_u16(_d: *mut c_void, target: i64, name_id: u32, out: *mut MsyReply) {
    unsafe {
        let hot = hot_of(name_id);
        if hot != Hot::None {
            if let Some(mut cx) = JSContext::get_from_thread() {
                let cx = &mut cx;
                let handled = match hot {
                    Hot::Pathname | Hot::Search => try_url_get(cx, target, hot, out),
                    Hot::TextContent => try_text_content_get(cx, target, out),
                    Hot::ClassList => try_class_list(cx, target, out),
                    Hot::Style => try_style(cx, target, out),
                    Hot::Length => try_length(cx, target, out),
                    Hot::Index(i) => try_index(cx, target, i, out),
                    _ => false,
                };
                if handled {
                    return;
                }
            }
        }
        bridge_wide(c"getWide", &[target as f64, name_id as f64], &[], out)
    }
}

extern "C" fn host_web_set_u16(
    _d: *mut c_void,
    target: i64,
    name_id: u32,
    value: *const MsyArg16,
    out: *mut MsyReply,
) {
    unsafe {
        let slice = if value.is_null() { &[][..] } else { std::slice::from_ref(&*value) };
        if !slice.is_empty() {
            let hot = hot_of(name_id);
            if hot == Hot::TextContent || hot == Hot::ClassName {
                if let Some(mut cx) = JSContext::get_from_thread() {
                    let cx = &mut cx;
                    let handled = match hot {
                        Hot::TextContent => try_text_content_set(cx, target, &slice[0], out),
                        Hot::ClassName => try_set_class_name(cx, target, &slice[0], out),
                        _ => false,
                    };
                    if handled {
                        return;
                    }
                }
            }
        }
        bridge_wide(c"setWide", &[target as f64, name_id as f64, refs_mask(slice), cb_mask(slice)], slice, out)
    }
}

extern "C" fn host_web_call_u16(
    _d: *mut c_void,
    target: i64,
    name_id: u32,
    args: *const MsyArg16,
    argc: usize,
    out: *mut MsyReply,
) {
    unsafe {
        let a = if args.is_null() { &[][..] } else { std::slice::from_raw_parts(args, argc) };
        let hot = hot_of(name_id);
        if hot != Hot::None {
            if let Some(mut cx) = JSContext::get_from_thread() {
                let cx = &mut cx;
                let handled = match hot {
                    Hot::CreateElement => try_create_element(cx, target, a, out),
                    Hot::AppendChild => try_append_child(cx, target, a, out),
                    Hot::DispatchEvent => try_dispatch_event(cx, target, a, out),
                    Hot::Contains => try_contains(cx, target, a, out),
                    Hot::SetProperty | Hot::GetPropertyValue => {
                        try_style_property(cx, target, hot, a, out)
                    },
                    Hot::QuerySelectorAll => try_query_selector_all(cx, target, a, out),
                    Hot::GetRandomValues => try_get_random_values(cx, target, a, out),
                    Hot::Encode => try_encode(cx, target, a, out),
                    Hot::Decode => try_decode(cx, target, a, out),
                    Hot::GetItem => try_storage_get_item(cx, target, a, out),
                    Hot::SetItem => try_storage_set_item(cx, target, a, out),
                    Hot::RemoveItem => try_storage_remove_item(cx, target, a, out),
                    _ => false,
                };
                if handled {
                    return;
                }
            }
        }
        bridge_wide(c"callWide", &[target as f64, name_id as f64, refs_mask(a), cb_mask(a)], a, out)
    }
}

extern "C" fn host_web_new_u16(
    _d: *mut c_void,
    ctor_id: u32,
    args: *const MsyArg16,
    argc: usize,
    out: *mut MsyReply,
) {
    unsafe {
        let a = if args.is_null() { &[][..] } else { std::slice::from_raw_parts(args, argc) };
        let hot = hot_of(ctor_id);
        if hot != Hot::None {
            if let Some(mut cx) = JSContext::get_from_thread() {
                let cx = &mut cx;
                let handled = match hot {
                    Hot::CtorUrl => try_ctor_url(cx, a, out),
                    Hot::CtorEvent => try_ctor_event(cx, a, out),
                    _ => false,
                };
                if handled {
                    return;
                }
            }
        }
        bridge_wide(c"newWide", &[ctor_id as f64, refs_mask(a), cb_mask(a)], a, out)
    }
}

// ---- typed-binding fast path (ABI v7, web_bind) ----------------------------
// The leanest tier: a JIT-compiled numeric web method (the canvas draw loop)
// crosses as a compile-time bind id plus raw f64s — no interned name, no
// MsyArg16 marshalling — and dispatches straight to Servo's
// CanvasRenderingContext2D methods. Ids must match MSY_BIND_* in
// crates/mersey_capi/include/mersey.h.
const BIND_CANVAS2D_FILLRECT: u32 = 1;
const BIND_CANVAS2D_CLEARRECT: u32 = 2;
const BIND_CANVAS2D_STROKERECT: u32 = 3;
const BIND_CANVAS2D_RECT: u32 = 4;
const BIND_CANVAS2D_MOVETO: u32 = 5;
const BIND_CANVAS2D_LINETO: u32 = 6;
const BIND_CANVAS2D_TRANSLATE: u32 = 7;
const BIND_CANVAS2D_SCALE: u32 = 8;
const BIND_CANVAS2D_ROTATE: u32 = 9;

fn bind_method_name(id: u32) -> &'static str {
    match id {
        BIND_CANVAS2D_FILLRECT => "fillRect",
        BIND_CANVAS2D_CLEARRECT => "clearRect",
        BIND_CANVAS2D_STROKERECT => "strokeRect",
        BIND_CANVAS2D_RECT => "rect",
        BIND_CANVAS2D_MOVETO => "moveTo",
        BIND_CANVAS2D_LINETO => "lineTo",
        BIND_CANVAS2D_TRANSLATE => "translate",
        BIND_CANVAS2D_SCALE => "scale",
        BIND_CANVAS2D_ROTATE => "rotate",
        _ => "",
    }
}

extern "C" fn host_web_bind(
    _d: *mut c_void,
    target: i64,
    bind_id: u32,
    args: *const f64,
    argc: usize,
    out: *mut MsyReply,
) {
    unsafe {
        *out = MsyReply::default();
        let a = |i: usize| if !args.is_null() && i < argc { *args.add(i) } else { 0.0 };
        if let Some(mut cx) = JSContext::get_from_thread() {
            let cx = &mut cx;
            if let Some(ctx2d) = resolve_canvas2d(cx, target) {
                match bind_id {
                    BIND_CANVAS2D_FILLRECT => return ctx2d.FillRect(a(0), a(1), a(2), a(3)),
                    BIND_CANVAS2D_CLEARRECT => return ctx2d.ClearRect(a(0), a(1), a(2), a(3)),
                    BIND_CANVAS2D_STROKERECT => return ctx2d.StrokeRect(a(0), a(1), a(2), a(3)),
                    BIND_CANVAS2D_RECT => return ctx2d.Rect(a(0), a(1), a(2), a(3)),
                    BIND_CANVAS2D_MOVETO => return ctx2d.MoveTo(a(0), a(1)),
                    BIND_CANVAS2D_LINETO => return ctx2d.LineTo(a(0), a(1)),
                    BIND_CANVAS2D_TRANSLATE => return ctx2d.Translate(a(0), a(1)),
                    BIND_CANVAS2D_SCALE => return ctx2d.Scale(a(0), a(1)),
                    BIND_CANVAS2D_ROTATE => return ctx2d.Rotate(a(0)),
                    _ => {},
                }
            }
        }
        // Receiver is not a canvas context (or an unknown id): reflective call
        // under the method's real name. Never hit by the canvas workload.
        let name = bind_method_name(bind_id);
        if name.is_empty() {
            return;
        }
        let mut json = String::from("[");
        for i in 0..argc {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&a(i).to_string());
        }
        json.push(']');
        let t = target.to_string();
        let mut dummy: usize = 0;
        let _ = call_bridge_str(c"call", &[&t, name, &json], &mut dummy);
    }
}

fn host_table() -> MsyHostTable {
    MsyHostTable {
        data: ptr::null_mut(),
        print: Some(host_print),
        print_level: None,
        error: None,
        caps: Some(host_caps),
        web_global: Some(host_web_global),
        web_get: Some(host_web_get),
        web_set: Some(host_web_set),
        web_call: Some(host_web_call),
        web_new: Some(host_web_new),
        web_iterate: Some(host_web_iterate),
        web_instanceof: Some(host_web_instanceof),
        web_release: Some(host_web_release),
        web_bytes_read: None,
        web_bytes_write: None,
        time_ms: Some(host_time_ms),
        random_bytes: None,
        dom_set_text: None,
        dom_get_text: None,
        dom_add_listener: None,
        // Interned + scalar fast paths: a name crosses once as an id, scalar args
        // cross as JS values (no per-call args JSON). Ops these don't cover
        // (object args, property sets) fall back to the reflective ops above.
        web_intern: Some(host_web_intern),
        web_get_id: Some(host_web_get_id),
        web_set_str: None,
        web_set_num: None,
        web_call_str: Some(host_web_call_str),
        web_call_scalars: Some(host_web_call_scalars),
        web_new_scalars: Some(host_web_new_scalars),
        // Wide-string fast paths (UTF-16, no JSON): the fastest reflective tier,
        // and the one that keeps object-argument calls (appendChild,
        // getRandomValues) and property sets (textContent=) off the JSON path.
        web_get_u16: Some(host_web_get_u16),
        web_set_u16: Some(host_web_set_u16),
        web_call_u16: Some(host_web_call_u16),
        web_new_u16: Some(host_web_new_u16),
        // Typed-binding fast path: the JIT-compiled canvas loop calls this with
        // raw f64s; we dispatch straight to CanvasRenderingContext2D.
        web_bind: Some(host_web_bind),
    }
}

/// Create the engine context once per thread.
unsafe fn ensure_runner(global_obj: *mut JSObject) -> *mut Runner {
    let existing = runner_ptr();
    if !existing.is_null() {
        (*existing).global = global_obj;
        return existing;
    }
    let runner = Box::into_raw(Box::new(Runner {
        ctx: ptr::null_mut(),
        global: global_obj,
        bridge_ready: false,
        scratch: String::new(),
        reply16: Vec::new(),
        hot: Vec::new(),
        natives: HashMap::new(),
        start: Instant::now(),
    }));
    RUNNER.with(|c| c.set(runner));

    let mut table = host_table();
    table.data = runner as *mut c_void;
    (*runner).ctx = msy_context_new(&table);
    runner
}

/// Run one inline `<script type="text/mersey">` body in the engine.
pub(crate) fn run_mersey_script(global: &GlobalScope, cx: &mut JSContext, source: &str) {
    let mut realm = enter_auto_realm(cx, global);
    let cx = &mut realm.current_realm();
    let global_obj = global.reflector().get_jsobject().get();
    unsafe {
        let runner = ensure_runner(global_obj);
        if runner.is_null() || (*runner).ctx.is_null() {
            return;
        }
        // Inject __merseyInvoke and evaluate the reflective bridge, once.
        if !(*runner).bridge_ready {
            let global_handle = HandleObject::from_marked_location(&(*runner).global);
            let name = c"__merseyInvoke";
            let _ = JS_DefineFunction(cx, global_handle, name.as_ptr(), Some(mersey_invoke), 2, 0);
            let _ = JS_DefineFunction(cx, global_handle, c"mersey".as_ptr(), Some(mersey_repl), 1, 0);
            let _ = global.evaluate_js_on_global(
                cx,
                Cow::Borrowed(BRIDGE_JS),
                "mersey-bridge.js",
                None,
                None,
            );
            (*runner).bridge_ready = true;
        }
        let src = source.as_bytes();
        // The mersey program is a running script: push the entry-script
        // settings (spec "prepare to run a script") for the whole run, the
        // way evaluate_js_on_global does for JS. Without it, bindings the
        // bridge reaches reflectively that consult entry_global() —
        // Location's cross-origin getters (location.host) — panic on the
        // empty settings stack. One push per run; per-bridge-call cost: none.
        run_a_script::<crate::DomTypeHolder, _, _>(cx, global, |_cx| {
            msy_context_run((*runner).ctx, src.as_ptr() as *const c_char, src.len());
        });
    }
}
