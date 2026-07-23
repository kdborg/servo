/* Universal web bridge (shared by the browser loader and the test harness).
 *
 * This is what makes *every* web technology reachable from Mersey: instead
 * of one hand-written host function per API, five reflective operations
 * (global / get / set / call / new) reach any object in the JS realm.
 * Object identity is preserved through a handle table, Mersey closures
 * cross as real JS callbacks, and Promises are handed back as live objects
 * (so `.then(...)` works today, before `async`/`await` lands in the engine).
 *
 * Wire format (tagged JSON, matching crates/mersey_interp/src/webjson.rs):
 *   primitives  -> JSON scalars
 *   host object -> {"__ref__": handle}
 *   Mersey fn   -> {"__cb__": id}   (engine → host direction)
 *   reply       -> {"ok": value} | {"err": "message"}
 */
const CALLS = new Map(), GETS = new Map(), SETS = new Map(), CTORS = new Map();

function makeBridge(globalObject, invokeCallback) {
  const realmHTMLElement = () => globalObject.HTMLElement;
  // Generated bindings: resolve (interface, member) → direct thunk once, then
  // cache. Reflection is only the fallback for objects outside the IDL
  // corpus (plain JS objects, cross-realm values).
  const thunkCache = new Map();
  const ifaceNames = (obj) => {
    const names = [];
    for (let p = obj; p && p !== Object.prototype; p = Object.getPrototypeOf(p)) {
      const n = p.constructor && p.constructor.name;
      if (n) names.push(n);
    }
    return names;
  };
  // Escape hatch for A/B measurement (see web/test/bench.mjs).
  const useBindings = !globalObject.__MERSEY_NO_BINDINGS;
  const bound = (obj, prop, table, tag) => {
    if (!useBindings) return null;
    const ctor = obj && obj.constructor ? obj.constructor.name : "";
    const key = `${tag}|${ctor}|${prop}`;
    if (thunkCache.has(key)) return thunkCache.get(key);
    let thunk = null;
    for (const iface of ifaceNames(obj)) {
      const t = table.get(`${iface}.${prop}`);
      if (t) {
        thunk = t;
        break;
      }
    }
    thunkCache.set(key, thunk);
    return thunk;
  };
  // Generated ctor thunks reference the IDL global by its bare name; a realm
  // built over a plain object (the tests, the engine bench leg) may define
  // the constructor only on the realm, so the thunk throws ReferenceError
  // where the reflective path would succeed. Treat that as "no thunk" and
  // let the caller fall through to reflection.
  const boundCtor = (name, args) => {
    const c = useBindings ? CTORS.get(name) : null;
    if (!c) return null;
    try {
      return { v: c(args) };
    } catch (e) {
      if (e instanceof ReferenceError) return null;
      throw e;
    }
  };

  const handles = [globalObject]; // handle 0 = the global object
  const byObject = new Map([[globalObject, 0]]);

  const handleFor = (obj) => {
    let h = byObject.get(obj);
    if (h === undefined) {
      h = handles.length;
      handles.push(obj);
      byObject.set(obj, h);
    }
    return h;
  };

  const isPrimitive = (v) =>
    v === null || v === undefined || typeof v === "boolean" ||
    typeof v === "number" || typeof v === "string";

  // JS value -> tagged JSON value
  const encode = (v) => {
    if (v === undefined) return null;
    if (isPrimitive(v)) return v;
    if (typeof v === "bigint") return v.toString();
    return { __ref__: handleFor(v) };
  };

  // One wrapper per callback id. The engine keys ids by closure identity, so
  // the same Mersey function is the same JS function on every crossing —
  // `removeEventListener` removes, and a hot `setTimeout` loop doesn't
  // allocate a wrapper per call. Shared by the JSON path ({"__cb__":id}) and
  // the wide tier's cb mask (ABI v8).
  const cbWrappers = new Map();
  const cbFor = (id) => {
    const cached = cbWrappers.get(id);
    if (cached) return cached;
    // A Mersey closure: JS calls it with real arguments (event objects,
    // resolved promise values, …), which cross back as handles.
    const fn = (...args) => invokeCallback(id, JSON.stringify(args.map(encode)));
    fn.__merseyCallback = id; // so the host can release it later
    cbWrappers.set(id, fn);
    return fn;
  };

  // tagged JSON value -> JS value (callbacks become real functions)
  const decode = (v) => {
    if (v === null || typeof v !== "object") return v;
    if (Array.isArray(v)) return v.map(decode);
    if ("__ref__" in v) return handles[v.__ref__];
    if ("__cb__" in v) {
      return cbFor(v.__cb__);
    }
    const out = {};
    for (const [k, val] of Object.entries(v)) out[k] = decode(val);
    return out;
  };

  const encodeAny = (v) => (Array.isArray(v) ? v.map(encode) : encode(v));
  const ok = (value) => JSON.stringify({ ok: encodeAny(value) });
  const err = (e) => JSON.stringify({ err: String(e && e.message ? e.message : e) });

  // Wide-string path: hand the host a value it can type by inspection. A scalar
  // stays a scalar; a host object becomes {r: handle} (a ref); a top-level array
  // becomes {j: json} (rare — the host parses it). No JSON for the common cases.
  const wideResult = (v) => {
    if (v === null || v === undefined) return null;
    const t = typeof v;
    if (t === "string" || t === "number" || t === "boolean") return v;
    if (t === "bigint") return v.toString();
    if (Array.isArray(v)) return { j: JSON.stringify(encodeAny(v)) };
    return { r: handleFor(v) };
  };

  // Interned member names: a name crosses the boundary once, then it is an
  // integer id — no TextDecoder per call.
  const names = [];
  const nameIds = new Map();
  const OK_NULL = JSON.stringify({ ok: null });

  return {
    intern(name) {
      // Opt out of fast paths for measurement (see web/test/bench.mjs).
      if (globalObject.__MERSEY_NO_FASTPATH) return 0xffffffff;
      let id = nameIds.get(name);
      if (id === undefined) {
        id = names.length;
        names.push(name);
        nameIds.set(name, id);
      }
      return id;
    },
    getId(target, nameId) {
      try {
        const obj = handles[target];
        const prop = names[nameId];
        if (obj == null) return err(`stale handle ${target}`);
        const g = bound(obj, prop, GETS, "g");
        if (g) return ok(g(obj));
        const v = obj[prop];
        return ok(typeof v === "function" ? v.bind(obj) : v);
      } catch (e) {
        return err(e);
      }
    },
    setScalar(target, nameId, value) {
      try {
        const obj = handles[target];
        const prop = names[nameId];
        const s = bound(obj, prop, SETS, "s");
        if (s) s(obj, value);
        else obj[prop] = value;
        return OK_NULL;
      } catch (e) {
        return err(e);
      }
    },
    callStr(target, nameId, arg) {
      try {
        const obj = handles[target];
        const method = names[nameId];
        if (obj == null) return err(`stale handle ${target}`);
        const c = bound(obj, method, CALLS, "c");
        if (c) return ok(c(obj, [arg]));
        const fn = obj[method];
        if (typeof fn !== "function") return err(`${method} is not a function`);
        return ok(fn.call(obj, arg));
      } catch (e) {
        return err(e);
      }
    },
    // Interned multi-scalar call: target[nameId](...args), args already decoded
    // JS scalars (numbers/strings). No JSON, for setItem(k,v), fillRect(…), etc.
    callScalars(target, nameId, ...args) {
      try {
        const obj = handles[target];
        const method = names[nameId];
        if (obj == null) return err(`stale handle ${target}`);
        const c = bound(obj, method, CALLS, "c");
        if (c) return ok(c(obj, args));
        const fn = obj[method];
        if (typeof fn !== "function") return err(`${method} is not a function`);
        return ok(fn.apply(obj, args));
      } catch (e) {
        return err(e);
      }
    },
    // Interned constructor with scalar args: new names[ctorId](...args).
    newScalars(ctorId, ...args) {
      try {
        const name = names[ctorId];
        const b = boundCtor(name, args);
        if (b) return ok(b.v);
        const Ctor = name.split(".").reduce((o, k) => (o == null ? o : o[k]), globalObject);
        if (typeof Ctor !== "function") return err(`${name} is not a constructor`);
        return ok(new Ctor(...args));
      } catch (e) {
        return err(e);
      }
    },
    // Wide-string fast path: return the *raw* value (no JSON) for the host to
    // type by inspection — a scalar as itself, a host object as {r: handle}
    // (→ a ref), a top-level array as {j: json} (rare), and throw on error.
    // The host passes/receives strings as UTF-16, matching Gecko natively.
    getWide(target, nameId) {
      const obj = handles[target];
      if (obj == null) throw new Error(`stale handle ${target}`);
      const prop = names[nameId];
      const g = bound(obj, prop, GETS, "g");
      const v = g ? g(obj) : obj[prop];
      return wideResult(typeof v === "function" && !g ? v.bind(obj) : v);
    },
    // refsMask: bit i set means args[i] is a handle number — resolve it to the
    // object it names, so calls with object arguments (appendChild(el),
    // getRandomValues(buf)) stay on the wide path. cbMask (ABI v8) marks
    // stable callback ids the same way — resolved to cached wrappers.
    setWide(target, nameId, refsMask, cbMask, value) {
      const obj = handles[target];
      if (obj == null) throw new Error(`stale handle ${target}`);
      if (refsMask & 1) value = handles[value];
      else if (cbMask & 1) value = cbFor(value);
      const prop = names[nameId];
      const s = bound(obj, prop, SETS, "s");
      if (s) s(obj, value);
      else obj[prop] = value;
      return null;
    },
    callWide(target, nameId, refsMask, cbMask, ...args) {
      const obj = handles[target];
      if (obj == null) throw new Error(`stale handle ${target}`);
      for (let i = 0, m = refsMask | cbMask; m >> i; i++) {
        if ((refsMask >> i) & 1) args[i] = handles[args[i]];
        else if ((cbMask >> i) & 1) args[i] = cbFor(args[i]);
      }
      const method = names[nameId];
      // ABI v8: the interned EMPTY name — the handle is itself callable
      // (an imported `setTimeout(cb, ms)`, `fetch(url)`).
      if (method === "") {
        if (typeof obj !== "function") throw new Error("value is not a function");
        return wideResult(obj(...args));
      }
      const c = bound(obj, method, CALLS, "c");
      if (c) return wideResult(c(obj, args));
      const fn = obj[method];
      if (typeof fn !== "function") throw new Error(`${method} is not a function`);
      return wideResult(fn.apply(obj, args));
    },
    newWide(ctorId, refsMask, cbMask, ...args) {
      for (let i = 0, m = refsMask | cbMask; m >> i; i++) {
        if ((refsMask >> i) & 1) args[i] = handles[args[i]];
        else if ((cbMask >> i) & 1) args[i] = cbFor(args[i]);
      }
      const name = names[ctorId];
      const b = boundCtor(name, args);
      if (b) return wideResult(b.v);
      const Ctor = name.split(".").reduce((o, k) => (o == null ? o : o[k]), globalObject);
      if (typeof Ctor !== "function") throw new Error(`${name} is not a constructor`);
      return wideResult(new Ctor(...args));
    },
    // The raw object a handle names — the C++ host unwraps it once to a native
    // pointer (Element, canvas context) and then calls Gecko directly, no JS.
    handleObj(h) {
      return handles[h];
    },
    // Register a host-CREATED object (a DOMURL, an Element the C++ fork built
    // directly) and return its handle — keeping it alive in the handle table so
    // the engine can name it, while the host caches the native pointer it already
    // holds. The direct-DOM counterpart of `handleObj`, for the create direction.
    register(obj) {
      return handleFor(obj);
    },
    // A C++-created object (negative handle) escaping to a JS-path API: the
    // host materialized its reflector and registers it under the SAME handle,
    // so every path — lookups and re-encoding — sees one identity.
    adopt(h, obj) {
      handles[h] = obj;
      byObject.set(obj, h);
      return null;
    },
    global(name) {
      // Ambient globals only: the engine already gates this by import.
      return name in globalObject ? handleFor(globalObject[name]) : -1;
    },
    get(target, prop) {
      try {
        const obj = handles[target];
        if (obj == null) return err(`stale handle ${target}`);
        const g = bound(obj, prop, GETS, "g");
        if (g) return ok(g(obj)); // generated binding
        const v = obj[prop]; // fallback: reflection
        return ok(typeof v === "function" ? v.bind(obj) : v);
      } catch (e) {
        return err(e);
      }
    },
    set(target, prop, valueJson) {
      try {
        const obj = handles[target];
        const v = decode(JSON.parse(valueJson));
        const s = bound(obj, prop, SETS, "s");
        if (s) s(obj, v);
        else obj[prop] = v;
        return JSON.stringify({ ok: null });
      } catch (e) {
        return err(e);
      }
    },
    call(target, method, argsJson) {
      try {
        const obj = handles[target];
        if (obj == null) return err(`stale handle ${target}`);
        const args = JSON.parse(argsJson).map(decode);
        // method "" => the handle is itself callable (e.g. imported fetch)
        if (method === "") {
          if (typeof obj !== "function") return err("value is not a function");
          return ok(obj(...args));
        }
        const c = bound(obj, method, CALLS, "c");
        if (c) return ok(c(obj, args)); // generated binding
        const fn = obj[method]; // fallback: reflection
        if (typeof fn !== "function") return err(`${method} is not a function`);
        return ok(fn.apply(obj, args));
      } catch (e) {
        return err(e);
      }
    },
    /// Bulk read: a host typed array / ArrayBuffer as raw bytes.
    bytesRead(target) {
      const obj = handles[target];
      if (obj == null) return null;
      if (ArrayBuffer.isView(obj)) {
        return new Uint8Array(obj.buffer, obj.byteOffset, obj.byteLength);
      }
      if (obj instanceof ArrayBuffer) return new Uint8Array(obj);
      return null;
    },
    /// Bulk write: raw bytes → a fresh host Uint8ClampedArray (canvas-ready).
    bytesWrite(view) {
      return handleFor(new Uint8ClampedArray(view)); // copies out of wasm memory
    },
    /// `object instanceof constructor`, both sides being handles.
    instanceOf(target, ctor) {
      const obj = handles[target];
      const Ctor = handles[ctor];
      if (obj == null || typeof Ctor !== "function") return 0;
      try {
        return obj instanceof Ctor ? 1 : 0;
      } catch {
        return 0;
      }
    },
    /// Drop a handle: the object becomes collectable by the JS GC.
    release(target) {
      const obj = handles[target];
      if (obj != null) {
        byObject.delete(obj);
        handles[target] = null;
      }
    },
    /// Snapshot a host iterable as a plain array of encoded values.
    iterate(target) {
      try {
        const obj = handles[target];
        if (obj == null) return err(`stale handle ${target}`);
        let items;
        if (Array.isArray(obj)) items = obj;
        else if (typeof obj[Symbol.iterator] === "function") items = Array.from(obj);
        else if (typeof obj.length === "number") {
          // Array-likes without the iterator protocol (older collections).
          items = Array.prototype.slice.call(obj);
        } else {
          return err("value is not iterable");
        }
        return ok(items);
      } catch (e) {
        return err(e);
      }
    },
    /// Register a custom element whose lifecycle calls back into Mersey.
    /// `handlers` is a decoded record: { connected?, disconnected?,
    /// attributeChanged?, observed? }.
    defineElement(tag, handlers) {
      const h = handlers ?? {};
      const observed = Array.isArray(h.observed) ? h.observed : [];
      class MerseyElement extends realmHTMLElement() {
        static get observedAttributes() {
          return observed;
        }
        connectedCallback() {
          if (h.connected) h.connected(this);
        }
        disconnectedCallback() {
          if (h.disconnected) h.disconnected(this);
        }
        attributeChangedCallback(name, oldV, newV) {
          if (h.attributeChanged) h.attributeChanged(this, name, oldV ?? "", newV ?? "");
        }
      }
      globalObject.customElements.define(tag, MerseyElement);
      return null;
    },
    construct(ctorName, argsJson) {
      try {
        const args = JSON.parse(argsJson).map(decode);
        const b = boundCtor(ctorName, args); // generated binding
        if (b) return ok(b.v);
        // A dotted name is a *namespaced* constructor — `Intl.NumberFormat`. Walk
        // the path rather than looking up one flat key, which would never find it.
        const Ctor = ctorName.split(".").reduce((o, k) => (o == null ? o : o[k]), globalObject);
        if (typeof Ctor !== "function") return err(`${ctorName} is not a constructor`);
        return ok(new Ctor(...args));
      } catch (e) {
        return err(e);
      }
    },
  };
}

globalThis.__merseyBridge = makeBridge(globalThis, function (cb, argsJson) {
  return globalThis.__merseyInvoke(cb, argsJson);
});
// Mersey runs natively here: the Stage A polyfill loader sees this and
// stands down (no WASM fetch, no double execution).
globalThis.merseyNative = true;
