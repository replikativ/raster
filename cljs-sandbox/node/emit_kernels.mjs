// Exploration I (producer half) — emit a STATIC .wasm artifact the way the
// build-time-AOT "Model A" backend would, for cljs to fetch + instantiate.
//
// Kernel: saxpy  y[i] = a*x[i] + y[i]  (in-place over shared linear memory),
// scalar + SIMD128 f64x2. Exports `saxpy`, `saxpy_simd`, and `memory`.
// ABI: saxpy(xptr:i32, yptr:i32, n:i32, a:f64) — pointers are byte offsets into
// the exported memory; cljs writes x/y there and reads y back. No fressian; flat
// f64 buffers + (ptr,len) — exactly raster's hot-path boundary.
//
//   node node/emit_kernels.mjs   → writes public/kernels.wasm

import binaryen from "binaryen";
import { writeFile } from "node:fs/promises";

const PAGES = 256; // 16 MiB linear memory — plenty for the demo

function build() {
  const m = new binaryen.Module();
  m.setFeatures(binaryen.Features.All);
  m.setMemory(PAGES, PAGES, "memory");
  const { i32, f64, v128 } = binaryen;

  // locals: 0=xptr 1=yptr 2=n 3=a(f64)  4=i 5=ax 6=ay
  const xptr = () => m.local.get(0, i32), yptr = () => m.local.get(1, i32),
        n = () => m.local.get(2, i32), a = () => m.local.get(3, f64);

  // scalar
  {
    const body = [
      m.local.set(4, m.i32.const(0)),
      m.loop("L", m.block(null, [
        m.local.set(5, m.i32.add(xptr(), m.i32.mul(m.local.get(4, i32), m.i32.const(8)))),
        m.local.set(6, m.i32.add(yptr(), m.i32.mul(m.local.get(4, i32), m.i32.const(8)))),
        m.f64.store(0, 8, m.local.get(6, i32),
          m.f64.add(m.f64.mul(a(), m.f64.load(0, 8, m.local.get(5, i32))),
                    m.f64.load(0, 8, m.local.get(6, i32)))),
        m.local.set(4, m.i32.add(m.local.get(4, i32), m.i32.const(1))),
        m.br_if("L", m.i32.lt_s(m.local.get(4, i32), n())),
      ])),
    ];
    m.addFunction("saxpy", binaryen.createType([i32, i32, i32, f64]), binaryen.none,
      [i32, i32, i32], m.block(null, body)); // locals 4=i 5=ax 6=ay
    m.addFunctionExport("saxpy", "saxpy");
  }

  // SIMD f64x2 (a broadcast to both lanes); assumes n even for the demo
  {
    const body = [
      m.local.set(4, m.i32.const(0)),
      m.local.set(7, m.f64x2.splat(a())),                 // local 7 = {a,a}
      m.loop("L2", m.block(null, [
        m.local.set(5, m.i32.add(xptr(), m.i32.mul(m.local.get(4, i32), m.i32.const(8)))),
        m.local.set(6, m.i32.add(yptr(), m.i32.mul(m.local.get(4, i32), m.i32.const(8)))),
        m.v128.store(0, 8, m.local.get(6, i32),
          m.f64x2.add(m.f64x2.mul(m.local.get(7, v128), m.v128.load(0, 8, m.local.get(5, i32))),
                      m.v128.load(0, 8, m.local.get(6, i32)))),
        m.local.set(4, m.i32.add(m.local.get(4, i32), m.i32.const(2))),
        m.br_if("L2", m.i32.lt_s(m.local.get(4, i32), n())),
      ])),
    ];
    m.addFunction("saxpy_simd", binaryen.createType([i32, i32, i32, f64]), binaryen.none,
      [i32, i32, i32, v128], m.block(null, body)); // locals 4=i 5=ax 6=ay 7=avec
    m.addFunctionExport("saxpy_simd", "saxpy_simd");
  }

  if (!m.validate()) throw new Error("validation failed");
  m.optimize();
  return m.emitBinary();
}

const bytes = build();
const out = new URL("../public/kernels.wasm", import.meta.url).pathname;
await writeFile(out, bytes);
console.log(`wrote ${out} (${bytes.length} bytes)`);
