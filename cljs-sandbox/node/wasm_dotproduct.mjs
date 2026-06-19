// Experiment 1 — can we emit a fast numeric kernel as WebAssembly from a
// "compiler-shaped" code generator, the way raster emits JVM bytecode today?
//
// This is the load-bearing de-risk for the cljs CPU backend: binaryen.js is the
// proposed analogue of java.lang.classfile. Here we build a dot-product kernel
// TWO ways and compare against a plain-JS baseline:
//   (a) scalar  f64  loop
//   (b) SIMD128 f64x2 loop (the v128 path that replaces JDK Vector API)
//
// Run:  cd cljs-sandbox && npm install && node node/wasm_dotproduct.mjs
//
// What to look for: both wasm kernels produce the same sum as JS (correctness),
// and the SIMD kernel is faster than scalar — proving the emit→optimize→
// instantiate pipeline and the SIMD128 path both work end-to-end under node.

import binaryen from "binaryen";

const N = 1 << 20; // 1,048,576 elements — coarse kernel, amortizes the call
const ITERS = 200;

// ---------------------------------------------------------------------------
// Build a wasm module with two exported kernels over a shared linear memory.
// Layout: a[] at byte offset 0, b[] at byte offset N*8 (both f64).
// ---------------------------------------------------------------------------
function buildModule() {
  const m = new binaryen.Module();
  m.setFeatures(binaryen.Features.All); // enable SIMD, bulk-memory, etc.

  // 1 page = 64KiB; need 2*N*8 bytes => ceil(2*N*8 / 65536) pages.
  const pages = Math.ceil((2 * N * 8) / 65536);
  m.setMemory(pages, pages, "mem");

  const i32 = binaryen.i32;
  const f64 = binaryen.f64;
  const v128 = binaryen.v128;

  const bOffsetBytes = N * 8;

  // --- (a) scalar f64 dot product ------------------------------------------
  // double dot_scalar(int n) {
  //   double acc = 0; for (i=0;i<n;i++) acc += a[i]*b[i]; return acc; }
  // locals: 0=n(param) 1=i 2=acc 3=addrA 4=addrB
  {
    const n = m.local.get(0, i32);
    const body = [];
    body.push(m.local.set(1, m.i32.const(0)));        // i = 0
    body.push(m.local.set(2, m.f64.const(0)));        // acc = 0
    const loop = m.loop("L",
      m.block(null, [
        // addrA = i*8 ; addrB = bOffset + i*8
        m.local.set(3, m.i32.mul(m.local.get(1, i32), m.i32.const(8))),
        m.local.set(4, m.i32.add(m.i32.const(bOffsetBytes),
                                 m.i32.mul(m.local.get(1, i32), m.i32.const(8)))),
        // acc += a[addrA] * b[addrB]
        m.local.set(2, m.f64.add(m.local.get(2, f64),
          m.f64.mul(m.f64.load(0, 8, m.local.get(3, i32)),
                    m.f64.load(0, 8, m.local.get(4, i32))))),
        // i++
        m.local.set(1, m.i32.add(m.local.get(1, i32), m.i32.const(1))),
        // if (i < n) continue
        m.br_if("L", m.i32.lt_s(m.local.get(1, i32), n)),
      ]));
    body.push(loop);
    body.push(m.return(m.local.get(2, f64)));
    m.addFunction("dot_scalar", binaryen.createType([i32]), f64,
      [i32, f64, i32, i32], m.block(null, body));
    m.addFunctionExport("dot_scalar", "dot_scalar");
  }

  // --- (b) SIMD128 f64x2 dot product ---------------------------------------
  // Two lanes at a time; horizontal-add the accumulator at the end.
  // locals: 0=n(param) 1=i 2=accVec(v128) 3=addrA 4=addrB
  {
    const n = m.local.get(0, i32);
    const body = [];
    body.push(m.local.set(1, m.i32.const(0)));
    body.push(m.local.set(2, m.f64x2.splat(m.f64.const(0))));  // accVec = {0,0}
    const loop = m.loop("L2",
      m.block(null, [
        m.local.set(3, m.i32.mul(m.local.get(1, i32), m.i32.const(8))),
        m.local.set(4, m.i32.add(m.i32.const(bOffsetBytes),
                                 m.i32.mul(m.local.get(1, i32), m.i32.const(8)))),
        // accVec += a2 * b2   (f64x2 lanewise)
        m.local.set(2, m.f64x2.add(m.local.get(2, v128),
          m.f64x2.mul(
            m.v128.load(0, 8, m.local.get(3, i32)),
            m.v128.load(0, 8, m.local.get(4, i32))))),
        m.local.set(1, m.i32.add(m.local.get(1, i32), m.i32.const(2))), // i += 2
        m.br_if("L2", m.i32.lt_s(m.local.get(1, i32), n)),
      ]));
    body.push(loop);
    // horizontal add: lane0 + lane1
    body.push(m.return(
      m.f64.add(
        m.f64x2.extract_lane(m.local.get(2, v128), 0),
        m.f64x2.extract_lane(m.local.get(2, v128), 1))));
    m.addFunction("dot_simd", binaryen.createType([i32]), f64,
      [i32, v128, i32, i32], m.block(null, body));
    m.addFunctionExport("dot_simd", "dot_simd");
  }

  if (!m.validate()) throw new Error("binaryen validation failed");
  m.optimize(); // run wasm-opt passes — the C2 analogue
  return m;
}

function bench(label, fn) {
  // warmup
  let r = 0;
  for (let i = 0; i < 20; i++) r = fn();
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < ITERS; i++) r = fn();
  const t1 = process.hrtime.bigint();
  const us = Number(t1 - t0) / 1000 / ITERS;
  console.log(`  ${label.padEnd(18)} ${us.toFixed(1)} µs/call   sum=${r.toFixed(6)}`);
  return r;
}

async function main() {
  console.log(`binaryen ${binaryen.VERSION ?? "(version n/a)"}  N=${N}  iters=${ITERS}\n`);

  const m = buildModule();
  const bytes = m.emitBinary();
  console.log(`emitted wasm: ${bytes.length} bytes`);
  const wat = m.emitText();
  console.log(`(wat lines: ${wat.split("\n").length})\n`);

  const { instance } = await WebAssembly.instantiate(bytes, {});
  const { dot_scalar, dot_simd, mem } = instance.exports;

  // Fill shared linear memory: a[i]=i*1e-3, b[i]=(i%7)*1e-2
  const heap = new Float64Array(mem.buffer);
  const a = heap.subarray(0, N);
  const b = heap.subarray(N, 2 * N);
  for (let i = 0; i < N; i++) { a[i] = i * 1e-3; b[i] = (i % 7) * 1e-2; }

  // JS baseline
  const jsDot = () => { let s = 0; for (let i = 0; i < N; i++) s += a[i] * b[i]; return s; };

  console.log("results:");
  const ref = bench("js scalar", jsDot);
  const sc  = bench("wasm scalar", () => dot_scalar(N));
  const sd  = bench("wasm simd f64x2", () => dot_simd(N));

  const ok = Math.abs(sc - ref) < 1e-3 && Math.abs(sd - ref) < 1e-3;
  console.log(`\ncorrectness: ${ok ? "PASS" : "FAIL"} (|wasm - js| < 1e-3)`);
  if (!ok) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
