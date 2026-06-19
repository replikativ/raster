// Exploration II (step 3): translate raster's POST-PASS IR (dumped as JSON from
// the JVM by jvm/dump_ir.clj → jvm/ir_saxpy.sample.json) into WebAssembly via
// binaryen, run it, and diff against the saxpy oracle. This is the tracer-bullet
// for an IR->wasm backend: it walks the *real* lowered IR raster hands its
// bytecode backend, reading the carried `op`/`tag` metadata (NOT the mangled
// impl name), exactly as the design rules require.
//
//   node node/ir_to_wasm.mjs
//   (regenerate the IR with: cd <raster> && clojure -M -e '(load-file "cljs-sandbox/jvm/dump_ir.clj")')
//
// Handled IR node set (everything saxpy lowers to):
//   let* / loop / recur / if / do / .invk(+,*) / aget / aset! / < / inc / long / sym / int
// Arithmetic op is taken from node.op (:raster.op/original); element type from .tag.

import binaryen from "binaryen";
import { readFile } from "node:fs/promises";

const IR_PATH = new URL("../jvm/ir_saxpy.sample.json", import.meta.url).pathname;
const ir = JSON.parse(await readFile(IR_PATH, "utf8"));

const m = new binaryen.Module();
m.setFeatures(binaryen.Features.All);
m.setMemory(256, 256, "memory");
const { i32, f64 } = binaryen;

// kernel signature: saxpy(a:f64, xptr:i32, yptr:i32, n:i32) -> void
// param indices: a=0(f64) xptr=1(i32) yptr=2(i32) n=3(i32); loop var i=4(i32)
const P = { a: { idx: 0, t: f64 }, x: { idx: 1, t: i32 }, y: { idx: 2, t: i32 }, n: { idx: 3, t: i32 } };
const I = 4; // local index for loop var `i`
const elemBytes = 8; // f64

// address of array element: base_ptr + index*8   (index expr is i32)
const addr = (ptrLocalIdx, idxExpr) =>
  m.i32.add(m.local.get(ptrLocalIdx, i32), m.i32.mul(idxExpr, m.i32.const(elemBytes)));

// Translate a value expression node -> binaryen expr.
function expr(node) {
  switch (node.node) {
    case "int":   return m.i32.const(node.v);
    case "float": return m.f64.const(node.v);
    case "sym": {
      const nm = node.name;
      if (nm === "i") return m.local.get(I, i32);
      if (P[nm])      return m.local.get(P[nm].idx, P[nm].t);
      // an impl symbol (raster.numeric/_plus...-impl) reached as a value: ignore
      throw new Error("unexpected sym " + nm);
    }
    case "call": {
      const h = node.head, A = node.args;
      if (h === "long") return expr(A[0]);                 // index cast: identity for i32
      if (h === "clojure.core/<")   return m.i32.lt_s(expr(A[0]), expr(A[1]));
      if (h === "clojure.core/inc") return m.i32.add(expr(A[0]), m.i32.const(1));
      if (h === "clojure.core/aget") {                     // (aget arr idx) -> f64.load
        const ptr = P[A[0].name].idx;
        return m.f64.load(0, elemBytes, addr(ptr, expr(A[1])));
      }
      if (h === ".invk") {                                 // (.invk impl op1 op2) -> arith
        const op = node.op;                                // <-- semantic op from metadata
        const a = expr(A[1]), b = expr(A[2]);              // A[0] is the impl symbol (skip)
        if (op === "raster.numeric/+") return m.f64.add(a, b);
        if (op === "raster.numeric/*") return m.f64.mul(a, b);
        throw new Error("unhandled .invk op " + op);
      }
      throw new Error("unhandled value head " + h);
    }
    default: throw new Error("unhandled value node " + node.node);
  }
}

// Translate a statement node -> binaryen statement expr.
function stmt(node) {
  if (node.node === "call") {
    const h = node.head, A = node.args;
    if (h === "raster.arrays/aset!") {                     // (aset! arr idx val) -> f64.store
      const ptr = P[A[0].name].idx;
      return m.f64.store(0, elemBytes, addr(ptr, expr(A[1])), expr(A[2]));
    }
    if (h === "recur") {                                   // (recur (inc i)) -> i = ...
      return m.local.set(I, expr(A[0]));
    }
    if (h === "do") return m.block(null, A.map(stmt));
  }
  throw new Error("unhandled stmt " + JSON.stringify(node).slice(0, 80));
}

// Top: (let* [] (loop [i init] (if cond (do ...body... (recur ...)) tail)))
function build(top) {
  if (top.head !== "let*") throw new Error("expected let*");
  const loop = top.args[1];
  if (loop.head !== "loop") throw new Error("expected loop");
  const init = loop.args[0].items[1];          // [i 0] -> 0
  const ifn = loop.args[1];
  const cond = ifn.args[0], body = ifn.args[1]; // body is the (do ...)
  // while loop: i=init; block B { loop L { br_if B !cond; body; br L } }
  return m.block(null, [
    m.local.set(I, expr(init)),
    m.block("B", [
      m.loop("L", m.block(null, [
        m.br("B", m.i32.eqz(expr(cond))),
        stmt(body),                            // do { aset!...; recur(i=inc i) }
        m.br("L"),
      ])),
    ]),
  ]);
}

m.addFunction("saxpy", binaryen.createType([f64, i32, i32, i32]), binaryen.none,
  [i32], build(ir));      // one extra local: i
m.addFunctionExport("saxpy", "saxpy");

if (!m.validate()) throw new Error("binaryen validation failed");
m.optimize();
console.log(`emitted wasm from raster IR: ${m.emitBinary().length} bytes`);

// --- run + diff vs saxpy oracle ---
const { instance } = await WebAssembly.instantiate(m.emitBinary(), {});
const { saxpy, memory } = instance.exports;
const n = 1000, a = 3.0;
const heap = new Float64Array(memory.buffer);
const xIdx = 0, yIdx = n;             // x at byte 0, y at byte n*8
const oracle = new Float64Array(n);
for (let i = 0; i < n; i++) {
  heap[xIdx + i] = i;
  heap[yIdx + i] = 0.5 * i;
  oracle[i] = a * i + 0.5 * i;        // saxpy oracle
}
saxpy(a, xIdx * 8, yIdx * 8, n);
let maxErr = 0;
for (let i = 0; i < n; i++) maxErr = Math.max(maxErr, Math.abs(heap[yIdx + i] - oracle[i]));
console.log(`saxpy(IR→wasm): n=${n}  y[7]=${heap[yIdx + 7]} (oracle ${oracle[7]})  max|err|=${maxErr}`);
console.log(maxErr < 1e-9 ? "=> PASS  (raster IR → wasm computes saxpy correctly)"
                          : "=> FAIL");
process.exit(maxErr < 1e-9 ? 0 : 1);
