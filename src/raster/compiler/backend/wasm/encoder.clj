(ns raster.compiler.backend.wasm.encoder
  "Minimal hand-rolled WebAssembly *binary* encoder (Track A of the cljs/browser
   port). Pure JVM, no node/binaryen dependency — raster emits .wasm bytes the way
   the JVM backend emits classfile bytes, so the wasm target is usable for any host
   (browser, WASI, server), not just JS.

   Scope: exactly the module shape + opcode set the scalar numeric kernels need
   (see emit.clj). Extended opcode-by-opcode as coverage grows. Byte sequences are
   built as Clojure vectors of ints 0-255, converted to a byte[] at assembly.

   Refs: WebAssembly core spec (binary format), MVP + SIMD value types.")

;; ---------------------------------------------------------------------------
;; LEB128
;; ---------------------------------------------------------------------------
(defn uleb
  "Unsigned LEB128 → vector of bytes (ints 0-255)."
  [n]
  (loop [n (long n), out []]
    (let [b (bit-and n 0x7f), n' (unsigned-bit-shift-right n 7)]
      (if (zero? n')
        (conj out b)
        (recur n' (conj out (bit-or b 0x80)))))))

(defn sleb
  "Signed LEB128 → vector of bytes (ints 0-255)."
  [n]
  (loop [n (long n), out []]
    (let [b (bit-and n 0x7f), n' (bit-shift-right n 7)
          ;; sign bit of the 7-bit group
          sign-set? (not (zero? (bit-and b 0x40)))
          done? (or (and (zero? n') (not sign-set?))
                    (and (= n' -1) sign-set?))]
      (if done?
        (conj out b)
        (recur n' (conj out (bit-or b 0x80)))))))

(defn f64-bytes
  "8 little-endian bytes of an IEEE-754 double."
  [^double x]
  (let [bits (Double/doubleToRawLongBits x)]
    (mapv (fn [i] (int (bit-and (unsigned-bit-shift-right bits (* 8 i)) 0xff))) (range 8))))

(defn f32-bytes
  [x]
  (let [bits (Float/floatToRawIntBits (float x))]
    (mapv (fn [i] (int (bit-and (unsigned-bit-shift-right bits (* 8 i)) 0xff))) (range 4))))

;; ---------------------------------------------------------------------------
;; value types & opcodes
;; ---------------------------------------------------------------------------
(def valtype {:i32 0x7f :i64 0x7e :f32 0x7d :f64 0x7c :v128 0x7b})
(def ^:const empty-block 0x40)

(def op
  "Opcode bytes for the instructions emit.clj uses."
  {:unreachable 0x00 :nop 0x01
   :block 0x02 :loop 0x03 :if 0x04 :else 0x05 :end 0x0b
   :br 0x0c :br-if 0x0d :return 0x0f :call 0x10 :drop 0x1a
   :local.get 0x20 :local.set 0x21 :local.tee 0x22
   :i32.load 0x28 :i64.load 0x29 :f32.load 0x2a :f64.load 0x2b
   :i32.store 0x36 :i64.store 0x37 :f32.store 0x38 :f64.store 0x39
   :i32.const 0x41 :i64.const 0x42 :f32.const 0x43 :f64.const 0x44
   :i32.eqz 0x45 :i32.eq 0x46 :i32.ne 0x47
   :i32.lt_s 0x48 :i32.gt_s 0x4a :i32.le_s 0x4c :i32.ge_s 0x4e
   :f64.eq 0x61 :f64.ne 0x62 :f64.lt 0x63 :f64.gt 0x64 :f64.le 0x65 :f64.ge 0x66
   :f32.eq 0x5b :f32.ne 0x5c :f32.lt 0x5d :f32.gt 0x5e :f32.le 0x5f :f32.ge 0x60
   :i32.add 0x6a :i32.sub 0x6b :i32.mul 0x6c :i32.div_s 0x6d :i32.rem_s 0x6f
   :i64.add 0x7c :i64.sub 0x7d :i64.mul 0x7e
   :f32.add 0x92 :f32.sub 0x93 :f32.mul 0x94 :f32.div 0x95
   :f32.min 0x96 :f32.max 0x97 :f32.sqrt 0x91 :f32.abs 0x8b :f32.neg 0x8c
   :f32.floor 0x8e :f32.trunc 0x8f
   :f64.add 0xa0 :f64.sub 0xa1 :f64.mul 0xa2 :f64.div 0xa3
   :f64.min 0xa4 :f64.max 0xa5 :f64.sqrt 0x9f :f64.abs 0x99 :f64.neg 0x9a
   :f64.floor 0x9c :f64.trunc 0x9d :select 0x1b
   ;; conversions
   :i32.wrap_i64 0xa7 :i64.extend_i32_s 0xac
   :f32.convert_i32_s 0xb2
   :f64.convert_i32_s 0xb7 :f64.convert_i64_s 0xb9
   :i32.trunc_f64_s 0xaa :i64.trunc_f64_s 0xb0
   :f32.demote_f64 0xb6 :f64.promote_f32 0xbb})

(def simd-op
  "SIMD (v128) sub-opcodes — emitted as 0xfd ++ uleb(subop). f64x2 = 2-wide f64,
   f32x4 = 4-wide f32 (the lane widths the loop vectorizer targets)."
  {:v128.load 0x00 :v128.store 0x0b
   :f32x4.splat 0x13 :f64x2.splat 0x14
   :f32x4.add 0xe4 :f32x4.sub 0xe5 :f32x4.mul 0xe6 :f32x4.div 0xe7
   :f32x4.min 0xe8 :f32x4.max 0xe9
   :f64x2.add 0xf0 :f64x2.sub 0xf1 :f64x2.mul 0xf2 :f64x2.div 0xf3
   :f64x2.min 0xf4 :f64x2.max 0xf5})

;; ---------------------------------------------------------------------------
;; instruction helpers — each returns a byte vector
;; ---------------------------------------------------------------------------
(defn i [op-key] [(op op-key)])
(defn i32-const [n] (into [(op :i32.const)] (sleb n)))
(defn i64-const [n] (into [(op :i64.const)] (sleb n)))
(defn f64-const [x] (into [(op :f64.const)] (f64-bytes x)))
(defn f32-const [x] (into [(op :f32.const)] (f32-bytes x)))
(defn local-get [idx] (into [(op :local.get)] (uleb idx)))
(defn local-set [idx] (into [(op :local.set)] (uleb idx)))
(defn local-tee [idx] (into [(op :local.tee)] (uleb idx)))
(defn br [label] (into [(op :br)] (uleb label)))
(defn br-if [label] (into [(op :br-if)] (uleb label)))
(defn call [fidx] (into [(op :call)] (uleb fidx)))
;; memarg = align (uleb) ++ offset (uleb)
(defn mem-load  [op-key align offset] (into (into [(op op-key)] (uleb align)) (uleb offset)))
(defn mem-store [op-key align offset] (into (into [(op op-key)] (uleb align)) (uleb offset)))
;; SIMD: 0xfd prefix ++ uleb(subop) [++ memarg for load/store]
(defn v [simd-key] (into [0xfd] (uleb (simd-op simd-key))))
(defn v128-load  [align offset] (into (into [0xfd] (uleb (simd-op :v128.load)))  (into (uleb align) (uleb offset))))
(defn v128-store [align offset] (into (into [0xfd] (uleb (simd-op :v128.store))) (into (uleb align) (uleb offset))))
(defn block  [body] (into (into [(op :block) empty-block] body) [(op :end)]))
(defn loop*  [body] (into (into [(op :loop)  empty-block] body) [(op :end)]))
;; block whose result is a single value of valtype vt-kw
(defn block-t [vt-kw body] (into (into [(op :block) (valtype vt-kw)] body) [(op :end)]))
;; value-position if: cond already on stack → if(result vt){then}else{else}
(defn if-t [vt-kw then-bytes else-bytes]
  (-> [(op :if) (valtype vt-kw)] (into then-bytes) (into [(op :else)]) (into else-bytes) (into [(op :end)])))

;; ---------------------------------------------------------------------------
;; module assembly
;; ---------------------------------------------------------------------------
(defn- section
  "id + uleb(size) + content-bytes."
  [id content]
  (into (into [id] (uleb (count content))) content))

(defn- vec-section
  "A section whose body is uleb(count) ++ concat(items)."
  [id items]
  (section id (into (uleb (count items)) (apply concat items))))

(defn- name-bytes [s]
  (let [bs (mapv #(bit-and (int %) 0xff) (.getBytes ^String s "UTF-8"))]
    (into (uleb (count bs)) bs)))

(defn functype
  "0x60 ++ uleb(#params) ++ param-valtypes ++ uleb(#results) ++ result-valtypes."
  [param-types result-types]
  (into (into (into (into [0x60] (uleb (count param-types))) (mapv valtype param-types))
              (uleb (count result-types)))
        (mapv valtype result-types)))

(defn- locals-vec
  "Compress a seq of local valtype-keywords into wasm (count,type) runs."
  [local-types]
  (let [runs (reduce (fn [acc t] (if (and (seq acc) (= (peek (peek acc)) t))
                                   (conj (pop acc) [(inc (first (peek acc))) t])
                                   (conj acc [1 t])))
                     [] local-types)]
    (into (uleb (count runs))
          (apply concat (map (fn [[n t]] (into (uleb n) [(valtype t)])) runs)))))

(defn build-module
  "funcs: vector of {:name str :type-idx int :param-types [..] :result-types [..]
                     :locals [valtype-kw...] :body byte-vec (no trailing END)}.
   types: vector of functype byte-vecs (indexed by :type-idx).
   memory: {:min pages :max pages :export str} or nil.
   Returns a Java byte[] of the whole module."
  [{:keys [types funcs memory]}]
  (let [magic   [0x00 0x61 0x73 0x6d 0x01 0x00 0x00 0x00]
        type-sec (vec-section 1 types)
        func-sec (vec-section 3 (mapv #(uleb (:type-idx %)) funcs))
        mem-sec  (when memory
                   (vec-section 5 [(if (:max memory)
                                     (into (into [0x01] (uleb (:min memory))) (uleb (:max memory)))
                                     (into [0x00] (uleb (:min memory))))]))
        exports  (cond-> (vec (map-indexed (fn [idx f] (into (into (name-bytes (:name f)) [0x00]) (uleb idx)))
                                           funcs))
                   (:export memory) (conj (into (into (name-bytes (:export memory)) [0x02]) (uleb 0))))
        export-sec (vec-section 7 exports)
        code-sec (vec-section 10
                              (mapv (fn [f]
                                      (let [body (into (into (locals-vec (:locals f)) (:body f))
                                                       [(op :end)])]
                                        (into (uleb (count body)) body)))
                                    funcs))
        all (concat magic type-sec func-sec (or mem-sec []) export-sec code-sec)]
    (byte-array (map unchecked-byte all))))
