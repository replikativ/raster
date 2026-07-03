(ns raster.compiler.backend.cpu.codegen
  "Native CPU C backend — generalizes the OpenCL kernel emitter to host C.

  Reuses the shared expression emitter (raster.compiler.backend.gpu.c-emit) for
  bodies; only the kernel scaffold differs: a host `for` loop with `restrict`
  pointers instead of OpenCL work-items + `__global`. Compiled with the system
  C compiler at -O3 -march=native (the autovec hygiene Futhark omits), loaded
  via Panama FFM with Linker$Option/critical (zero-copy on-heap array passing).

  This is the long-tail path (SOAC par/map + par/reduce → autovec C, beating the
  JVM for free). The int8 quantized-matmul MAC is a separate op-descriptor
  primitive whose lowering emits the maddubs intrinsic (the one ISA seam)."
  (:require [clojure.string :as str]
            [clojure.java.io :as io]
            [raster.compiler.backend.gpu.c-emit :as ce])
  (:import [java.lang.foreign Arena Linker Linker$Option FunctionDescriptor
            SymbolLookup ValueLayout MemoryLayout MemorySegment]
           [java.security MessageDigest]))

;; ---- backend config: native C (float uses fabsf/fmaxf, statement-exprs ok) ----
(def cpu-config
  {:cast-style       :c
   :atomic-add-int   "__atomic_add"      ;; not used on the single-thread path yet
   :atomic-add-float :cas-helper
   :float-abs        "fabsf"
   :float-max        "fmaxf"
   :float-min        "fminf"
   :float-suffix?    true})

;; Reuse the shared faceted registry (dtype/native-types via c-emit) rather than a
;; private copy — the old literal drifted (:long "long" vs the registry's C
;; "long long") and lacked :half/:byte. E-slice of the dtype-spine convergence.
(def type-map ce/type-map)

(defn- emit-body
  "Emit a body S-expr to a C expression string under the CPU config + scalar type."
  [body-expr dtype]
  (binding [ce/*emit-config* cpu-config
            ce/*scalar-type* (type-map dtype "double")]
    (ce/emit-expr body-expr nil #{} "idx")))

(defn emit-elementwise
  "Emit a host-loop element-wise C kernel: out[i] = body(arr0[i],arr1[i],...).
  body-expr uses x,y,z,w for the input array elements (like the OpenCL emitter)."
  [kernel-name dtype body-expr n-arrays]
  (let [ct (type-map dtype "double")
        vars ["x" "y" "z" "w"]
        params (str/join ", " (concat (map #(str "const " ct "* restrict arr" %) (range n-arrays))
                                      [(str ct "* restrict out") "int n"]))
        reads (str/join "\n" (map #(str "    " ct " " (get vars % (str "v" %)) " = arr" % "[idx];")
                                  (range n-arrays)))]
    (str "#include <math.h>\n"
         "void " kernel-name "(" params ") {\n"
         "  for (int idx = 0; idx < n; idx++) {\n"
         reads "\n"
         "    out[idx] = " (emit-body body-expr dtype) ";\n"
         "  }\n}\n")))

;; ---- compile + load (clang -O3 -march=native -> .so -> Panama) ----
(defn- sha [^String s]
  (let [d (.digest (MessageDigest/getInstance "SHA-1") (.getBytes s))]
    (apply str (map #(format "%02x" %) (take 8 d)))))

(def ^:private cc (or (System/getenv "RASTER_CC") "clang"))
(def ^:private cache-dir (let [d (io/file (System/getProperty "java.io.tmpdir") "raster-cpu-kernels")]
                           (.mkdirs d) (.getAbsolutePath d)))

(def ^:private cflags
  "clang flags. -ffast-math + -fveclib=libmvec let the LLVM vectorizer reassociate
  reductions AND call glibc's vector transcendentals (e.g. _ZGVdN4v_exp) — the
  equivalent of the JVM Vector API's lanewise EXP. Without these, exp/log loops stay
  scalar (libm) and lose to the JVM. Inference-grade precision (matches llama.cpp)."
  ["-O3" "-march=native" "-funroll-loops" "-ffast-math" "-fveclib=libmvec"
   "-shared" "-fPIC"])

(def ^:private link-libs ["-lmvec" "-lm"])

(defn compile-source!
  "Compile C source to a cached .so. Returns path."
  [^String src]
  (let [base (str cache-dir "/k_" (sha src))
        so (str base ".so")]
    (when-not (.exists (io/file so))
      (let [c (str base ".c")]
        (spit c src)
        (let [r (-> (ProcessBuilder. ^java.util.List
                     (concat [cc] cflags [c "-o" so] link-libs))
                    (.redirectErrorStream true) (.start))
              out (slurp (.getInputStream r))]
          (when-not (zero? (.waitFor r))
            (throw (ex-info (str "C compile failed: " out) {:src src}))))))
    so))

(def ^:private linker (Linker/nativeLinker))

(def ^:private scalar-layout
  {:int ValueLayout/JAVA_INT :long ValueLayout/JAVA_LONG
   :float ValueLayout/JAVA_FLOAT :double ValueLayout/JAVA_DOUBLE})

(defn load-kernel
  "Load a void(addr*, ..., scalars) kernel from a compiled .so. `n-ptrs` = number of
  leading pointer args (arrays). `scalars` describes the trailing scalar args (in
  order, after the arrays): either a COUNT (all JAVA_INT, the common counter case)
  or a VECTOR of type keywords (:int/:long/:float/:double) when types are mixed
  (e.g. a double `eps`). Pointers must precede scalars in the call."
  ([so name n-ptrs] (load-kernel so name n-ptrs 1))
  ([so name n-ptrs scalars]
   (let [stypes (if (number? scalars) (vec (repeat scalars :int)) (vec scalars))
         lib (SymbolLookup/libraryLookup ^String so (Arena/global))
         layouts (concat (repeat n-ptrs ValueLayout/ADDRESS) (map scalar-layout stypes))
         fd (FunctionDescriptor/ofVoid (into-array MemoryLayout layouts))
         mh (.downcallHandle linker (.orElseThrow (.find lib name)) fd
                             (into-array Linker$Option [(Linker$Option/critical true)]))]
     (fn [& args]
       ;; Hint mh as MethodHandle (matches the BLAS bindings) so invokeWithArguments
       ;; isn't a reflective call. Per-call cost is dominated by critical(true)
       ;; heap-pinning of the array segments; amortized in the monolithic
       ;; one-call-per-compiled-fn model. Arrays first (pinned segments), then
       ;; scalars marshalled to their declared primitive type.
       (let [ptr-args (map #(MemorySegment/ofArray %) (take n-ptrs args))
             scal-args (map (fn [t v] (case t :int (int v) :long (long v)
                                            :float (float v) :double (double v)))
                            stypes (drop n-ptrs args))]
         (.invokeWithArguments ^java.lang.invoke.MethodHandle mh
                               (object-array (concat ptr-args scal-args))))))))

(defn compile-elementwise
  "Convenience: emit + compile + load an element-wise kernel. Returns a fn
  (in0 ... in_{k-1} out n)."
  [name dtype body-expr n-arrays]
  (let [src (emit-elementwise name dtype body-expr n-arrays)]
    (load-kernel (compile-source! src) name (inc n-arrays))))
