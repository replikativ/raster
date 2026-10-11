(ns raster.compiler.backend.cpu.codegen
  "Native CPU C backend — generalizes the OpenCL kernel emitter to host C.

  Reuses the shared expression emitter (raster.compiler.backend.gpu.c-emit) for
  bodies; only the kernel scaffold differs: a host `for` loop with `restrict`
  pointers instead of OpenCL work-items + `__global`. Compiled with the system
  C compiler at -O3 -march=native (the autovec hygiene Futhark omits), loaded
  via ordinary Panama FFM calls (heap arrays use invocation-owned staging;
  caller-owned native segments can be passed without a copy).

  This is the long-tail path (SOAC par/map + par/reduce → autovec C, beating the
  JVM for free). The int8 quantized-matmul MAC is a separate op-descriptor
  primitive whose lowering emits the maddubs intrinsic (the one ISA seam)."
  (:require [clojure.string :as str]
            [clojure.java.io :as io]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.backend.intrinsics :as intrinsics])
  (:import [java.lang.foreign Arena Linker Linker$Option FunctionDescriptor
            SymbolLookup ValueLayout MemoryLayout MemorySegment]
           [java.nio.channels FileChannel]
           [java.nio.charset StandardCharsets]
           [java.nio.file Files OpenOption StandardCopyOption StandardOpenOption]
           [java.security MessageDigest]))

;; ---- backend config: native C (source extrema, statement-exprs ok) ----
(def cpu-config
  {:cast-style       :c
   :atomic-add-int   "__atomic_add"      ;; not used on the single-thread path yet
   :atomic-add-float :cas-helper
   :float-abs        (get-in intrinsics/table [:abs :native-c :fn])
   :float-max        (get-in intrinsics/table [:max :native-c :fn])
   :float-min        (get-in intrinsics/table [:min :native-c :fn])
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
                                  (range n-arrays)))
        body (emit-body body-expr dtype)]
    (str "#include <math.h>\n"
         (intrinsics/native-c-helper-sources body)
         "void " kernel-name "(" params ") {\n"
         "  for (int idx = 0; idx < n; idx++) {\n"
         reads "\n"
         "    out[idx] = " body ";\n"
         "  }\n}\n")))

;; ---- compile + load (clang -O3 -march=native -> .so -> Panama) ----
(defn- sha256 [^String s]
  (let [d (.digest (MessageDigest/getInstance "SHA-256")
                   (.getBytes s StandardCharsets/UTF_8))]
    (apply str (map #(format "%02x" (bit-and 0xff %)) d))))

(def ^:private cc (or (System/getenv "RASTER_CC") "clang"))
(def ^:private cache-dir (let [d (io/file (System/getProperty "java.io.tmpdir") "raster-cpu-kernels")]
                           (.mkdirs d) (.getAbsolutePath d)))

(def ^:private cflags
  "Preserve source floating-point operations, including NaNs and signed zero.
  Reassociation and contraction require explicit numerical permission in the
  lowering; they are not a property of selecting the native backend. Explicit
  SIMD/FMA intrinsics selected by a legal schedule remain available. These flags
  participate in compiler-identity, invalidating older fast-math artifacts."
  ["-O3" "-march=native" "-funroll-loops" "-ffp-contract=off" "-fveclib=libmvec"
   "-shared" "-fPIC"])

(def ^:private link-libs ["-lmvec" "-lm"])

(def ^:private cache-schema :raster.cpu-native-cache/v2)

(defn- command-output
  [arguments]
  (try
    (let [process-builder (doto (ProcessBuilder. ^java.util.List (into [cc] arguments))
                            (.redirectErrorStream true)
                            (.directory (io/file cache-dir)))
          process (.start process-builder)]
      ;; Target probing uses stdin as its translation unit.
      (.close (.getOutputStream process))
      (let [output (slurp (.getInputStream process))]
        (.waitFor process)
        output))
    (catch Throwable error
      (str "unavailable:" (.getName (class error))))))

(def ^:private compiler-identity
  (delay
    {:schema cache-schema
     :compiler cc
     :compiler-version (command-output ["--version"])
     ;; -march=native output is CPU-specific. Probe the compiler's resolved
     ;; target so a shared temp directory cannot reuse a binary for a different
     ;; microarchitecture with the same generic os.arch value.
     :native-target (command-output ["-###" "-march=native" "-x" "c"
                                     "-c" "-" "-o" "/dev/null"])
     :flags cflags
     :link-libs link-libs
     :os (System/getProperty "os.name")
     :architecture (System/getProperty "os.arch")}))

(defn- source-cache-key
  [^String source]
  (sha256 (pr-str [@compiler-identity source])))

(defn compile-source!
  "Compile C source to a cached .so. Returns path.

   Compilation is serialized per artifact across JVMs and published by atomic
   move, so concurrent cold starts cannot observe a partial shared object."
  [^String src]
  (let [base (str cache-dir "/k_" (source-cache-key src))
        so (str base ".so")
        so-file (io/file so)]
    (when-not (.exists so-file)
      (let [lock-path (.toPath (io/file (str base ".lock")))
            options (into-array OpenOption [StandardOpenOption/CREATE StandardOpenOption/WRITE])]
        (with-open [channel (FileChannel/open lock-path options)
                    lock (.lock channel)]
          ;; Another JVM may have published the artifact while this JVM waited.
          (when-not (.exists so-file)
            (let [directory (io/file cache-dir)
                  c-file (java.io.File/createTempFile "raster-kernel-" ".c" directory)
                  tmp-so (java.io.File/createTempFile "raster-kernel-" ".so" directory)]
              (try
                (spit c-file src)
                (let [r (-> (ProcessBuilder. ^java.util.List
                                             (concat [cc] cflags
                                                     [(.getAbsolutePath c-file) "-o"
                                                      (.getAbsolutePath tmp-so)]
                                                     link-libs))
                            (.redirectErrorStream true) (.start))
                      out (slurp (.getInputStream r))]
                  (when-not (zero? (.waitFor r))
                    (throw (ex-info (str "C compile failed: " out) {:src src})))
                  (Files/move (.toPath tmp-so) (.toPath so-file)
                              (into-array StandardCopyOption
                                          [StandardCopyOption/ATOMIC_MOVE
                                           StandardCopyOption/REPLACE_EXISTING])))
                (finally
                  (Files/deleteIfExists (.toPath c-file))
                  (Files/deleteIfExists (.toPath tmp-so)))))))))
    so))

(def ^:private linker (Linker/nativeLinker))

(def ^:private scalar-layout
  {:int ValueLayout/JAVA_INT :long ValueLayout/JAVA_LONG
   :float ValueLayout/JAVA_FLOAT :double ValueLayout/JAVA_DOUBLE})

(defn- array-segment
  "Resolve FFM's primitive-array overloads without reflective method lookup.
  This is a Java storage boundary, not scalar type inference."
  ^MemorySegment [x]
  (cond
    (instance? double/1 x) (MemorySegment/ofArray ^doubles x)
    (instance? float/1 x) (MemorySegment/ofArray ^floats x)
    (instance? long/1 x) (MemorySegment/ofArray ^longs x)
    (instance? int/1 x) (MemorySegment/ofArray ^ints x)
    (instance? short/1 x) (MemorySegment/ofArray ^shorts x)
    (instance? byte/1 x) (MemorySegment/ofArray ^bytes x)
    (instance? char/1 x) (MemorySegment/ofArray ^chars x)
    :else (throw (ex-info "Native pointer argument requires a primitive numeric array"
                          {:reason :unsupported-native-pointer
                           :argument-class (some-> x class .getName)}))))

(defn- scalar-marshaller [dtype]
  (case dtype
    :int #(int %) :long #(long %) :float #(float %) :double #(double %)
    (throw (ex-info "Unsupported native scalar argument ABI"
                    {:reason :unsupported-native-scalar :dtype dtype}))))

(defn- pointer-segment ^MemorySegment [x]
  (let [segment (if (instance? MemorySegment x) x (array-segment x))]
    (when (and (instance? MemorySegment x) (not (.isNative ^MemorySegment segment)))
      (throw (ex-info "Heap segment views require an explicit staging/alias contract"
                      {:reason :unsupported-native-heap-view})))
    segment))

(defn- invoke-ordinary
  [^java.lang.invoke.MethodHandle mh ^objects arguments n-ptrs access no-alias?]
  ;; A native segment is caller-owned for the synchronous call. Heap arrays use
  ;; invocation-owned native storage; sharing a Java array shares that storage.
  (let [heaps (java.util.IdentityHashMap.)
        ranges (java.util.ArrayList.)]
    (dotimes [i n-ptrs]
      (let [segment ^MemorySegment (aget arguments i)
            writable? (not= :read (nth access i))
            heap (.orElse (.heapBase segment) nil)]
        (when (or (not (.isAlive (.scope segment)))
                  (not (.isAccessibleBy segment (Thread/currentThread)))
                  (and writable? (.isReadOnly segment)))
          (throw (ex-info "Native pointer is not accessible with its declared access"
                          {:reason :inaccessible-native-pointer :index i})))
        (when no-alias?
          (doseq [[previous previous-heap previous-write?] ranges]
            (when (and (or writable? previous-write?)
                       (if heap
                         (identical? heap previous-heap)
                         (and (nil? previous-heap)
                              (< (.address segment) (+ (.address ^MemorySegment previous)
                                                       (.byteSize ^MemorySegment previous)))
                              (< (.address ^MemorySegment previous)
                                 (+ (.address segment) (.byteSize segment))))))
              (throw (ex-info "Writable pointer overlap violates the emitted no-alias ABI"
                              {:reason :native-pointer-alias :index i})))))
        (.add ranges [segment heap writable?])))
    (if (every? #(nil? (second %)) ranges)
      (.invokeWithArguments mh arguments)
      (with-open [arena (Arena/ofConfined)]
        (dotimes [i n-ptrs]
          (let [[^MemorySegment source heap writable?] (.get ranges i)]
            (when heap
              (let [existing (.get heaps heap)
                    destination ^MemorySegment
                    (or (:segment existing)
                        (let [storage (.allocate arena (.byteSize ^MemorySegment source) 8)]
                          (MemorySegment/copy source 0 ^MemorySegment storage 0 (.byteSize source))
                          storage))]
                (.put heaps heap {:segment destination :source source
                                  :write? (or writable? (:write? existing))})
                (aset arguments i destination)))))
        (let [result (.invokeWithArguments mh arguments)]
          ;; On a failed call no staged writes are published. Caller-owned native
          ;; memory is not transactional; its mutations cannot be rolled back.
          (doseq [entry (.values heaps) :when (:write? entry)]
            (MemorySegment/copy ^MemorySegment (:segment entry) 0 ^MemorySegment (:source entry) 0
                                (.byteSize ^MemorySegment (:source entry))))
          result)))))

(defn load-kernel
  "Load a kernel from a compiled .so. Old arities return void; the explicit
  return ABI supports :void/:int/:long/:float/:double. `n-ptrs` = number of
  leading pointer args (arrays). `scalars` describes the trailing scalar args (in
  order, after the arrays): either a COUNT (all JAVA_INT, the common counter case)
  or a VECTOR of type keywords (:int/:long/:float/:double) when types are mixed
  (e.g. a double `eps`). Pointers must precede scalars in the call. Invocation
  requires exactly the declared argument count; extra arguments are rejected.
  Ordinary calls stage heap arrays and accept caller-owned native segments.
  :pointer-access defaults to :read-write for every pointer. :no-alias? enforces
  an emitted restrict ABI. :call-policy :critical requires explicit caller
  guarantees {:extremely-short? true :no-upcalls? true} in :critical-contract;
  duration must be bounded in ALL cases, not just expected to be fast."
  ([so name n-ptrs] (load-kernel so name n-ptrs 1))
  ([so name n-ptrs scalars]
   (load-kernel so name n-ptrs scalars :void))
  ([so name n-ptrs scalars return-type]
   (load-kernel so name n-ptrs scalars return-type {}))
  ([so name n-ptrs scalars return-type
    {:keys [call-policy critical-contract pointer-access no-alias?]
     :or {call-policy :ordinary}}]
   (when-not (or (= :void return-type) (contains? scalar-layout return-type))
     (throw (ex-info "Unsupported native scalar return ABI"
                     {:reason :unsupported-native-return :return-type return-type})))
   (when-not (or (= call-policy :ordinary)
                 (and (= call-policy :critical)
                      (= true (:extremely-short? critical-contract))
                      (= true (:no-upcalls? critical-contract))))
     (throw (ex-info "Native critical calls require explicit bounded no-upcall guarantees"
                     {:reason :unsupported-native-call-policy :call-policy call-policy})))
   (when (and (= call-policy :critical) no-alias?)
     (throw (ex-info "Critical calls with a restrict ABI are not yet admitted"
                     {:reason :unsupported-critical-no-alias})))
   (let [access (or pointer-access (vec (repeat n-ptrs :read-write)))
         _ (when-not (and (= n-ptrs (count access))
                          (every? #{:read :write :read-write} access))
             (throw (ex-info "Native pointer access must match the pointer ABI"
                             {:reason :invalid-native-pointer-access})))
         stypes (if (number? scalars) (vec (repeat scalars :int)) (vec scalars))
         marshallers (into (vec (repeat n-ptrs (if (= call-policy :critical)
                                               array-segment pointer-segment)))
                           (map scalar-marshaller stypes))
         arity (count marshallers)
         lib (SymbolLookup/libraryLookup ^String so (Arena/global))
         layouts (concat (repeat n-ptrs ValueLayout/ADDRESS) (map scalar-layout stypes))
         fd (if (= :void return-type)
              (FunctionDescriptor/ofVoid (into-array MemoryLayout layouts))
              (FunctionDescriptor/of (scalar-layout return-type)
                                     (into-array MemoryLayout layouts)))
         mh (.downcallHandle linker (.orElseThrow (.find lib name)) fd
                             (into-array Linker$Option
                                         (if (= call-policy :critical)
                                           [(Linker$Option/critical true)] [])))]
     (fn [& args]
       ;; The ABI-directed conversion plan is compiled once. In particular,
       ;; overloaded ofArray must not invoke Reflector on every pointer argument.
       (let [arguments (object-array arity)]
         (loop [index 0 remaining (seq args)]
           (if (< index arity)
             (if remaining
               (do (aset arguments index ((nth marshallers index) (first remaining)))
                   (recur (inc index) (next remaining)))
               (throw (ex-info "Too few native kernel arguments"
                               {:reason :native-call-arity :expected arity :actual index})))
             (when remaining
               (throw (ex-info "Too many native kernel arguments"
                               {:reason :native-call-arity :expected arity
                                :actual (+ index (count remaining))})))))
         (if (= call-policy :critical)
           (.invokeWithArguments ^java.lang.invoke.MethodHandle mh arguments)
           (invoke-ordinary mh arguments n-ptrs access no-alias?)))))))

(defn compile-elementwise
  "Convenience: emit + compile + load an element-wise kernel. Returns a fn
  (in0 ... in_{k-1} out n)."
  [name dtype body-expr n-arrays]
  (let [src (emit-elementwise name dtype body-expr n-arrays)]
    (load-kernel (compile-source! src) name (inc n-arrays))))
