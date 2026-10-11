(ns raster.runtime.native-call
  "Synchronous native pointer ownership, independent of compiler and numerical ABI.

  Adapters own descriptors, scalar conversion, arity and access declarations.
  Heap arrays use call-owned staging; native storage remains caller-owned.
  Primitive-array-backed views preserve backing identity and exact byte ranges."
  (:import [java.lang.foreign Arena MemorySegment]
           [java.lang.invoke MethodHandle]))

(defn array-segment
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

(defn- heap-backing [^MemorySegment segment]
  (when-not (.isNative segment)
    (or (.orElse (.heapBase segment) nil)
        ;; Read-only heap segments intentionally hide their backing array. Do
        ;; not defeat this JDK boundary with reflection or guessed identities.
        (throw (ex-info "Heap view backing is unavailable for alias-preserving staging"
                        {:reason :unsupported-native-heap-view})))))

(defn pointer-segment
  "Marshal a primitive array or segment. Heap views require primitive backing
  and a byte range contained in that backing; native storage remains caller-owned.
  Read-only heap views hide their backing and are refused; declare :read access
  on an ordinary heap view instead. Read-only native segments remain supported."
  ^MemorySegment [x]
  (let [segment (if (instance? MemorySegment x) x (array-segment x))]
    (when-not (.isNative ^MemorySegment segment)
      ;; JDK heap addresses are byte offsets in their backing array. Full ofArray
      ;; segments start at zero; validating with asSlice avoids unchecked sums.
      (let [backing (array-segment (heap-backing segment))]
        (.asSlice backing (.address ^MemorySegment segment) (.byteSize ^MemorySegment segment))))
    segment))

(defn- pointer-access! [n access]
  (when-not (and (= n (count access))
                 (every? #{:read :write :read-write} access))
    (throw (ex-info "Native pointer access must match the pointer ABI"
                    {:reason :invalid-native-pointer-access}))))

(defn- pointer-indices! [slots arity access]
  (let [indices (if (and (integer? slots) (<= 0 slots arity))
                  (vec (range slots))
                  (when (vector? slots) slots))]
    (when-not (and indices (= (count indices) (count (distinct indices)))
                   (every? #(and (integer? %) (<= 0 % (dec arity))) indices))
      (throw (ex-info "Native pointer slots must be distinct valid argument indices"
                      {:reason :invalid-native-pointer-slots :slots slots :arity arity})))
    (pointer-access! (count indices) access)
    indices))

(defn- invoke-prepared!
  [^MethodHandle mh ^objects arguments indices n-ptrs access no-alias?]
  ;; A native segment is caller-owned for the synchronous call. Heap arrays use
  ;; invocation-owned native storage; sharing a Java array shares that storage.
  (let [heaps (java.util.IdentityHashMap.)
        ranges (java.util.ArrayList.)]
    (dotimes [i n-ptrs]
      (let [argument-index (if indices (nth indices i) i)
            segment ^MemorySegment (aget arguments argument-index)
            writable? (not= :read (nth access i))
            heap (heap-backing segment)
            backing (when heap (array-segment heap))
            offset (if heap (.address segment) 0)]
        (when heap
          (.asSlice ^MemorySegment backing offset (.byteSize segment)))
        (when (or (not (.isAlive (.scope segment)))
                  (not (.isAccessibleBy segment (Thread/currentThread)))
                  (and writable? (.isReadOnly segment)))
          (throw (ex-info "Native pointer is not accessible with its declared access"
                          {:reason :inaccessible-native-pointer :index argument-index})))
        (when no-alias?
          (doseq [[^MemorySegment previous _ previous-write?] ranges]
            (when (and (or writable? previous-write?)
                       (.isPresent (.asOverlappingSlice segment previous)))
              (throw (ex-info "Writable pointer overlap violates the emitted no-alias ABI"
                              {:reason :native-pointer-alias :index argument-index})))))
        (.add ranges [segment heap writable? offset backing])))
    (if (every? #(nil? (second %)) ranges)
      (.invokeWithArguments mh arguments)
      (with-open [arena (Arena/ofConfined)]
        (dotimes [i n-ptrs]
          (let [[^MemorySegment source heap _ offset ^MemorySegment backing] (.get ranges i)]
            (when heap
              (let [existing (.get heaps heap)
                    ^MemorySegment destination
                    (or existing
                        (let [storage (.allocate arena (.byteSize backing) 8)]
                          (MemorySegment/copy backing 0 ^MemorySegment storage 0 (.byteSize backing))
                          storage))]
                (.put heaps heap destination)
                (aset arguments (int (if indices (nth indices i) i))
                      (.asSlice destination (long offset) (.byteSize source)))))))
        (let [result (.invokeWithArguments mh arguments)]
          ;; On a failed call no staged writes are published. Caller-owned native
          ;; memory is not transactional; its mutations cannot be rolled back.
          (doseq [[^MemorySegment source heap writable? offset] ranges
                  :when (and heap writable?)]
            (MemorySegment/copy ^MemorySegment (.get heaps heap) (long offset) source 0
                                (.byteSize source)))
          result)))))

(defn invoke-ordinary!
  "Invoke an ordinary handle using an adapter-validated argument array.
  Pointer slots are a leading count or an explicit vector of argument indices.
  Their entries must be segments from pointer-segment, and access must contain
  one :read/:write/:read-write declaration per pointer. Scalar ABI and arity
  validation belong to the adapter. Mutates its private argument array to install
  staged addresses. Successful calls publish writable heap results; thrown calls
  close staging without copyback. Native writes are not rolled back."
  [^MethodHandle mh ^objects arguments slots access no-alias?]
  (if (integer? slots)
    (do
      (when-not (<= 0 slots (alength arguments))
        (throw (ex-info "Native pointer slots must be distinct valid argument indices"
                        {:reason :invalid-native-pointer-slots :slots slots :arity (alength arguments)})))
      (pointer-access! slots access)
      (invoke-prepared! mh arguments nil slots access no-alias?))
    (let [indices (pointer-indices! slots (alength arguments) access)]
      (invoke-prepared! mh arguments indices (count indices) access no-alias?))))

(defn prepare-call
  "Prepare an adapter's ordinary scalar-return FFM handle for interleaved ABI
  arguments. The adapter owns descriptors, scalar conversion and provider policy.
  Pointer positions come only from the resolved handle's MemorySegment carriers;
  pointer-access is snapshotted in that order. Returns a function of one argument
  collection, preserving original ABI slot order. Aggregate/allocator conventions
  and pointer returns require another explicit contract and are refused."
  [^MethodHandle mh pointer-access]
  (let [signature (.type mh)
        carriers (vec (.parameterArray signature))
        scalar-types #{Integer/TYPE Long/TYPE Float/TYPE Double/TYPE}
        _ (when-not (and (every? #(or (= MemorySegment %) (scalar-types %)) carriers)
                          (or (= Void/TYPE (.returnType signature))
                              (scalar-types (.returnType signature))))
            (throw (ex-info "Native call requires a supported scalar-return ABI"
                            {:reason :unsupported-native-handle-shape})))
        indices (vec (keep-indexed #(when (= MemorySegment %2) %1) carriers))
        access (vec pointer-access)
        arity (count carriers)
        _ (pointer-indices! indices arity access)]
    (fn [arguments]
      (let [values (object-array arguments)]
        (when-not (= arity (alength values))
          (throw (ex-info "Native call arguments must match the resolved ABI"
                          {:reason :native-call-arity :expected arity :actual (alength values)})))
        (doseq [i indices]
          (aset values (int i) (pointer-segment (aget values (int i)))))
        (invoke-prepared! mh values indices (count indices) access false)))))
