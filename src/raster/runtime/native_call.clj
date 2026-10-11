(ns raster.runtime.native-call
  "Synchronous native pointer ownership, independent of compiler and numerical ABI.

  Adapters own descriptors, scalar conversion, arity and access declarations.
  Heap arrays use call-owned staging; native storage remains caller-owned.
  Primitive-array-backed views preserve backing identity and exact byte ranges."
  (:import [java.lang.foreign Arena MemorySegment]))

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

(defn invoke-ordinary!
  "Invoke an ordinary handle using an adapter-validated argument array.
  The first n-ptrs entries must be segments from pointer-segment, and access must
  contain one :read/:write/:read-write declaration per pointer. Scalar ABI and
  arity validation belong to the adapter. Mutates its private argument array to
  install staged addresses. Successful calls publish writable heap results;
  thrown calls close staging without copyback. Native writes are not rolled back."
  [^java.lang.invoke.MethodHandle mh ^objects arguments n-ptrs access no-alias?]
  ;; A native segment is caller-owned for the synchronous call. Heap arrays use
  ;; invocation-owned native storage; sharing a Java array shares that storage.
  (let [heaps (java.util.IdentityHashMap.)
        ranges (java.util.ArrayList.)]
    (dotimes [i n-ptrs]
      (let [segment ^MemorySegment (aget arguments i)
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
                          {:reason :inaccessible-native-pointer :index i})))
        (when no-alias?
          (doseq [[^MemorySegment previous _ previous-write?] ranges]
            (when (and (or writable? previous-write?)
                       (.isPresent (.asOverlappingSlice segment previous)))
              (throw (ex-info "Writable pointer overlap violates the emitted no-alias ABI"
                              {:reason :native-pointer-alias :index i})))))
        (.add ranges [segment heap writable? offset backing])))
    (if (every? #(nil? (second %)) ranges)
      (.invokeWithArguments mh arguments)
      (with-open [arena (Arena/ofConfined)]
        (dotimes [i n-ptrs]
          (let [[^MemorySegment source heap _ offset ^MemorySegment backing] (.get ranges i)]
            (when heap
              (let [existing (.get heaps heap)
                    destination ^MemorySegment
                    (or existing
                        (let [storage (.allocate arena (.byteSize backing) 8)]
                          (MemorySegment/copy backing 0 ^MemorySegment storage 0 (.byteSize backing))
                          storage))]
                (.put heaps heap destination)
                (aset arguments i (.asSlice destination (long offset) (.byteSize source)))))))
        (let [result (.invokeWithArguments mh arguments)]
          ;; On a failed call no staged writes are published. Caller-owned native
          ;; memory is not transactional; its mutations cannot be rolled back.
          (doseq [[^MemorySegment source heap writable? offset] ranges
                  :when (and heap writable?)]
            (MemorySegment/copy ^MemorySegment (.get heaps heap) (long offset) source 0
                                (.byteSize source)))
          result)))))
