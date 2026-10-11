(ns raster.runtime.native-call
  "Synchronous native pointer ownership, independent of compiler and numerical ABI.

  Adapters own descriptors, scalar conversion, arity and access declarations.
  Heap arrays use call-owned staging; native storage remains caller-owned.
  Heap segment views are not admitted until backing ranges are represented."
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

(defn pointer-segment
  "Marshal a primitive array or caller-owned native segment; refuse heap views."
  ^MemorySegment [x]
  (let [segment (if (instance? MemorySegment x) x (array-segment x))]
    (when (and (instance? MemorySegment x) (not (.isNative ^MemorySegment segment)))
      (throw (ex-info "Heap segment views require an explicit staging/alias contract"
                      {:reason :unsupported-native-heap-view})))
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
