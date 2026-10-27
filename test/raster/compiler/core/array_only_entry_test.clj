(ns raster.compiler.core.array-only-entry-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.runtime.hardware :as runtime-hardware]))

;; Every parameter is a reference. No scalar annotation can accidentally select
;; the typed producer and make this migration's positive cases vacuous.
(deftm array-only-owned [input :- (Array float)] :- (Array float) input)
(deftm array-symbol-identity [input :- floats] :- floats input)
(deftm array-unused-input [events :- (Array long), ignored :- (Array float)] :- Object
  (aset events 0 7)
  nil)
(deftm array-checked-size [events :- (Array long), ignored :- (Array float)] :- (Array float)
  (aset events 0 1)
  (float-array (int (aget events 1))))
(deftm array-return-only [value events] :- (Array float)
  (aset ^longs events 0 1)
  value)
(deftm array-only-square! [input :- (Array float), out :- (Array float)] :- (Array float)
  (par/map! out i (alength input) float (* (aget input i) (aget input i)))
  out)
(deftm array-only-source-square! [input :- (Array float), out :- (Array float)] :- (Array float)
  (par/map! out i (alength input) float (* (aget input i) (aget input i)))
  out)

(defn- typed-implementation [v]
  (let [source (core/resolve-deftm-var v)
        receiver (ns-resolve (.ns source) (symbol (str (.sym source) "-impl")))]
    ;; Invoke the actual real deftm's IFn entry, not generic overload selection:
    ;; wrong-Class/null controls concern this selected entry's contract.
    (:implementation (core/ensure-current-owned-bytecode! receiver))))

(deftest primitive-array-entry-selection-uses-the-shared-annotation-contract
  (doseq [annotation (concat (keys types/primitive-array-info)
                            (keep (fn [[element tag]]
                                    (when (contains? types/primitive-array-info tag)
                                      (list 'Array element)))
                                  types/array-elem->tag)
                            '[(Frozen (Array float)) (Param floats)])]
    (is (true? (boolean (types/has-primitive-annotations? [annotation] nil))) (str annotation))
    (is (true? (boolean (types/has-primitive-annotations? [] annotation))) (str annotation)))
  (doseq [annotation '[Object objects (Array Object) (Array String) (Array UnknownElement) nil]]
    (is (not (types/has-primitive-annotations? [annotation] nil)))
    (is (not (types/has-primitive-annotations? [] annotation))))
  (let [interfaces (atom 0)]
    (with-redefs [types/ensure-fn-interface!
                  (fn [& _] (swap! interfaces inc) (throw (AssertionError. "query generated an interface")))]
      (doseq [annotation '[(Frozen (Fn [Double] Double))
                          (Param (Frozen (Fn [Double] Double)))]]
        (is (not (types/has-primitive-annotations? [annotation] nil)))
        (is (not (types/has-primitive-annotations? [] annotation)))))
    (is (zero? @interfaces))))

(deftest array-only-original-entry-has-authentic-typed-source-ownership
  (let [source (core/resolve-deftm-var #'array-only-owned)
        receiver (ns-resolve (.ns source) (symbol (str (.sym source) "-impl")))
        observed (entry/call-with-definition-inputs
                  #(entry/source-var-body-entry-carriers source))
        frontend (entry/call-with-definition-inputs #(core/prepare-deftm-frontend! source))]
    (is (var? receiver))
    (is (= [(class (float-array 0))] (:parameter-carriers observed)))
    (is (= :typed-source-body-entry (:scope observed)))
    (is (:pending-source? observed) "no numerical invocation was needed to construct the entry")
    (is (some? (:source-frontend frontend)))
    (is (entry/assert-frontend-source! frontend source))
    (is (= 'floats (get-in frontend [:source-entry :return-tag])))
    (is (= {:unchecked-math false} (get-in frontend [:source-entry :source-policy])))))

(deftest cold-array-only-source-can-demand-its-real-physical-producer
  ;; This exact array-only shape previously had a plain IFn source and no typed
  ;; receiver. Do not invoke it on the JVM or prepare its module before compile.
  (let [scope (create-ns (gensym "raster.cold-array-source."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm exponential [input :- (Array float)] :- (Array float)
                 (let [out (raster.arrays/zeros-like input (alength input))]
                   (raster.par/map! out i (alength input) nil
                                   (float (Math/exp (double (aget input i)))))))))
      (let [function (ns-resolve scope 'exponential)
            source (core/resolve-deftm-var function)
            before (entry/call-with-definition-inputs #(entry/source-var-body-entry-carriers source))
            target :cuda:cold-array-only-source
            descriptor {:device-id target :device-type :gpu :backend :cuda :fp64? true
                        :max-workgroup-size 1024 :shared-local-memory 65536 :total-eus 32}]
        (is (:pending-source? before))
        (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "cold compilation opened a session")))
                      runtime-hardware/ensure-init! (fn [& _] (throw (AssertionError. "cold compilation probed hardware")))]
          (let [result (equation-first/compile function {:target target :dtype :float} descriptor)]
            (is (seq (:kernels result)))
            (is (every? #(= :kernel-body (get-in % [:attributes :emission-route])) (:kernels result))))))
      (finally (remove-ns (ns-name scope))))))

(deftest array-only-entry-checks-null-unused-arguments-and-return-in-source-order
  (let [input (float-array [1 2]) f (typed-implementation #'array-symbol-identity)]
    (is (identical? input (f input)))
    (is (nil? (f nil)))
    (is (thrown? ClassCastException (f (double-array 2)))))
  (let [events (long-array 2) f (typed-implementation #'array-unused-input)]
    (is (thrown? ClassCastException (f events (double-array 1))))
    (is (zero? (aget events 0)) "even an unused typed argument is checked before body effects")
    (is (nil? (f events nil)))
    (is (= 7 (aget events 0)) "an array declaration does not establish presence"))
  (doseq [[count expected-error] [[-1 NegativeArraySizeException] [2147483648 ArithmeticException]]
          :let [f (typed-implementation #'array-checked-size)]]
    (let [events (long-array [0 count])]
      (is (= expected-error (try (f events nil) :accepted (catch Throwable failure (class failure)))))
      (is (= 1 (aget events 0)) "the original body effect precedes size/cast failure")))
  (doseq [count [0 2]]
    (let [events (long-array [0 count])
          result ((typed-implementation #'array-checked-size) events nil)]
      (is (= count (alength ^floats result)))
      (is (= 1 (aget events 0)))))
  (let [events (long-array 1) f (typed-implementation #'array-return-only)]
    (is (thrown? ClassCastException (f (double-array 1) events)))
    (is (= 1 (aget events 0)) "a return-only array check follows body effects")
    (is (nil? (f nil events)))))

(deftest array-only-float-storage-retains-jvm-aot-bit-parity
  (let [source (typed-implementation #'array-only-square!)
        aot (pipeline/compile-aot #'array-only-square! :dtype :float :simd? false)]
    ;; Typed Float arithmetic includes IEEE overflow; the former plain
    ;; array-only producer could instead check a Double-to-Float conversion.
    ;; This checks the new source contract, not backwards arithmetic parity.
    (doseq [values [[] [0.0 -0.0 1.0000001192092896 -3.25 4097.0]
                   [Float/MIN_VALUE Float/MAX_VALUE Float/POSITIVE_INFINITY
                    Float/NEGATIVE_INFINITY Float/NaN]]]
      (let [input (float-array values) source-out (float-array (count values))
            aot-out (float-array (count values))]
        (is (identical? source-out (source input source-out)))
        (is (identical? aot-out (aot input aot-out)))
        (is (= (mapv #(Float/floatToRawIntBits %) source-out)
               (mapv #(Float/floatToRawIntBits %) aot-out)))
        (is (= (mapv #(Float/floatToRawIntBits
                       (unchecked-float (* (double %) (double %)))) input)
               (mapv #(Float/floatToRawIntBits %) source-out)))))))

(deftest array-only-source-lowers-for-three-declared-gpu-targets-without-resources
  (doseq [backend [:ocl :cuda :hip]
          :let [target (keyword (str (name backend) ":array-only-entry-source"))
                device {:type backend :name "Declared array-only source fixture"
                        :capabilities {:warp-size 32 :subgroup-sizes [32]
                                       :max-workgroup-size 256 :shared-local-memory 32768
                                       :compute-capability [8 0] :gfx-arch :gfx1100
                                       :total-eus 32}}]]
    (with-redefs [runtime-hardware/device (constantly device)
                  runtime-hardware/ensure-init! (fn [& _] (throw (AssertionError. "source lowering probed hardware")))
                  gpu/make-session (fn [& _] (throw (AssertionError. "source lowering acquired a device")))
                  gpu/alloc! (fn [& _] (throw (AssertionError. "source lowering allocated device memory")))]
      (let [descriptor (#'hardware/build-descriptor target)
            input (float-array [1 2]) out (float-array 2)
            prepared (with-redefs [hardware/descriptor-for (constantly descriptor)]
                       (compiled/lower #'array-only-source-square!
                                       [input out]
                                       {:compiler :equation-first :target target :dtype :float
                                        :roles {'input :input 'out :state}}))]
        (is (compiled/prepared? prepared))
        (is (= target (:target prepared)))
        (is (seq (:in-tree prepared)))
        (is (seq (:out-tree prepared)))
        (is (= [1.0 2.0] (vec input)))
        (is (= [0.0 0.0] (vec out)) "source preparation does not execute the numerical body")
        (is (every? #(= :float (:dtype %)) (concat (:in-tree prepared) (:out-tree prepared))))))))
