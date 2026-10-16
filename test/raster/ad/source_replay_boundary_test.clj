(ns raster.ad.source-replay-boundary-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]
            [raster.ad.jvp :as jvp]
            [raster.gpu.core :as gpu]
            [raster.compiler.pipeline :as pipeline]))

(deftm replay-then-external-write
  [x :- Double, ys :- (Array double), zs :- (Array double)] :- Double
  (let [value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        ignored (aset zs 0 99.0)]
    value))

(deftm earlier-external-write
  [x :- Double, ys :- (Array double), zs :- (Array double)] :- Double
  (let [ignored (aset zs 0 3.0)]
    (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))))

(deftm inlined-gradient
  [x :- Double, ys :- (Array double), zs :- (Array double)] :- Double
  (let [vg ((rev/value+grad #'replay-then-external-write :wrt [0]) x ys zs)]
    (nth vg 1)))

(deftm differently-typed-inputs
  [x :- Double, ys :- (Array double), indices :- (Array int)] :- Double
  (let [value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        ignored (aset indices 0 (int 99))]
    value))

(deftm selected-no-write-arm
  [x :- Double, ys :- (Array double), zs :- (Array double), flag :- Long] :- Double
  (let [value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        other (if (= flag 1) x (do (aset zs 0 99.0) x))]
    (n/+ value other)))

(deftm selected-no-replay-arm
  [x :- Double, ys :- (Array double), zs :- (Array double), flag :- Long] :- Double
  (let [value (if (= flag 1)
                (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
                x)
        ignored (aset zs 0 99.0)]
    value))

(deftm sequential-replay-choices
  [x :- Double, ys :- (Array double), zs :- (Array double), first? :- Long, second? :- Long] :- Double
  (let [a (if (= first? 1) (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i)))) x)
        b (if (= second? 1) (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i)))) x)
        ignored (aset zs 0 99.0)]
    (n/+ a b)))

(deftm both-arms-replay
  [x :- Double, ys :- (Array double), zs :- (Array double), flag :- Long] :- Double
  (let [value (if (= flag 1)
                (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
                (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i)))))
        ignored (aset zs 0 99.0)]
    value))

(deftm parent-alias-written-in-child
  [x :- Double, ys :- (Array double), zs :- (Array double), flag :- Long] :- Double
  (let [target zs
        value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        ignored (if (= flag 1) (aset target 0 99.0) 0.0)]
    value))

(deftm actual-generated-guard-map
  [x :- Double, ys :- (Array double), zs :- (Array double), out :- (Array double)] :- (Array double)
  (let [vg ((rev/value+grad #'replay-then-external-write :wrt [0]) x ys zs)]
    (par/map-void! i 1 (aset out i (nth vg 1)))
    out))

(deftm explicit-guarded-map
  [ys :- (Array double), zs :- (Array double), cnt :- Long] :- (Array double)
  (if (identical? ys zs)
    (throw (ex-info "Reverse AD replay input aliases a later write"
                    {:reason :ad-replayed-array-alias}))
    nil)
  (par/map-void! i cnt (aset zs i (aget ys i)))
  zs)

(defn- check-refusal-before-write [f]
  (let [shared (double-array [1.0 2.0])]
    (is (= :ad-replayed-array-alias
           (try (f 2.0 shared shared) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= [1.0 2.0] (vec shared)))))

(deftest runtime-and-compiled-gradients-enforce-input-identity
  (doseq [compile? [false true]]
    (let [gradient (rev/value+grad #'replay-then-external-write :wrt [0] :compile? compile?)]
      (when compile? (is (true? (:raster.ad.reverse/compiled? (meta gradient)))))
      (is (= [6.0 3.0 nil nil]
             (gradient 2.0 (double-array [1.0 2.0]) (double-array [7.0 8.0]))))
      (check-refusal-before-write gradient)))
  ;; vjp differentiates every floating-array argument, including zs. Its write
  ;; now requires a memory-version derivative even when the inputs are disjoint.
  ;; value+grad's explicit :wrt [0] controls above retain the replay coverage.
  (let [ys (double-array [1.0 2.0]) zs (double-array [7.0 8.0])]
    (is (= :ad-active-storage-write
           (try (rev/vjp #'replay-then-external-write 2.0 ys zs) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= [1.0 2.0] (vec ys)))
    (is (= [7.0 8.0] (vec zs)))))

(deftest legitimate-earlier-writes-remain-valid-for-identical-inputs
  (let [shared (double-array [1.0 2.0])]
    (is (= [10.0 5.0 nil nil]
           ((rev/value+grad #'earlier-external-write :wrt [0]) 2.0 shared shared)))
    (is (= [3.0 2.0] (vec shared)))))

(deftest guards-follow-both-write-and-replay-paths
  (doseq [compile? [false true]]
    (let [write-arm (rev/value+grad #'selected-no-write-arm :wrt [0] :compile? compile?)
          replay-arm (rev/value+grad #'selected-no-replay-arm :wrt [0] :compile? compile?)
          shared (double-array [1.0 2.0])]
      (when compile?
        (is (true? (:raster.ad.reverse/compiled? (meta write-arm))))
        (is (true? (:raster.ad.reverse/compiled? (meta replay-arm)))))
      (is (= [8.0 4.0 nil nil nil] (write-arm 2.0 shared shared 1)))
      (is (= [1.0 2.0] (vec shared)))
      (check-refusal-before-write #(write-arm %1 %2 %3 0))
      (is (= [2.0 1.0 nil nil nil] (replay-arm 2.0 shared shared 0)))
      (is (= [99.0 2.0] (vec shared)))
      (check-refusal-before-write #(replay-arm %1 %2 %3 1)))))

(deftest conditional-replay-obligations-compose-lazily
  (let [gradient (rev/value+grad #'sequential-replay-choices :wrt [0])
        shared (double-array [1.0 2.0])]
    (is (= [4.0 2.0 nil nil nil nil] (gradient 2.0 shared shared 0 0)))
    (is (= [99.0 2.0] (vec shared)))
    (doseq [[a b] [[1 0] [0 1] [1 1]]]
      (check-refusal-before-write #(gradient %1 %2 %3 a b))))
  (let [gradient (rev/value+grad #'both-arms-replay :wrt [0])]
    (doseq [flag [0 1]]
      (check-refusal-before-write #(gradient %1 %2 %3 flag)))))

(deftest guards-resolve-enclosing-aliases-at-child-writes
  (let [gradient (rev/value+grad #'parent-alias-written-in-child :wrt [0])
        shared (double-array [1.0 2.0])]
    (is (= [6.0 3.0 nil nil nil] (gradient 2.0 shared shared 0)))
    (check-refusal-before-write #(gradient %1 %2 %3 1))))

(deftest inlining-and-dead-code-elimination-do-not-erase-the-guard
  (let [compiled (pipeline/compile-aot #'inlined-gradient :dtype :double)]
    (is (= 3.0 (compiled 2.0 (double-array [1.0 2.0]) (double-array [7.0 8.0]))))
    (check-refusal-before-write compiled)))

(deftest incompatible-whole-array-classes-do-not-get-identity-guards
  (let [indices (int-array 2)
        prepared (rev/prepare-value+grad #'differently-typed-inputs [0])]
    (is (= [6.0 3.0 nil nil]
           ((rev/value+grad #'differently-typed-inputs :wrt [0])
            2.0 (double-array [1.0 2.0]) indices)))
    (is (= [99 0] (vec indices)))
    (is (not-any? #(and (seq? %) (= 'clojure.core/identical? (first %)))
                  (tree-seq coll? seq (:walked-body prepared))))))

(deftest unlowered-native-guards-are-explicit-refusals
  ;; The generated guard is an actual source effect. Until native preflight
  ;; transports it, target compilation must refuse rather than erase it.
  (let [data (try (let [compiled (pipeline/compile-aot #'inlined-gradient :target :c :dtype :double)]
                (when (instance? java.lang.AutoCloseable compiled)
                  (.close ^java.lang.AutoCloseable compiled))
                nil)
                 (catch clojure.lang.ExceptionInfo e (ex-data e)))]
    (is (= :unsupported-c-family-ir (:reason data)))
    (is (= "Reverse AD replay input aliases a later write" (:expression data))))
  (is (= :structured-control-host-continuation
         (try (pipeline/compile-gpu-program #'explicit-guarded-map :ocl:source-replay-test :dtype :double) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
  (let [allocations (atom 0)
        prepared (rev/prepare-value+grad #'replay-then-external-write [0])
        data (with-redefs [gpu/alloc! (fn [& _]
                                       (swap! allocations inc)
                                       (throw (ex-info "Unexpected compiler allocation"
                                                       {:reason :unexpected-device-allocation})))]
               (try (pipeline/compile-gpu-program #'actual-generated-guard-map :ocl:source-replay-test :dtype :double) nil
                    (catch clojure.lang.ExceptionInfo e (ex-data e))))]
    ;; The actual generated gradient currently fails an earlier, independent
    ;; GPU typedness gate for its backward loop/tuple projection. Do not claim
    ;; that this fixture reaches the retained-host-effect gate above.
    (is (= {'loop* 1 'clojure.core/nth 1} (:non-exempt-untagged data)))
    (is (some #{"Reverse AD replay input aliases a later write"}
              (tree-seq coll? seq (:walked-body prepared))))
    (is (some #(and (symbol? %) (:raster.effect/effectful (meta %))
                    (.startsWith (name %) "_replay_alias_guard"))
              (tree-seq coll? seq (:walked-body prepared))))
    (is (zero? @allocations))))

(deftest higher-order-memory-regions-remain-explicit-refusals
  ;; These builders do not establish a public replay-object scope. The memory
  ;; proof now refuses before their older unlinearized-loop refusal; neither
  ;; path may admit this region or silently produce a higher derivative.
  (doseq [construct [#(jvp/hvp #'replay-then-external-write)
                     #(rev/compile-hvp-fn #'replay-then-external-write)]]
    (let [data (try (construct) nil
                    (catch clojure.lang.ExceptionInfo e (ex-data e)))]
      (is (= :ad-unprojectable-replay-root (:reason data)))
      (is (some #(= "ys" (first (clojure.string/split (name %) #"_α_")))
                (:replays data))))))
