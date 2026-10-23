(ns raster.compiler.passes.scalar.inline-dispatch-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.par :as par]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.types :as types]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.gpu.core :as gpu]))

(deftm long-only [unused :- Long] :- Long 7)
(deftm zero-argument [] :- Long 9)
(deftm promoted [value :- Double] :- Double (* value value))
(deftm prioritized [value :- Number] :- Long 11)
(deftm prioritized [value :- Double] :- Long 13)
(deftm parametric-first
  (All [T] [values :- (Array T)] :- T (let [value (aget values 0)] value)))

(deftm incompatible-caller [value :- Double] :- (Array double)
  (let [size (long-only value)
        out (double-array size)]
    (par/map! out i size double 1.0)))

(deftm compatible-caller [value :- Long] :- (Array double)
  (let [size (long-only value)
        out (double-array size)]
    (par/map! out i size double 1.0)))

(defn- resolve-inline [function tags]
  (binding [inline/*inline-scalar-bodies?* true]
    (#'inline/try-resolve-deftm function tags)))

(deftest generic-selection-does-not-use-a-single-method-as-an-acceptance-proof
  (let [function 'raster.compiler.passes.scalar.inline-dispatch-test/long-only]
    (doseq [tags [nil [nil] ['double] ['Object]]]
      (is (nil? (resolve-inline function tags))))
    (is (= '[long] (:tags (resolve-inline function '[long]))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Arity mismatch"
                         (resolve-inline function '[long long]))))
  (is (= [] (:tags (resolve-inline
                   'raster.compiler.passes.scalar.inline-dispatch-test/zero-argument []))))
  (is (nil? (inference/specialize-call-signature
             'raster.compiler.passes.scalar.inline-dispatch-test/zero-argument nil)))
  (is (nil? (inference/specialize-call-signature
             'raster.compiler.passes.scalar.inline-dispatch-test/long-only [nil]))))

(deftest shared-selection-retains-priority-specialization-and-promotion-obligations
  (let [function 'raster.compiler.passes.scalar.inline-dispatch-test/prioritized]
    (is (= '[double] (:tags (resolve-inline function '[double]))))
    (is (= (:tags (inference/registered-call-signature function '[double]))
           (:tags (resolve-inline function '[double])))))
  (let [function 'raster.compiler.passes.scalar.inline-dispatch-test/parametric-first]
    (doseq [tags ['[floats] '[doubles] '[floats]]]
      (is (= tags (:tags (resolve-inline function tags))))))
  (let [function 'raster.compiler.passes.scalar.inline-dispatch-test/promoted
        signature (inference/specialize-call-signature function '[float])]
    (is (= '[double] (:promotion-casts signature)))
    (is (nil? (resolve-inline function '[float]))
        "a dispatch promotion is not a formal argument completion proof")))

(deftest incompatible-public-call-is-not-erased-into-an-accepted-gpu-plan
  (doseq [value [1.0 Double/NaN Double/POSITIVE_INFINITY Double/MAX_VALUE]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"No matching method"
                         (incompatible-caller value))))
  (let [target :ocl:inline-dispatch-acceptance
        descriptor {:device-id target :device-type :gpu :backend :ocl
                    :subgroup-dialect :opencl-portable :max-workgroup-size 256}
        failure (with-redefs [gpu/alloc! (fn [& _] (throw (AssertionError. "dispatch probe allocated")))]
                  (try (equation/compile #'incompatible-caller {:target target :dtype :double} descriptor)
                       nil (catch clojure.lang.ExceptionInfo error error)))]
    (is (instance? clojure.lang.ExceptionInfo failure))
    (is (= 1 (get-in (ex-data failure)
                     [:undevirtualized 'raster.compiler.passes.scalar.inline-dispatch-test/long-only])))
    (is (= (vec (repeat 7 1.0)) (vec (compatible-caller 2))))
    (let [compilation (equation/compile #'compatible-caller {:target target :dtype :double} descriptor)
          lowering (equation/lower compilation [2])]
      (is (= [[7]] (mapv #(get-in % [:view :shape]) (vals (:nodes lowering))))))))

(deftest ad-construction-is-distinct-from-eventual-gradient-invocation
  (let [runtime (reverse/value+grad #'promoted)
        expanded (inline/inline-value+grad-call
                  {:var-sym 'raster.compiler.passes.scalar.inline-dispatch-test/promoted
                   :args '[(draw)] :mode :value+grad})
        ;; Reserved completion syntax belongs to its real bytecode consumer,
        ;; not Clojure eval. Object draw results must pass the entry's actual
        ;; parameter conversion before the projected numerical body.
        interface (types/ensure-fn-interface! '[Object])
        compiled (:implementation
                  (core/emit-owned-bytecode!
                   bytecode/compile-typed-impl! "raster.compiled.InlineADConstruction"
                   '[draw] '[Object] 'Object
                   [(list 'let* (vec (mapcat identity (:bindings expanded))) (:elements expanded))]
                   (the-ns 'raster.compiler.passes.scalar.inline-dispatch-test)
                   (:iface-name interface) 'inline-ad-construction))
        describe (fn [run]
                   (try [:value (mapv (fn [value]
                                       [(class value)
                                        (if (instance? Float value)
                                          (Float/floatToRawIntBits value)
                                          (Double/doubleToRawLongBits value))]) (run))]
                        (catch Exception error [:error (class error)])))]
    (doseq [value [(float 2.0) 2.0 (float -0.0) Double/NaN nil]]
      (let [calls (atom 0)
            draw #(do (swap! calls inc) value)]
        (is (= (describe #(runtime (draw))) (describe #(compiled draw))))
        (is (= 2 @calls))))
    (let [failure (ex-info "argument evaluation failed" {})]
      (is (identical? failure
                      (try (compiled #(throw failure)) nil
                           (catch Exception error error)))))))

(deftest generic-selection-retains-the-definition-that-decided-priority
  (let [scope (the-ns 'raster.compiler.passes.scalar.inline-dispatch-test)
        name (gensym "selection-history-")
        function (symbol (str (ns-name scope)) (str name))
        _ (binding [*ns* scope] (eval (list 'deftm name '[value :- Long] ':- 'Long 7)))
        generic (ns-resolve scope name)
        {:keys [inputs info]}
        (entry/call-with-definition-inputs
         #(let [info (resolve-inline function '[long])]
            {:info info :inputs @entry/*emission-inputs*}))
        signature (inference/registered-call-signature function '[long])
        backing (resolve (:mangled-sym signature))
        original @backing]
    (is (some? info))
    (is (contains? (:definition-proofs inputs) generic))
    (binding [*ns* scope] (eval (list 'deftm name '[value :- Number] ':- 'Long 11)))
    (is (identical? original @backing) "the imported backing method itself did not change")
    (is (= :stale-definition-proof-input
           (try (entry/assert-definition-inputs-current! inputs) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftest generic-selection-retains-the-actual-namespace-alias-lookup
  (let [consumer (create-ns (gensym "raster.inline-dispatch-alias."))]
    (try
      (binding [*ns* consumer]
        (alias 'candidate 'raster.compiler.passes.scalar.inline-dispatch-test)
        (let [{:keys [inputs info]}
              (entry/call-with-definition-inputs
               #(let [info (resolve-inline 'candidate/long-only '[long])]
                  {:info info :inputs @entry/*emission-inputs*}))]
          (is (some? info))
          (ns-unalias consumer 'candidate)
          (alias 'candidate 'clojure.core)
          (is (= :stale-definition-proof-input
                 (try (entry/assert-definition-inputs-current! inputs) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
      (finally (remove-ns (ns-name consumer))))))
