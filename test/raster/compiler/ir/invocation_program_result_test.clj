(ns raster.compiler.ir.invocation-program-result-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.parallel-program :as program]))

(defn- fixture []
  (let [scalar (av/tensor {:dtype :double :shape []})
        opaque (av/make {:kind :opaque :memory-space :host :ownership :borrowed})
        reduction '(raster.par/reduce acc 0.0 i 3 (+ acc 1.0))
        snapshot (completion/complete-value completion/object-input-contract 'total)
        cast (completion/complete-value
              (completion/source-floating-cast-contract :float {:unchecked-math false}) 'alias)
        bindings [['saved snapshot] ['alias 'saved] ['rounded cast]]
        numerical (program/make
                   {:dialect :typed-parallel
                    :source (list 'let* (into ['total reduction] (mapcat identity bindings)) nil)
                    :values {'total scalar} :outputs '[total]
                    ;; Declaration-only fixture: never emitted, linked or executed.
                    :equations [(program/->ProgramEquation :reduce [:binding 'total] reduction
                                                           [] '[total] nil [{:test-only :reduction}]
                                                           #{} {} {})]})
        plan (invocation/from-prefix
              {:id :imported-scalar :parameters [] :parameter-values {}
               :bindings bindings :binding-values {'saved opaque 'alias opaque
                                                   'rounded (av/tensor {:dtype :float :shape []})}
               :result-imports {'saved {:equation :reduce :result 'total}}
               :program-values {'total scalar} :program-inputs [] :program-outputs '[total]})
        _ (invocation/source-order plan numerical)]
    {:plan plan :program numerical :cast cast}))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest result-import-is-an-explicit-suspension-not-an-evaluator-or-device-read
  (let [{:keys [plan]} (fixture)
        initial (materialization/begin-materialization plan [])
        import (first (:steps plan))
        calls (atom 0)
        evaluate (fn [& _] (swap! calls inc) (throw (AssertionError. "unexpected evaluation")))
        payload (proxy [Object] []
                  (equals [_] (throw (AssertionError. "payload equality")))
                  (hashCode [] (throw (AssertionError. "payload hashing")))
                  (toString [] (throw (AssertionError. "payload printing"))))]
    (is (= :invocation-materialization-program-result
           (reason #(materialization/materialize-next initial evaluate))))
    (is (= :invocation-materialization-program-result
           (reason #(materialization/materialize plan [] {:scalar evaluate :source-completion evaluate}))))
    (is (zero? @calls))
    (is (= :invocation-materialization-import-step
           (reason #(materialization/supply-program-result initial (into {} import) payload))))
    (let [supplied (materialization/supply-program-result initial import payload)
          alias-state (materialization/materialize-next supplied evaluate)
          imported (get-in supplied [:values (:id import)])
          aliased (get-in alias-state [:values (:id (second (:steps plan)))])]
      (is (identical? payload (:source imported)))
      (is (identical? imported aliased))
      (is (identical? supplied (materialization/validate-state! supplied)))
      (is (identical? alias-state (materialization/validate-state! alias-state)))
      (is (= :invocation-materialization-import-step
             (reason #(materialization/supply-program-result supplied import payload))))
      (is (= :invocation-materialization-import-identity
             (reason #(materialization/validate-state!
                       (assoc-in supplied [:values (:id import) :id] [:foreign-import])))))
      (is (zero? @calls)))))

(deftest imported-object-feeds-the-existing-owned-source-completion
  (let [{:keys [plan cast]} (fixture)
        body [cast]
        interface (types/ensure-fn-interface! '[Object])
        emitted (core/emit-owned-bytecode!
                 bytecode/compile-typed-impl! "raster.compiled.ImportedSourceCompletion"
                 '[alias] '[Object] 'Object body *ns* (:iface-name interface)
                 'imported-source-completion)
        region (bytecode/compile-source-floating-completion-region (:emission emitted) body cast)
        execute {:source-completion
                 (fn [_ values]
                   {:type :float :value (apply (:invoke region)
                                              (map #(get-in values [% :source]) (:parameters region)))})}
        initial (materialization/begin-materialization plan [])
        supplied (materialization/supply-program-result initial (first (:steps plan)) (double 6.0))
        alias-state (materialization/materialize-next supplied nil)
        completed (materialization/materialize-next alias-state execute)
        finished (materialization/finish-materialization completed)]
    (is (= {:type :float :value (float 6.0)}
           (get-in finished [:values (:id (last (:steps plan)))])))
    (is (empty? (:program-scalars finished)) "opaque imports never become numerical operands implicitly")
    (is (= :invocation-materialization-import-identity
           (reason #(materialization/validate!
                     (assoc-in finished [:values (:id (first (:steps plan))) :id] [:foreign-import])))))
    (is (identical? finished (materialization/validate! finished)))))
