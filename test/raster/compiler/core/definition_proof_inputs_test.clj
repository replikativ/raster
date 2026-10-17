(ns raster.compiler.core.definition-proof-inputs-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.passes.scalar.local-products :as products]
            [raster.compiler.passes.scalar.normal-return :as normal]
            [raster.numeric :as n]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- array-sum-proof []
  (normal/result-fact '(raster.numeric/+ a b)
                      {'a {:presence :present :array-tag 'floats}
                       'b {:presence :present :array-tag 'floats}}))

(deftest checked-array-projection-preserves-empty-unequal-and-rounded-results
  (doseq [[left right expected] [[[] [] []]
                                [[1.0 2.0] [3.0] [4.0 0.0]]
                                [[1.0] [3.0 4.0] [4.0]]
                                [[16777216.0] [1.0] [16777216.0]]]]
    (let [source (list 'let* ['a (list 'clojure.core/float-array left)
                               'b (list 'clojure.core/float-array right)]
                       '(raster.ad.reverse/grad-acc a b))
          lowered (entry/call-with-definition-inputs
                   #(products/lower-local-products source))
          actual (eval lowered)]
      (is (not-any? #{'raster.ad.reverse/grad-acc} (tree-seq coll? seq lowered)))
      (is (= (class (float-array 0)) (class actual)))
      (is (= expected (vec actual) (vec (eval source)))))))

(deftest array-projection-does-not-reassociate-contribution-order
  (let [source '(let* [a (clojure.core/float-array [16777216.0])
                       b (clojure.core/float-array [1.0])
                       c (clojure.core/float-array [-16777216.0])
                       ab (raster.ad.reverse/grad-acc a b)]
                 (raster.ad.reverse/grad-acc ab c))
        lowered (entry/call-with-definition-inputs #(products/lower-local-products source))]
    (is (not-any? #{'raster.ad.reverse/grad-acc} (tree-seq coll? seq lowered)))
    (is (= [0.0] (vec (eval source)) (vec (eval lowered))))))

(deftest mixed-and-unowned-array-projections-remain-optional
  (let [source '(let* [a (clojure.core/float-array 0)
                       b (clojure.core/double-array 0)]
                 (raster.ad.reverse/grad-acc a b))]
    (is (some #{'raster.ad.reverse/grad-acc}
              (tree-seq coll? seq
                        (entry/call-with-definition-inputs
                         #(products/lower-local-products source))))))
  (binding [entry/*emission-inputs* nil]
    (is (some #{'raster.ad.reverse/grad-acc}
              (tree-seq coll? seq
                        (products/lower-local-products
                         '(let* [a (clojure.core/float-array 0)
                                 b (clojure.core/float-array 0)]
                            (raster.ad.reverse/grad-acc a b))))))))

(deftest transitive-empty-array-allocation-proof-retains-actual-definitions
  (let [inputs (atom {:static [] :values []})
        signature (inference/registered-call-signature 'raster.numeric/similar '[floats])
        allocator (find-var (:mangled-sym signature))]
    (binding [entry/*emission-inputs* inputs]
      (is (= {:presence :present :array-tag 'floats} (array-sum-proof))))
    (is (contains? (:definition-proofs @inputs) allocator)
        "the transitive allocation helper, not merely numeric/+, owns the proof")
    (is (true? (entry/assert-definition-inputs-current! @inputs)))
    (is (= [] (vec (n/+ (float-array 0) (float-array 0)))))
    (with-redefs-fn {allocator (fn [_] nil)}
      #(do
         (is (nil? (n/+ (float-array 0) (float-array 0)))
             "empty inputs expose the nullable allocator without a store throwing")
         (is (= :stale-definition-proof-input
                (reason (fn [] (entry/assert-definition-inputs-current! @inputs)))))))
    (is (true? (entry/assert-definition-inputs-current! @inputs)))))

(deftest static-owner-rejects-a-transitive-change-before-publication
  (let [signature (inference/registered-call-signature 'raster.numeric/similar '[floats])
        allocator (find-var (:mangled-sym signature))
        original @allocator]
    (try
      (is (= :stale-definition-proof-input
             (reason
              #(entry/call-with-definition-inputs
                (fn []
                  (is (= {:presence :present :array-tag 'floats} (array-sum-proof)))
                  (alter-var-root allocator (constantly (fn [_] nil)))
                  :must-not-be-published)))))
      (finally (alter-var-root allocator (constantly original))))))

(deftest proof-collection-never-refreshes-a-first-observation
  (let [inputs (atom {:static [] :values []})
        v #'n/similar original @v]
    (binding [entry/*emission-inputs* inputs]
      (entry/observe-definition-input! v (:ns (meta v)) 'raster.numeric/similar)
      (try
        (alter-var-root v (constantly (fn [_] nil)))
        (is (= :unstable-definition-proof-input
               (reason #(entry/observe-definition-input!
                         v (:ns (meta v)) 'raster.numeric/similar))))
        (is (identical? original (get-in @inputs [:definition-proofs v :value])))
        (finally (alter-var-root v (constantly original)))))))

(deftest emission-handoff-preserves-every-consulted-lookup-site
  (let [scope (create-ns (gensym "raster.proof-lookup."))
        inputs (atom {:static [] :values []})]
    (try
      (binding [*ns* scope] (alias 'chosen 'raster.numeric))
      (binding [entry/*emission-inputs* inputs]
        (entry/observe-definition-input! #'n/similar (the-ns 'raster.numeric)
                                         'raster.numeric/similar)
        (entry/observe-definition-input! #'n/similar scope 'chosen/similar))
      (let [transported (:values @inputs)]
        (is (= 1 (count transported)))
        (is (= 2 (count (:lookups (first transported)))))
        (is (every? entry/var-snapshot-current? transported))
        (ns-unalias scope 'chosen)
        (is (not (every? entry/var-snapshot-current? transported))
            "the second lookup remains necessary after emission artifact handoff"))
      (finally (remove-ns (ns-name scope))))))

(defn- generated-with-consulted-proof [consult!]
  ;; Fault-inject a real consulted definition into normalization, not into the
  ;; resulting source. A post-hoc syntactic scan cannot recover this dependency.
  (let [producer-var (ns-resolve 'raster.ad.reverse 'generated-product-form)
        producer @producer-var
        x (with-meta 'x {:raster.type/tag 'double})
        y (with-meta 'y {:raster.type/tag 'double})
        policy {:unchecked-math false}]
    (with-redefs-fn
            {producer-var
             (fn [program params]
               (let [result (producer program params)]
                 (consult!)
                 result))}
            #(#'reverse/make-runtime-value+grad-fn
                 {:kind :generated-product-bindings :source-policy policy
                  :bindings [y (with-meta '(raster.numeric/* x x)
                                 {:raster.type/tag 'double})]
                  :result y :active-inputs [x]}
                 [x] {:generated-source-policy policy}))))

(deftest generated-owner-retains-normalization-inputs-for-delayed-execution
  (let [f (generated-with-consulted-proof
           #(is (= {:presence :present :array-tag 'floats} (array-sum-proof))))
        [value pullback] (f 3.0)
        allocator (find-var (:mangled-sym
                             (inference/registered-call-signature
                              'raster.numeric/similar '[floats])))]
    (is (= 9.0 value))
    (is (= [6.0] (pullback 1.0)))
    (with-redefs-fn {allocator (fn [_] nil)}
      #(do
         (is (= :stale-definition-proof-input (reason (fn [] (f 3.0)))))
         (is (= :stale-definition-proof-input (reason (fn [] (pullback 1.0)))))))
    (is (= [6.0] (pullback 1.0)))))

(deftest generated-only-accumulation-premise-survives-source-projection
  (let [f (generated-with-consulted-proof
           #(let [source '(let* [a (clojure.core/double 1.0)
                                 b (clojure.core/double 2.0)]
                            (raster.ad.reverse/grad-acc a b))
                  lowered (products/lower-local-products source)]
              (is (not-any? #{'raster.ad.reverse/grad-acc}
                            (tree-seq coll? seq lowered)))
              (is (= 3.0 (eval lowered)))))
        [value pullback] (f 3.0)]
    (is (= 9.0 value))
    (is (= [6.0] (pullback 1.0)))
    (with-redefs [reverse/grad-acc (fn [& _] (throw (AssertionError. "must not run replacement")))]
      (is (= :stale-definition-proof-input (reason #(f 3.0))))
      (is (= :stale-definition-proof-input (reason #(pullback 1.0)))))
    (is (= [6.0] (pullback 1.0)))))
