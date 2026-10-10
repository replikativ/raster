(ns raster.compiler.lexical-unknown-mask-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.pipeline :as pipeline]))

(def environments [{'x 'long} {'x {:tag 'long}}])

(deftest unknown-locals-mask-outer-types-in-both-environment-shapes
  (doseq [env environments]
    (is (nil? (inf/infer-arg-tag '(unknown-source) env)))
    (doseq [expression ['(let* [x (unknown-source)] x)
                        '(let [x (unknown-source)] x)
                        '(loop* [x (unknown-source)] x)
                        '(loop [x (unknown-source)] x)]]
      (is (nil? (inf/infer-arg-tag expression env))))
    (is (= [nil] (inf/infer-result-tags '(let* [x (unknown-source)] x) env *ns*)))))

(deftest lexical-key-presence-masks-stale-use-metadata
  (let [use (with-meta 'x {:tag 'long :raster.type/tag 'long})]
    (doseq [env [{'x nil} {'x {:tag nil}}]]
      (is (nil? (inf/infer-arg-tag use env))))
    (doseq [env [{'x 'double} {'x {:tag 'double}}]]
      (is (= 'double (inf/infer-arg-tag use env))))
    (is (= 'long (inf/infer-arg-tag use {})))
    (doseq [env environments]
      (let [expression (list 'let* ['x '(unknown-source)] use)]
        (is (nil? (inf/infer-arg-tag expression env)))
        (is (= [nil] (inf/infer-result-tags expression env *ns*)))))))

(deftest the-shared-mask-retains-nontype-metadata-and-known-locals
  (let [use (with-meta 'x {:tag 'long :raster.type/tag 'long
                          :raster.type/element 'double :raster.type/fn-info {:arity 1}
                          :line 12})]
    (doseq [env [{'x nil} {'x {:tag nil}}]]
      (is (= {:line 12} (meta (inf/mask-unknown-local use env)))))
    (is (identical? use (inf/mask-unknown-local use {'x 'long})))
    (is (identical? use (inf/mask-unknown-local use {})))))

(deftest sequential-initializers-see-the-previous-binding-before-masking
  (doseq [env environments]
    (doseq [[expression expected]
            [['(let* [x x] x) 'long]
             ['(let* [y x x (unknown-source)] y) 'long]
             ['(let* [x (unknown-source) y x] y) nil]
             ['(let* [x 1 x (unknown-source)] x) nil]
             ['(let* [x (unknown-source) x 1.0] x) 'double]]]
      (is (= expected (inf/infer-arg-tag expression env)))
      (is (= [expected] (inf/infer-result-tags expression env *ns*))))))

(deftest lexical-cast-heads-do-not-regain-global-operator-types
  (doseq [head ['long 'double 'short 'char 'boolean]]
    (doseq [env [{head nil} {head {:tag nil}}]]
      (let [expression (list head 1)]
        (is (nil? (inf/infer-arg-tag expression env)))
        (is (= [nil] (inf/infer-result-tags expression env *ns*))))))
  (let [expression '(let* [long (unknown-source)] (long 1))]
    (is (nil? (inf/infer-arg-tag expression {})))
    (is (= [nil] (inf/infer-result-tags expression {} *ns*))))
  (is (= 'long (inf/infer-arg-tag '(clojure.core/long 1) {'long nil})))
  (is (= ['long] (inf/infer-result-tags '(clojure.core/long 1) {'long nil} *ns*))))

(deftest public-dispatch-projection-does-not-publish-an-inherited-result-tag
  (let [source '(let* [x (unknown-source)] x)
        expression (list 'let* ['result source] 'result)
        original (eval (list 'fn ['x 'unknown-source] expression))]
    (is (= "reference" (original 1 (fn [] "reference"))))
    (doseq [env environments]
      (let [projected (inline/resolve-generic-deftm-calls expression env)
            result-binding (first (second projected))
            transformed (eval (list 'fn ['x 'unknown-source] projected))]
        (is (nil? (:tag (meta result-binding))))
        (is (nil? (:raster.type/tag (meta result-binding))))
        (is (= "reference" (transformed 1 (fn [] "reference")))))
      (let [tagged (#'pipeline/tag-binding-types expression env)]
        (is (nil? (:tag (meta (first (second tagged))))))
        (is (nil? (:tag (meta (nth tagged 2)))))))))

(deftest public-passes-remove-stale-use-tags-of-an-unknown-shadow
  (let [use (with-meta 'x {:tag 'long :raster.type/tag 'long})
        expression (list 'let* ['x '(unknown-source)] use)]
    (doseq [env environments]
      (doseq [projected [(inline/resolve-generic-deftm-calls expression env)
                        (#'pipeline/tag-binding-types expression env)]]
        (is (nil? (:tag (meta (last projected)))))
        (is (nil? (:raster.type/tag (meta (last projected)))))))))

(deftest explicit-binder-facts-use-the-existing-hint-authority
  (let [binding (with-meta 'x {:tag 'double :raster.type/tag 'double})
        expression (list 'let* [binding '(unknown-source)] 'x)]
    (doseq [env environments]
      (is (= 'double (inf/infer-arg-tag expression env)))
      (is (= [nil] (inf/infer-result-tags expression env *ns*))
          "a binder hint does not certify an unknown initializer's return alternatives")
      (doseq [projected [(inline/resolve-generic-deftm-calls expression env)
                        (#'pipeline/tag-binding-types expression env)]]
        (is (= 'double (inf/hint-tag (first (second projected)))))))))

(deftest binder-hints-do-not-erase-incompatible-result-alternatives
  (let [binding (with-meta 'value {:tag 'raster.ad.forward.Dual
                                  :raster.type/tag 'raster.ad.forward.Dual})
        expression (list 'let* [binding '(if c x (double x))] 'value)
        env {'x 'raster.ad.forward.Dual 'c 'long}]
    (is (= #{'raster.ad.forward.Dual 'double}
           (set (inf/infer-result-tags expression env *ns*))))))

(deftest public-loop-initializer-projection-uses-sequential-masks
  (let [use (with-meta 'x {:tag 'long :raster.type/tag 'long})
        expression (list 'loop* ['x '(unknown-source) 'y use] 'y)
        source (list 'let* ['result expression] 'result)]
    (doseq [env environments]
      (let [projected (inline/resolve-generic-deftm-calls source env)
            loop-expression (second (second projected))
            second-initializer (nth (second loop-expression) 3)]
        (is (nil? (inf/hint-tag second-initializer)))
        (is (nil? (inf/hint-tag (first (second projected)))))
        (is (= "reference"
               ((eval (list 'fn ['x 'unknown-source] projected))
                1 (fn [] "reference"))))))))

(deftest the-walker-establishes-the-same-unknown-local-mask
  (let [use (with-meta 'x {:tag 'long :raster.type/tag 'long})
        expression (list 'let* ['x nil] use)
        projected (walker/walk expression (walker/make-ctx {:type-env {'x {:tag 'long}}}))]
    (is (nil? (:tag (meta (last projected)))))
    (is (nil? (:raster.type/tag (meta (last projected)))))
    (is (nil? ((eval (list 'fn ['x] projected)) 1)))))

(deftest flattened-unknown-bindings-mask-the-outer-walker-context
  (let [expression '(let* [result (let* [x nil] x)] result)
        projected (walker/walk expression (walker/make-ctx {:type-env {'x {:tag 'long}}}))
        bindings (into {} (map vec (partition 2 (second projected))))
        result-binding (nth (second projected) 2)]
    (is (contains? bindings 'x) "the inner unknown binding was flattened")
    (is (nil? (inf/hint-tag result-binding)))
    (is (nil? (inf/hint-tag (last projected))))
    (is (nil? ((eval (list 'fn ['x] projected)) 1)))))

(deftest unknown-reduction-accumulators-mask-outer-inline-facts
  (let [use (with-meta 'acc {:tag 'long :raster.type/tag 'long})
        expression (list 'let* ['result (list 'raster.par/reduce 'acc '(unknown-source) 'i 0 use)]
                         'result)]
    (doseq [env [{'acc 'long} {'acc {:tag 'long}}]]
      (let [projected (inline/resolve-generic-deftm-calls expression env)
            reduction (second (second projected))
            accumulator-use (last reduction)]
        (is (nil? (:tag (meta accumulator-use))))
        (is (nil? (:raster.type/tag (meta accumulator-use))))))))

(deftest result-alternative-authority-does-not-gain-a-loop-fixed-point-proof
  (doseq [env environments]
    (is (= 'long (inf/infer-arg-tag '(loop* [x 1] x) env)))
    (is (= [nil] (inf/infer-result-tags '(loop* [x 1] x) env *ns*)))
    (is (= [nil] (inf/infer-result-tags '(loop* [x (unknown-source)] x) env *ns*)))))
