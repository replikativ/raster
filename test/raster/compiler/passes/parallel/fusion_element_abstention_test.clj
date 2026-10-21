(ns raster.compiler.passes.parallel.fusion-element-abstention-test
  (:require [clojure.test :refer [deftest is]]
            [raster.par]
            [raster.numeric]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as typed-fusion]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]))

(defn- source-program
  ([source arrays] (source-program source arrays {'n :long}))
  ([source arrays scalars]
   (frontend/form->program source {:dtype :double :array-types arrays :scalar-types scalars})))

(defn- with-unavailable-element-proof [f]
  ;; Capability fault injection, not new source admission: main's ordinary
  ;; Double programs already fuse. Simulate optional analysis abstaining only
  ;; after substitution introduces this producer's multiply into the element.
  (let [unavailable (fn [certify reason]
                      (fn [request dtype]
                        (if (some #{2.0} (flatten (:lambda request)))
                          (throw (ex-info "injected missing element proof"
                                          {:reason reason :reduction-op request}))
                          (certify request dtype))))]
    (with-redefs [scan/certify-reassociation (unavailable scan/certify-reassociation
                                                       :reduction-element-impure-or-unknown)
                  scan/certify (unavailable scan/certify :scan-element-impure-or-unknown)]
      (f))))

(deftest unavailable-element-proof-retains-the-materialized-reduction-value
  (let [source '(let* [tmp (raster.par/pmap i n double (* (aget x i) 2.0))
                       sum (raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))]
                      sum)
        program (source-program source {'x :double})
        [_ control-stats] (typed-fusion/fusion-fixpoint program)
        [fused _] (with-unavailable-element-proof #(typed-fusion/fusion-fixpoint program))
        reconstructed (dialect/equations fused)]
    (is (= 1 (:vertical control-stats)) "the normal supported program really is fusible")
    (is (= program fused)
        "unknown source-projected element effects do not abort a legal materialized program")
    (is (= 2 (count reconstructed)))
    (let [{:keys [parameters body-results]}
          (dialect/lambda-parts (:lambda (dialect/operation-parts (first reconstructed))))
          element (eval (list 'fn parameters (projection/scalar-folds->source (first body-results))))
          original (eval (list 'fn '[x n] source))]
      ;; The unchanged materialized producer preserves the original values.
      ;; This is not a claim about fusion or arbitrary parallel reduction trees.
      (doseq [values [[] [1.25] [Double/MAX_VALUE] [(- Double/MAX_VALUE)] [16777217.0]]]
        (let [input (double-array values)
              actual (reduce + 0.0 (map element input))
              expected (original input (alength input))]
          (is (= (Double/doubleToRawLongBits (double expected))
                 (Double/doubleToRawLongBits (double actual)))))))))

(deftest unavailable-reassociation-does-not-block-later-legal-fusion
  (let [source '(let* [tmp (raster.par/pmap i n double (* (aget x i) 2.0))
                       sum (raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))
                       a (raster.par/pmap k m double (* (aget d k) 2.0))
                       b (raster.par/pmap l m double (+ (aget a l) 1.0))]
                      [sum b])
        program (source-program source {'x :double 'tmp :double 'd :double 'a :double 'b :double}
                                {'n :long 'm :long})
        [result stats] (with-unavailable-element-proof #(typed-fusion/fusion-fixpoint program))]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
    (is (= ['map 'reduce 'map] (mapv dialect/operation-kind (dialect/equations result))))
    (is (= (first (dialect/equations program)) (first (dialect/equations result))))
    (is (= result (dialect/validate! result)))))

(deftest unavailable-element-proof-retains-the-original-scan
  (let [source '(let* [mapped (raster.par/pmap i n double (* (aget x i) 2.0))
                       result (raster.par/scan out acc 0.0 j n double (+ acc (aget mapped j)))]
                      result)
        program (source-program source {'x :double 'out :double})
        [_ control-stats] (typed-fusion/fusion-fixpoint program)
        [result stats] (with-unavailable-element-proof #(typed-fusion/fusion-fixpoint program))]
    (is (= 1 (:vertical control-stats)) "the uninjected scan can fuse")
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (= ['map 'scan] (mapv dialect/operation-kind (dialect/equations result))))))

(deftest unavailable-element-evidence-is-not-an-invalid-algebra-waiver
  (let [candidate {:kind :reduce
                   :attributes {:accumulators '[acc] :identities [0.0] :dtypes [:double]}
                   :body-results '[(+ acc element)]}]
    (with-redefs [scan/certify-reassociation
                  (fn [request _]
                    (throw (ex-info "unavailable" {:reason :reduction-element-impure-or-unknown
                                                   :reduction-op request})))]
      (is (nil? (#'typed-fusion/recertify-fused-algebra candidate))))
    (doseq [reason [:reduction-nonidentity-init :reduction-not-associative :stale-definition-proof-input
                    :reduction-element-impure-or-unknown]]
      (let [failure (ex-info "invalid or unrelated" {:reason reason})]
        (with-redefs [scan/certify-reassociation (fn [& _] (throw failure))]
          (is (identical? failure
                          (try (#'typed-fusion/recertify-fused-algebra candidate) nil
                               (catch clojure.lang.ExceptionInfo actual actual)))))))
    (is (= :invalid-canonical-scalar-conversion
           (try (#'typed-fusion/recertify-fused-algebra
                 (assoc candidate :body-results
                        '[(+ acc (raster.compiler.ir.soac-dialect/scalar-convert {} element))]))
                nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
