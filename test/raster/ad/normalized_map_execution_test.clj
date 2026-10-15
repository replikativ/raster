(ns raster.ad.normalized-map-execution-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-map-element-test :as element]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.par]))

(defn- source [dt options]
  (#'element/map-program dt options))

(defn- executable [program]
  (let [projection (route/materialize-host-map program)]
    (assoc projection :execute
           (binding [*ns* (the-ns 'raster.ad.normalized-map-execution-test)]
             (eval (list 'fn (:parameters projection) (:body projection)))))))

(defn- invoke [executable values]
  (apply (:execute executable) (mapv values (:inputs executable))))

(deftest physical-map-primal-snapshots-and-derivatives-use-existing-materialization
  (doseq [dt [:double :float]
          :let [array (if (= dt :float) float-array double-array)
                cast (if (= dt :float) float double)]
          n [0 3]]
    (let [bundle (normalized/linearize-map-program
                  (source dt {}) (the-ns 'raster.ad.normalized-map-execution-test))
          interface (get-in bundle [:map-programs :interface])
          primal (executable (get-in bundle [:map-programs :primal]))
          push (executable (get-in bundle [:map-programs :linear]))
          pull (executable (get-in bundle [:map-programs :transpose]))
          x (array (take n [-2.0 0.0 3.0]))
          dx (array (repeat n 0.25))
          dy (array (repeat n -0.5))
          outputs (invoke primal {'n n 'x x})
          snapshot (second outputs)
          residuals (zipmap (:outputs primal) outputs)
          pushed (first (invoke push (assoc residuals 'n n (first (:tangent-inputs interface)) dx)))
          expected-pull (mapv #(cast (* 2.0 % -0.5)) (seq x))]
      (is (= (mapv #(cast (* % %)) (seq x)) (vec (first outputs))))
      (is (= (vec x) (vec snapshot)))
      (is (not (identical? x snapshot)))
      (is (= (class x) (class snapshot) (class pushed)))
      ;; The derivative consumes primal-time coefficients, not mutable source storage.
      (dotimes [i n] (aset x i (cast 99.0)))
      (let [pulled (first (invoke pull (assoc residuals 'n n (:cotangent-input interface) dy)))]
        (is (= expected-pull (vec pulled)))
        (is (= (reduce + (map * (seq pushed) (seq dy)))
               (reduce + (map * (seq dx) (seq pulled))))))
      (is (= (vec snapshot) (vec (second outputs)))))))

(deftest physical-scalar-captures-and-exact-extent-identities
  (doseq [extent ['count :extent [:binding 'n]]]
    (let [program (soac/remap-values
                   (source :double {:captures '[scale]
                                    :body '(raster.numeric/* element scale)}) {'n extent})
          bundle (normalized/linearize-map-program
                  program (the-ns 'raster.ad.normalized-map-execution-test))
          interface (get-in bundle [:map-programs :interface])
          primal (executable (get-in bundle [:map-programs :primal]))
          pull (executable (get-in bundle [:map-programs :transpose]))]
      (is (= [[6.0 -3.0]] (mapv vec (invoke primal {extent 2 'x (double-array [2.0 -1.0])
                                                    'scale 3.0}))))
      (is (= [[1.5 -6.0]]
             (mapv vec (invoke pull {extent 2 (:cotangent-input interface) (double-array [0.5 -2.0])
                                     'scale 3.0})))))))

(deftest host-materialization-refuses-to-erase-placement-or-output-ownership
  (doseq [[id field value] [['x :memory-space :device]
                            ['x :placement {:device :arc}]
                            ['y :sharding {:axis 0}]
                            ['y :ownership :external]]]
    (let [program (source :double {:values
                                   {id (assoc (get-in (soac/facts (source :double {})) [:values id])
                                              field value)}})]
      (is (= :typed-soac-host-map-boundary
             (try (route/materialize-host-map program) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest invariant-local-coefficients-are-physical-primal-time-snapshots
  (let [program (source :double
                        {:captures '[scale]
                         :locals [(soac/local-value 'coefficient :double
                                                    '(raster.numeric/* scale scale))]
                         :body '(raster.numeric/* element coefficient)})
        bundle (normalized/linearize-map-program
                program (the-ns 'raster.ad.normalized-map-execution-test))
        interface (get-in bundle [:map-programs :interface])
        primal (executable (get-in bundle [:map-programs :primal]))
        pull (executable (get-in bundle [:map-programs :transpose]))
        outputs (invoke primal {'n 2 'x (double-array [1.0 3.0]) 'scale 2.0})
        residuals (zipmap (:outputs primal) outputs)]
    (is (= [[4.0 12.0] [4.0 4.0]] (mapv vec outputs)))
    (is (not (contains? (set (:inputs pull)) 'scale)))
    (is (= [[2.0 -8.0]]
           (mapv vec (invoke pull (assoc residuals 'n 2
                                         (:cotangent-input interface) (double-array [0.5 -2.0]))))))))

(defn- refusal [program]
  (try (route/materialize-host-map program) nil
       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest compiler-owned-stores-cannot-be-captured-by-internal-binders
  (doseq [index ['double 'float]]
    (is (= :typed-soac-host-map-binder-collision
           (refusal (source :double {:attributes {:index index}})))))
  (is (= :typed-soac-region-binders
         (try
           (source :double {:locals [(soac/local-value 'i :double
                                                       '(raster.numeric/* element element))]
                            :body '(raster.numeric/* element i)})
           nil
           (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))

(deftest numerical-policy-values-cannot-be-erased-by-host-projection
  (let [program (source :double {})
        facts (assoc-in (soac/facts program) [:attributes :source-arithmetic]
                        {:kind :relaxed :reassociation true})]
    (is (= :typed-soac-host-map-boundary
           (refusal (soac/make facts (soac/equations program) (soac/outputs program)))))))
