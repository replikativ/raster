(ns raster.compiler.passes.parallel.typed-imperative-map-packet-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- options [dt]
  {:dtype dt :target-device :ze:0
   :array-types {'x dt 'y dt 'out dt}
   :scalar-types {'rows :long 'width :long}})

(defn- source [dt]
  (let [identity (case dt :float (float 0.0) :double 0.0)
        body (list 'raster.par/reduce 'acc identity 'column 'width
                   '(+ acc (* (aget x (+ (* row width) column))
                              (aget y (+ (* row width) column)))))]
    ;; No whole-program typed admission is supplied. The unrelated host prefix must
    ;; remain at its original source position while the map obtains its own packet.
    (list 'let* ['host-prefix '(host-control x)
                 'result (list 'raster.par/map! 'out 'row 'rows nil
                               (list (case dt :float 'float :double 'double) body))] 'result)))

(deftest dense-imperative-map-recovers-an-independent-typed-packet
  (doseq [dt [:float :double]]
    (let [original (source dt)
          lowered (lower/segop-lower-pass original (options dt))
          scheduled (:form lowered)
          equation (first (filter #(seq (:operations %)) (:equations scheduled)))
          operation (first (:operations equation))
          algorithm (:algorithm equation)
          terms (tree-seq coll? seq algorithm)]
      (is (identical? scheduled (program/validate! scheduled)))
      (is (= 1 (get-in lowered [:stats :typed-soac-reused])))
      (is (= ['host-prefix '(host-control x)] (vec (take 2 (second (:source scheduled))))))
      (is (= :typed-soac (:algorithm-dialect operation)))
      (is (segop/seg-map? operation))
      (is (some? (:scalar-region operation)))
      (is (some dialect/scalar-fold-form? terms))
      (is (not-any? #(and (seq? %) (= 'raster.par/reduce (first %))) terms))
      (is (= dt (:dtype operation)))
      (is (= #{'out} (:outputs operation)))
      (is (= ['out] (dialect/physical-results (dialect/facts algorithm)
                                            (first (dialect/equations algorithm)))))
      (let [emitted (opencl/opencl-pass scheduled :dtype dt :compile-spirv? false :min-elements 0)]
        (is (= 1 (count (:kernels emitted))))
        (is (= 1 (get-in emitted [:stats :segop-reused])))
        (is (zero? (get-in emitted [:stats :segop-relowered] 0)))))))

(deftest offset-maps-do-not-acquire-dense-singleton-admission
  (let [expression '(raster.par/map! out row rows :offset base nil (aget x row))
        called (atom false)]
    (with-redefs-fn {#'lower/singleton-packet (fn [& _] (reset! called true))}
      #(is (nil? (#'lower/selected-operation-packet 'result expression (options :float) (atom #{})))))
    (is (false? @called))))

(deftest typed-map-admission-does-not-swallow-implementation-failures
  (doseq [failure [(ex-info "typed invariant" {:reason :raster/bug})
                   (ex-info "typed invariant" {:reason :raster/fatal})
                   (NullPointerException. "typed implementation bug")]]
    (with-redefs [route/attempt (fn [& _] (throw failure))]
      (is (identical? failure
                      (try (lower/segop-lower-pass (source :float) (options :float))
                           nil
                           (catch Throwable observed observed)))))))
