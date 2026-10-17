(ns raster.ad.normalized-scalar-accumulation-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ad.normalized-map-element-test :as element]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.reverse :as reverse]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- scalar [dtype]
  (av/tensor {:dtype dtype :shape []}))

(defn- accumulation [dtype]
  (with-meta '(raster.ad.reverse/grad-acc left right)
    {:raster.type/tag (case dtype :float 'float :double 'double)}))

(defn- project [values result expression]
  (#'normalized/canonical-scalar-accumulation! values result expression))

(deftest exact-homogeneous-non-nil-accumulations-project-to-typed-addition
  (doseq [dtype [:float :double]]
    (let [values {'left (scalar dtype) 'right (scalar dtype) 'result (scalar dtype)}
          expression (accumulation dtype)
          projected (project values 'result expression)
          program (#'normalized/program (select-keys values '[left right])
                                        ['result expression] '[result])
          region (#'normalized/scalar-program-lambda program '[left right] '[result])
          local (first (:locals (soac/lambda-parts region)))]
      (is (= (list (if (= :double dtype) 'clojure.core/+ 'raster.numeric/+) 'left 'right)
             projected))
      (is (= (case dtype :float 'float :double 'double)
             (:raster.type/tag (meta projected))))
      (is (= dtype (:dtype local)))
      (is (= projected (:init local)))
      (is (= expression (second (#'normalized/bindings program)))
          "the retained scalar program remains the derivation authority"))))

(deftest accumulation-projection-refuses-incomplete-or-conflicting-type-authority
  (let [f (scalar :float) d (scalar :double) array (av/tensor {:dtype :double :shape '[n]})
        typed-double (accumulation :double)]
    (doseq [[values expression expected]
            [[{'left f 'right d 'result d} typed-double :ad-normalized-scalar-accumulation]
             [{'left d 'right d 'result (scalar :float)} typed-double
              :ad-normalized-scalar-accumulation]
             [{'left d 'right d 'result d}
              (with-meta '(raster.ad.reverse/grad-acc left right)
                {:raster.type/tag 'float}) :ad-normalized-unknown-type]
             [{'left d 'right d 'result d}
              '(raster.ad.reverse/grad-acc left missing) :ad-normalized-scalar-accumulation]
             [{'left d 'right d 'result d}
              '(raster.ad.reverse/grad-acc left nil) :ad-normalized-scalar-accumulation]
             [{'left array 'right d 'result d} typed-double :ad-normalized-scalar-space]]]
      (is (= expected (reason #(project values 'result expression)))))
    (testing "lexical metadata cannot turn a local receiver spelling into the global helper"
      (is (= :ad-normalized-callee
             (reason #(project {'left d 'right d 'result d} 'result
                               (with-meta typed-double
                                 {:raster.type/tag 'double
                                  :raster.op/lexical-callee true})))))
      (is (= :ad-normalized-operation
             (reason #(project {'left d 'right d 'result d} 'result
                               '(some.other/grad-acc left right)))))
      (is (= :ad-normalized-scalar-accumulation
             (reason #(project
                       {'left d 'right d 'result d} 'result
                       (with-meta '(.invk some.other/implementation left right)
                         {:raster.op/original 'raster.ad.reverse/grad-acc
                          :raster.type/tag 'double})))))
      (is (= :ad-normalized-operation
             (reason #(project
                       {'left d 'right d 'result d} 'result
                       (with-meta '(some.other/implementation left right)
                         {:raster.op/original 'raster.ad.reverse/grad-acc
                          :raster.type/tag 'double}))))))))

(deftest accumulation-projection-refuses-helper-replacement-before-construction
  (doseq [dtype [:float :double]]
    (let [values {'left (scalar dtype) 'right (scalar dtype) 'result (scalar dtype)}
          expression (accumulation dtype)
          build #(normalized/linearize-map-program
                  (#'element/map-program dtype)
                  (the-ns 'raster.ad.normalized-scalar-accumulation-test))
          bundle (build)
          calls (atom 0)]
      (with-redefs [reverse/grad-acc (fn [& _]
                                     (swap! calls inc)
                                     (throw (ex-info "projection must never invoke the replacement" {})))]
        (is (false? (reverse/approved-grad-acc-definition?)))
        (doseq [construct [#(project values 'result expression) build #(runtime/prepare bundle)]]
          (is (= :ad-normalized-scalar-accumulation (reason construct))))
        (is (zero? @calls) "definition/dispatch checks never invoke the unapproved helper"))
      (is (true? (reverse/approved-grad-acc-definition?)))
      (is (= (list (if (= :double dtype) 'clojure.core/+ 'raster.numeric/+) 'left 'right)
             (project values 'result expression)))
      (is (map? (build)) "restoring the exact library root restores construction"))))

(deftest accumulation-projection-refuses-helper-metadata-replacement
  (doseq [dtype [:float :double]]
    (let [v #'reverse/grad-acc original-metadata (meta v)
          values {'left (scalar dtype) 'right (scalar dtype) 'result (scalar dtype)}
          expression (accumulation dtype)
          build #(normalized/linearize-map-program
                  (#'element/map-program dtype)
                  (the-ns 'raster.ad.normalized-scalar-accumulation-test))
          bundle (build)]
      (try
        (alter-meta! v assoc ::unapproved true)
        (is (false? (reverse/approved-grad-acc-definition?)))
        (doseq [construct [#(project values 'result expression) build #(runtime/prepare bundle)]]
          (is (= :ad-normalized-scalar-accumulation (reason construct))))
        (finally (reset-meta! v original-metadata)))
      (is (true? (reverse/approved-grad-acc-definition?)))
      (is (= original-metadata (meta v)))
      (is (= (list (if (= :double dtype) 'clojure.core/+ 'raster.numeric/+) 'left 'right)
             (project values 'result expression)))
      (is (map? (build)) "restoring exact metadata restores construction"))))

(deftest runtime-guard-retains-the-original-accumulation-helper-dependency
  (doseq [[dtype array cast] [[:double double-array double] [:float float-array float]]]
   (let [bundle (normalized/linearize-map-program
                (#'element/map-program dtype)
                (the-ns 'raster.ad.normalized-scalar-accumulation-test))
        scalar (get-in bundle [:map-programs :proof :nodes 0 :scalar-derivation])
        raw-programs (map scalar [:primal :linear :transpose])
        raw-expressions (mapcat #(take-nth 2 (rest (#'normalized/bindings %))) raw-programs)
        prepared (runtime/prepare bundle)
        arguments (mapv {'n (long 2) 'x (array [2.0 -3.0])} (:inputs prepared))
        [_ pullback] ((:vjp prepared) arguments)
        tangent (array [0.5 -0.25])
        seed (array [1.0 2.0])]
    (is (some #(and (seq? %) (= 'raster.ad.reverse/grad-acc (first %))) raw-expressions))
    (is (= [(cast 2.0) (cast 1.5)] (vec (second ((:jvp prepared) arguments tangent)))))
    (with-redefs [reverse/grad-acc (fn [& _] 0.0)]
      (is (= :ad-normalized-execution-callee
             (reason #((:vjp prepared) arguments))))
      (is (= :ad-normalized-execution-callee
             (reason #((:jvp prepared) arguments tangent))))
      (is (= :ad-normalized-execution-callee
             (reason #(pullback seed))))))))
