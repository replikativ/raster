(ns raster.compiler.kernel-call-pipeline-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as ra]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]))

(deftm resident-kernel-call-map
  [x :- (Array float) out :- (Array float) scale :- Float n :- Long] :- (Array float)
  (raster.par/map! out i n float (* scale (ra/aget x i))))

(deftm resident-kernel-call-reduce
  [x :- (Array float) out :- (Array float) scale :- Double n :- Long] :- Void
  (let [sum (raster.par/reduce acc 0.0 i n
                               (+ acc (* scale (ra/aget x i))))]
    (raster.par/map-void! j n
                          (ra/aset out j (* (ra/aget x j) sum)))))

(deftest resident-segmap-step-carries-one-executable-call-template
  (let [descriptor (pipeline/compile-gpu-program #'resident-kernel-call-map
                                                 :ze:0 :dtype :float)
        step (first (:steps descriptor))
        runtime-params [(float-array 513) (float-array 513) (float 2.0) 513]
        ordered-values
        (mapv (fn [{:keys [kind sym type value-fn]}]
                (if (= :scalar kind)
                  {:type type :value (value-fn runtime-params)}
                  (keyword (name sym))))
              (:argument-specs step))
        call (kcall/make (:artifact step) ordered-values)]
    (is (= :map (:convention step)))
    (is (kart/kernel-artifact? (:artifact step)))
    (is (= '[x out scale _n_bound]
           (mapv (comp :name :slot) (:argument-specs step))))
    (is (= [:input :output :scalar :scalar]
           (mapv :kind (:argument-specs step))))
    (is (= [256] (get-in call [:geometry :workgroup-size])))
    (is (= [3] (get-in call [:geometry :group-count])))))

(deftest resident-segred-schedule-is-an-explicit-kernel-call-override
  (let [descriptor (pipeline/compile-gpu-program #'resident-kernel-call-reduce
                                                 :ze:0 :dtype :float)
        step (first (filter #(= :reduce (:convention %)) (:steps descriptor)))
        runtime-params [(float-array 513) (float-array 513) 0.75 513]
        ordered-values
        (mapv (fn [{:keys [kind sym type value-fn]}]
                (if (= :scalar kind)
                  {:type type :value (value-fn runtime-params)}
                  (keyword (name sym))))
              (:argument-specs step))
        default-call (kcall/make (:artifact step) ordered-values)
        resident-call (kcall/make (:artifact step) ordered-values {:group-count [1]})]
    (is (kart/kernel-artifact? (:artifact step)))
    (is (= [:input :output :scalar :scalar]
           (mapv :kind (:argument-specs step))))
    (is (= [3] (get-in default-call [:geometry :group-count])))
    (is (= [1] (get-in resident-call [:geometry :group-count])))))
