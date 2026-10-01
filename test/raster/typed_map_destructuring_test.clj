(ns raster.typed-map-destructuring-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.params :as params]
            [raster.par :as par]
            [raster.compiler.equation-first :as equation]))

(deftm squared-norm
  [{:keys [x y z]} :- (HMap :mandatory {:x Double :y Double :z Double}
                            :complete? true)]
  :- Double
  (+ (* x x) (* y y) (* z z)))

(deftm renamed-nested
  [{a :x {:keys [b]} :nested} :- (HMap {:x Double :nested (HMap {:b Long})})]
  :- Double
  (+ a (double b)))

(deftm metadata-norm
  [^{:- (HMap {:x Double})} {:keys [x]}]
  :- Double
  (* x x))

(deftm shadowed-leaf
  [{:keys [x]} :- (HMap {:x Double})] :- Double
  (let [x 7.0] x))

(deftm carried-leaf
  [{:keys [x]} :- (HMap {:x Long})] :- Long
  (loop [x x] (if (< x 3) (recur (inc x)) x)))

(deftm project-map!
  [{:keys [values offset]} :- (HMap {:values (Array float) :offset Float})
   out :- (Array float) n :- Long] :- (Array float)
  (par/map-void! i n (aset out i (+ offset (aget values i))))
  out)

(deftest structured-calls-retain-flat-types-and-jvm-compilation
  (is (= 14.0 (squared-norm {:x 1.0 :y 2.0 :z 3.0})))
  (is (= 14.0 ((params/compile-aot #'squared-norm) {:z 3.0 :x 1.0 :y 2.0})))
  (is (= 5.5 (renamed-nested {:nested {:b 3} :x 2.5})))
  (is (= 4.0 (metadata-norm {:x 2.0})))
  (is (= 7.0 (shadowed-leaf {:x 2.0})))
  (is (= 3 (carried-leaf {:x 0})))
  (let [td (first (vals (:raster.params/treedefs (meta #'squared-norm))))]
    (is (= [[:x] [:y] [:z]] (mapv :path (:leaves td))))
    (is (= '[Double Double Double] (mapv :type (:leaves td)))))
  (doseq [bad [{:x 1.0 :y 2.0} {:x 1.0 :y 2.0 :z 3.0 :extra 4.0} nil]]
    (is (thrown? clojure.lang.ExceptionInfo (squared-norm bad)))))

(deftest structured-ad-reconstructs-the-declared-map
  (is (= [14.0 {:x 2.0 :y 4.0 :z 6.0}]
         ((params/value+grad #'squared-norm) {:x 1.0 :y 2.0 :z 3.0}))))

(deftest flattened-destructuring-enters-the-target-neutral-soac-vertical
  (let [flat-var (:raster.params/flat-var (meta #'project-map!))
        compilation (equation/compile flat-var {:target :cuda:0 :dtype :float})
        plan (get-in compilation [:semantic :attributes :invocation-plan])
        output (float-array 3)]
    (is (= [2.0 3.0 4.0]
           (vec (project-map! {:values (float-array [0 1 2]) :offset (float 2)}
                              output 3))))
    (is (= 4 (count (:parameters plan))))
    (is (= :none (get-in compilation [:stats :fallback])))))

(deftest issue-966-syntax-has-a-front-end-diagnostic
  (testing "inline annotations inside :keys are not silently accepted"
    (try
      (macroexpand
       '(raster.core/deftm bad-map [{:keys [x :- Double y :- Double]}]
          :- Double (+ x y)))
      (is false "invalid syntax reached the bytecode compiler")
      (catch Throwable e
        (let [cause (or (.getCause e) e)]
          (is (= :typed-parameter-destructuring (:reason (ex-data cause)))))))))
