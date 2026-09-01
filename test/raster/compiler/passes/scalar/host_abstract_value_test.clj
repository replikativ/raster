(ns raster.compiler.passes.scalar.host-abstract-value-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.passes.scalar.host-abstract-value :as host-av]))

(def ^:private initial
  (av/tensor {:dtype :double :shape ['?] :representation {:kind :plain}}))

(defn- source
  [copy-length]
  (list 'let*
        ['n '(int (clojure.core/alength u0))
         'u '(clojure.core/aclone u0)
         'scratch '(raster.numeric/similar_m_doubles u0)
         'next '(raster.par/pmap i (clojure.core/alength u) double
                                 (+ (clojure.core/aget u i) 1.0))
         'copy (list 'java.lang.System/arraycopy 'next '(int 0) 'u '(int 0) copy-length)]
        'copy))

(defn- binding-map
  [source]
  (into {} (map vec) (partition 2 (second source))))

(deftest host-shape-refinement-proves-a-complete-state-transition
  (let [analysis (host-av/analyze (source 'n) {:values {'u0 initial} :dtype :double})
        copy-expression (get (binding-map (source 'n)) 'copy)]
    (is (= ['n] (get-in analysis [:values 'u0 :shape])))
    (is (= ['n] (get-in analysis [:values 'u :shape])))
    (is (= ['n] (get-in analysis [:values 'scratch :shape])))
    (is (= ['n] (get-in analysis [:values 'next :shape])))
    (is (= {:source 'next :destination 'u :extent 'n :dtype :double}
           (host-av/full-array-copy analysis copy-expression)))))

(deftest partial-or-offset-copies-are-not-functionalized
  (let [analysis (host-av/analyze (source 'n) {:values {'u0 initial} :dtype :double})
        complete (get (binding-map (source 'n)) 'copy)]
    (testing "a shorter length lacks a full-state proof"
      (is (nil? (host-av/full-array-copy
                 analysis (list 'java.lang.System/arraycopy
                                'next 0 'u 0 '(dec n))))))
    (testing "an offset copy lacks a full-state proof"
      (is (nil? (host-av/full-array-copy
                 analysis (apply list (assoc (vec complete) 2 1))))))
    (testing "extra arguments do not masquerade as the arraycopy primitive"
      (is (nil? (host-av/full-array-copy analysis (concat complete [:extra])))))
    (testing "equal shape and dtype do not bridge incompatible representations"
      (is (nil? (host-av/full-array-copy
                 (assoc-in analysis [:values 'u :representation]
                           {:kind :quantized :scheme :q4-k})
                 complete))))))

(deftest declared-array-types-create-valid-conservative-seeds
  (let [analysis (host-av/analyze '(let* [] nil) {:array-types {'x :float}})]
    (is (= :float (get-in analysis [:values 'x :dtype])))
    (is (= '[(extent x)] (get-in analysis [:values 'x :shape])))))

(deftest retained-cast-and-descriptor-facts-drive-refinement
  (let [float-input (av/tensor {:dtype :float32 :shape ['m]
                                :representation {:kind :plain}})
        analysis
        (host-av/analyze
         '(let* [n (clojure.core/alength x)
                 y (raster.par/pmap i n float (float (clojure.core/aget x i)))
                 unknown (foo/similarity x)]
                y)
         {:values {'x float-input} :dtype :double})]
    (is (= :int (get-in analysis [:values 'n :dtype])))
    (is (= ['n] (get-in analysis [:values 'x :shape])))
    (is (= :float (get-in analysis [:values 'y :dtype])))
    (is (= ['n] (get-in analysis [:values 'y :shape])))
    (is (nil? (get-in analysis [:values 'unknown])))))

(deftest malformed-input-contracts-fail-loud
  (testing "authoritative values must be AbstractValues"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"expected an AbstractValue"
                          (host-av/analyze '(let* [] nil)
                                           {:values {'x {:dtype :double :shape ['n]}}}))))
  (testing "a dangling binding is never silently discarded"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"even binding vector"
                          (host-av/analyze '(let* [x 1 y] x) {})))))
