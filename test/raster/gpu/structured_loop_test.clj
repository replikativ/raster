(ns raster.gpu.structured-loop-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as opencl]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-loop-call :as loop-call]
            [raster.compiler.passes.parallel.structured-control-lower :as lower]
            [raster.gpu.structured-loop :as runtime]))

(defn- scheduled-call
  [trip-count]
  (let [extent (av/tensor {:dtype :long :shape []})
        scalar (av/tensor {:dtype :float :shape []})
        tensor (av/tensor {:dtype :float :shape '[n]})
        inner-tensor (av/tensor {:dtype :float :shape '[n-in]})
        equation
        (list '= 'advance '[u-next]
              (list 'map {:index 'i :extent 'n-in}
                    '[u-in] '[alpha-in iteration]
                    (soac/lambda-form
                     '[u-value alpha-value iteration-value]
                     '[(+ u-value alpha-value (* 0.0 iteration-value))])))
        body (soac/make
              (soac/default-program-facts
               {:values {'iteration extent 'n-in extent 'alpha-in scalar
                         'u-in inner-tensor 'u-next inner-tensor}
                :inputs '[iteration n-in alpha-in u-in]
                :equations {'advance (soac/default-equation-facts)}})
              [equation] '[u-next])
        algorithm
        (control/make
         {:id 'time-loop :effects #{} :provenance {:source :test}
          :attributes {:association :sequential}}
         '[iteration steps]
         [{:outer 'n :parameter 'n-in}
          {:outer 'alpha :parameter 'alpha-in}]
         [{:initial 'u0 :parameter 'u-in :result 'u-next :output 'u-final}]
         body
         {'steps extent 'n extent 'alpha scalar 'u0 tensor 'u-final tensor})
        scheduled (lower/schedule algorithm {:target-device :cpu:0 :dtype :float})
        emitted (opencl/generate-kernel-graph
                 (:graph scheduled) :scalar-types {'alpha-in :float 'iteration :long})]
    (loop-call/make
     scheduled emitted
     {'u0 :initial 'u-final :output}
     {'steps {:type :long :value trip-count}
      'n {:type :int :value 64}
      'alpha {:type :float :value 0.5}}
     (if (> trip-count 1) {'u-final :scratch} {}))))

(deftest host-repetition-uses-one-ordinary-graph-call-per-iteration
  (let [events (atom [])
        call (scheduled-call 3)
        outputs
        (runtime/run-with!
         call
         {:bind! (fn [key graph buffers scalars]
                   (let [handle {:key key :buffers buffers}]
                     (swap! events conj [:bind key graph buffers scalars])
                     handle))
          :run! (fn [handle] (swap! events conj [:run handle]))
          :release! (fn [handle] (swap! events conj [:release handle]))})
        binds (filter #(= :bind (first %)) @events)]
    (is (= {'u-final :output} outputs))
    (is (= 3 (count binds)))
    (is (= [{'u-in :initial 'u-next :output}
            {'u-in :output 'u-next :scratch}
            {'u-in :scratch 'u-next :output}]
           (mapv #(nth % 3) binds)))
    (is (= [:bind :run :release :bind :run :release :bind :run :release]
           (mapv first @events)))
    (is (= [0 1 2] (mapv #(nth (nth % 1) 2) binds)))))

(deftest host-repetition-releases-an-iteration-that-throws
  (let [events (atom [])
        call (scheduled-call 1)]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"iteration failed"
         (runtime/run-with!
          call
          {:bind! (fn [& _] (swap! events conj :bind) :handle)
           :run! (fn [_] (swap! events conj :run)
                   (throw (ex-info "iteration failed" {})))
           :release! (fn [_] (swap! events conj :release))})))
    (is (= [:bind :run :release] @events))))

(deftest zero-trip-host-repetition-does-not-contact-the-executor
  (let [call (scheduled-call 0)
        contacted? (atom false)
        touch (fn [& _] (reset! contacted? true))]
    (is (= {'u-final :initial}
           (runtime/run-with! call {:bind! touch :run! touch :release! touch})))
    (is (false? @contacted?))))
