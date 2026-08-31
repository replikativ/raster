(ns raster.gpu.compiled-scan-session-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.gpu.core :as gpu]))

(defn- scan-artifact
  []
  (kart/make
   {:kernel-name "session_scan"
    :source (str "__kernel void session_scan(__global const int* input, "
                 "__global int* output, int _n_bound) {}")
    :abi [(kabi/slot 'input :input :int :role :operand)
          (kabi/slot 'output :output :int :role :result)
          (kabi/slot '_n_bound :scalar :int :role :bound)]
    :arguments '[input output n]
    :launch (klaunch/spec {:workgroup-size [64]
                           :group-count [(klaunch/ceil-div 'n 64)]})
    :effects {:kind :scan :reads ['input] :writes ['output]}
    :attributes {:strategy :scheduled-graph}}))

(defn- scan-dispatch
  []
  (kdispatch/make
   {:id "session-scan-dispatch"
    :alternatives [(scan-artifact)]
    :default-strategy :scheduled-graph
    :selector {:kind :fixed-strategy :strategy :scheduled-graph}}))

(deftest compile-retains-first-class-dispatches-beside-flat-kernels
  (let [session (atom {:device-id :ze:0})
        compilations (atom 0)
        compiled {:kernels [:registered-kernel]
                  :dispatches [(scan-dispatch)]}]
    (with-redefs-fn
      {#'raster.gpu.core/compile-deftm-internal!
       (fn [_ _ _] (swap! compilations inc) compiled)}
      #(do
         (is (= [:registered-kernel]
                (gpu/compile! session :first #'identity)))
         (is (= [:registered-kernel]
                (gpu/compile! session :second #'identity)))))
    (is (= 1 @compilations) "the complete compiled result is cached once")
    (is (= (:dispatches compiled) (get-in @session [:dispatches :first])))
    (is (= (:dispatches compiled) (get-in @session [:dispatches :second])))))

(deftest scan-invocation-binds-the-retained-ordered-executable-abi
  (let [dispatch (scan-dispatch)
        session (atom {:dispatches {:scan [dispatch]}
                       :buffers {:output :resident-output}})
        calls (atom [])]
    (with-redefs [gpu/bind-kernel-executable!
                  (fn [actual-session key executable arguments]
                    (swap! calls conj [:bind actual-session key executable arguments])
                    ::handle)
                  gpu/run-kernel-graph!
                  (fn [actual-session handle]
                    (swap! calls conj [:run actual-session handle]))
                  gpu/release-kernel-graph!
                  (fn [actual-session handle]
                    (swap! calls conj [:release actual-session handle]))]
      (is (= :resident-output
             (gpu/invoke-scan! session :scan [:input] :output 513))))
    (let [[_ actual-session key executable arguments] (first @calls)]
      (is (identical? session actual-session))
      (is (= [:compiled-scan :scan] key))
      (is (= (kdispatch/default-alternative dispatch) executable))
      (is (= [:input :output {:type :int :value 513}] arguments)))
    (is (= [[:run session ::handle]
            [:release session ::handle]]
           (subvec @calls 1))))
  (testing "flat kernels cannot silently stand in for a missing scheduled dispatch"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"exactly one KernelDispatch"
         (gpu/invoke-scan! (atom {:kernels {:scan [:block :prop]}})
                           :scan [:input] :output 8)))))
