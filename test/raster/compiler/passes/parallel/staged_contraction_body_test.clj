(ns raster.compiler.passes.parallel.staged-contraction-body-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.backend.gpu.staged-contraction-fixtures :as fixtures]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.passes.parallel.staged-contraction-body :as staged]
            [raster.gpu.device-probe :as probe]))

(deftest packed-schedule-is-source-free-and-preserves-byte-storage
  (let [source (fixtures/packed-facts 3 5 3 32)
        scheduled (with-redefs [facts/contraction-facts (fn [& _] (throw (Exception. "reparsed source")))]
                    (staged/lower source))]
    (is (nil? (:form source)))
    (is (= source (:source scheduled)))
    (is (= '[a b da db out 15] (:arguments scheduled)))
    (is (= [288 480 9 15 15] (mapv (comp first :shape) (butlast (get-in scheduled [:body :parameters])))))
    (is (= :reassociated (get-in scheduled [:numerics :mode])))
    (doseq [dialect [:opencl-portable :opencl-intel :cuda :hip]]
      (let [artifact (target/emit-artifact "staged_packed" scheduled dialect)]
        (is (= [:byte :byte :float :float :float :int] (mapv :kernel-dtype (:abi artifact))))
        (is (= '[a b da db out 15] (:arguments artifact)))
        (is (re-find #"rstr_dp4a" (:source artifact)))))))

(defn- decline-rule [source]
  (try (staged/lower source) nil
       (catch clojure.lang.ExceptionInfo e
         (when-not (staged/declined? e) (throw e))
         (:missing-rule (ex-data e)))))

(deftest packed-schedule-declines-unproved-domains
  (let [source (fixtures/packed-facts 3 5 3 32)]
    (doseq [[label transform expected]
            [[:seed #(assoc-in % [:stages 1 :init] 1) :stage-legality]
             [:wide-carry #(assoc-in % [:stages 1 :dtype] :long) :stage-contract]
             [:epilogue #(assoc % :epilogue {}) :stage-contract]
             [:missing-layouts #(update % :opts dissoc :operands) :packed-admission]
             [:zero-domain #(assoc-in % [:free-axes 0 1] 0) :iteration-domain]
             [:symbolic-domain #(assoc-in % [:free-axes 0 1] 'rows) :iteration-domain]
             [:decoded #(assoc-in % [:opts :operands 0 :decode] '(fn [x] (- x 1))) :packed-admission]
             [:decode-options #(assoc-in % [:opts :decode] {'a '(- x 1)}) :decoded-operands]
             [:decode-facts #(assoc-in % [:operands 0 :decode] '(- x 1)) :decoded-operands]
             [:bad-index #(assoc-in % [:opts :operands 0 :map] (am/of-axes '[[j 5] [blk 3] [t 32]])) :packed-admission]
             [:inner-lift #(assoc-in % [:stages 0 :operands 0 :map] (am/of-axes '[[t 32]])) :operand-domain]
             [:undeclared-lift #(assoc-in % [:stages 0 :lift] '(* inner (aget other _))) :lift-region]
             [:arbitrary-lift #(assoc-in % [:stages 0 :lift] '(* inner (Math/sin 2.0))) :lift-region]
             [:alias #(assoc % :out 'a) :parameter-identities]]]
      (testing (name label) (is (= expected (decline-rule (transform source))))))
    (is (= :packed-admission (decline-rule (fixtures/packed-facts 1 1 1 131072))))
    (is (= :launch-range (decline-rule (fixtures/packed-facts Integer/MAX_VALUE 1 1 4))))
    (is (= :address-range (decline-rule (fixtures/packed-facts 1 1 600000000 4))))))

(defn- reference [m n blocks width a b da db]
  (vec (for [i (range m) j (range n)]
         (reduce
          (fn [sum blk]
            (let [dot (reduce + (for [t (range width)]
                                  (* (aget ^bytes a (+ (* i blocks width) (* blk width) t))
                                     (aget ^bytes b (+ (* j blocks width) (* blk width) t)))))
                  ;; Round each explicit stage operation, not a single host Double expression.
                  scaled (float (* (float dot) (aget ^floats da (+ (* i blocks) blk))))
                  lifted (float (* scaled (aget ^floats db (+ (* j blocks) blk))))]
              (float (+ sum lifted))))
          (float 0) (range blocks)))))

(deftest typed-packed-stages-match-reference-on-opencl
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "typed staged packed contractions")
    (let [ocl (find-ns 'raster.gpu.ocl-runtime)
          op #(ns-resolve ocl %)
          upload (op 'buffer-of-array)
          free! (op 'free-buffer!)]
      (doseq [width [4 32 64] blocks [1 3]]
        (let [m 3 n 5
              a (byte-array (take (* m blocks width) (cycle [-128 127 -1 0 3])))
              b (byte-array (take (* n blocks width) (cycle [127 -128 0 -1 7 2])))
              da (float-array (map #(Math/scalb 1.0 (int (- (mod % 5) 2))) (range (* m blocks))))
              db (float-array (map #(Math/scalb 1.0 (int (- (mod % 3) 1))) (range (* n blocks))))
              artifact (target/emit-artifact
                        (str "typed_staged_" width "_" blocks)
                        (staged/lower (fixtures/packed-facts m n blocks width)) :opencl-portable)
              buffers (mapv (fn [[array dtype]] (upload array dtype))
                            [[a :byte] [b :byte] [da :float] [db :float]
                             [(float-array (repeat (* m n) -555.0)) :float]])]
          (try
            ((op 'register-kernel!) (:kernel-name artifact) artifact)
            ((op 'launch-registered-bound!)
             ((op 'bind-kernel-call) (call/make artifact (conj buffers {:type :int :value (* m n)}))))
            (is (= (reference m n blocks width a b da db)
                   (vec ((op 'buffer->array) (peek buffers)))) (str width "x" blocks))
            (finally (doseq [buffer (reverse buffers)] (free! buffer)))))))))
