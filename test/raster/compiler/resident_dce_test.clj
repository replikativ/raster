(ns raster.compiler.resident-dce-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.dce :as dce]))

(deftest emitted-submissions-retain-dependencies-without-driver-effect-analysis
  (doseq [head @#'pipeline/gpu-invoke-heads]
    (let [call (list head 'argument)
          source (list 'let* ['argument 7 'submission call 'unused 99] 0)
          original @#'dce/keep-unconditionally?
          analyzed (atom [])
          result (with-redefs-fn
                   {#'dce/keep-unconditionally?
                    (fn [expr]
                      (swap! analyzed conj expr)
                      (when (= expr call)
                        (throw (ex-info "Driver body must not be analyzed" {})))
                      (original expr))}
                   #(#'pipeline/eliminate-dead-resident-bindings source))]
      (is (= (list 'let* ['argument 7 'submission call] 0) (:form result)))
      (is (= 1 (get-in result [:stats :bindings-removed])))
      (is (not-any? #{call} @analyzed))
      ;; Deliberately invalid arity: retaining a marker must not admit it for execution.
      (is (nil? (#'pipeline/parse-gpu-step 'submission call))))))

(deftest resident-cleanup-still-preserves-non-driver-observable-effects
  (let [source '(let* [dead (clojure.core/double-array 1)
                      effect (do (clojure.core/int x)
                                 (clojure.core/aset dead 0 1.0))
                      unused 99]
                     0)
        result (#'pipeline/eliminate-dead-resident-bindings source)
        run (fn [form x]
              (try ((eval (list 'fn '[x] form)) x)
                   (catch ArithmeticException _ :arithmetic-exception)))]
    (is (= 1 (get-in result [:stats :bindings-removed])))
    (doseq [x [1 2147483648]]
      (is (= (run source x) (run (:form result) x))))))

(deftest resident-cleanup-leaves-non-let-forms-unchanged
  (doseq [source [nil 1 '(if flag 1 2)]]
    (is (= {:form source :stats {:bindings-removed 0}}
           (#'pipeline/eliminate-dead-resident-bindings source)))))

(deftest resident-cleanup-preserves-once-only-argument-effects-and-source-metadata
  (let [source (with-meta
                 '(let* [argument (do (swap! log conj :argument) 7)
                         submission (raster.compiler.pipeline/invoke-scheduled-executable!
                                     :probe "unregistered" [argument])
                         unused 99]
                        0)
                 {:line 42 :file "resident-oracle.clj"})
        lowered (:form (#'pipeline/eliminate-dead-resident-bindings source))
        run (fn [form]
              (let [log (atom [])]
                (with-redefs [pipeline/invoke-scheduled-executable!
                              (fn [_ _ arguments] (swap! log conj [:submit arguments]))]
                  ((eval (list 'fn '[log] form)) log))
                @log))]
    (is (= (meta source) (meta lowered)))
    (is (= [:argument [:submit [7]]] (run source)))
    (is (= (run source) (run lowered)))))
