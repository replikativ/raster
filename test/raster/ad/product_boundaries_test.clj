(ns raster.ad.product-boundaries-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
            [raster.arrays :as ra]
            [raster.par :as par]
            [raster.ad.jvp :as jvp]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.local-products :as products]))

(deftm product-stored-loop [x :- Double] :- Double
  (let [out (float-array 2)
        tuple [(loop [i 0 acc x]
                 (if (< i 2)
                   (let [next (if (n/== i 0) (n/+ acc 0.0)
                                  (let [z (n/- acc 16777216.0)] (n/* z z)))]
                     (ra/aset out i (float next))
                     (recur (inc i) next))
                   out)) nil]
        result (nth tuple 0)]
    (double (ra/aget result 1))))

(deftm inactive-product-stored-loop [h :- Double] :- Double
  (let [out (float-array 2)
        tuple [(loop [i 0 acc 0.5]
                 (if (< i 2)
                   (let [next (n/* acc acc)]
                     (ra/aset out i (float next))
                     (recur (inc i) next))
                   out)) nil]
        result (nth tuple 0)]
    (n/+ h (double (ra/aget result 1)))))

(deftest product-normalization-does-not-hide-inexact-carry-storage
  (let [source (first (core/ensure-walked-body! #'product-stored-loop))
        reason (fn [f] (try (f) nil
                            (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (= 0.25 (product-stored-loop 16777216.5)))
    (is (zero? (let [rounded (double (float 16777216.5))
                     residual (- rounded 16777216.0)]
                 (* residual residual)))
        "using Float output storage as the Double loop carry would change the source result")
    (doseq [prepare [#(reverse/ad-prepare % {'x 'double})
                    #(reverse/transform-body % '[x])]]
      (let [captured (atom []) original products/lower-local-products]
        (with-redefs [products/lower-local-products
                      (fn [body]
                        (swap! captured conj
                               (boolean (some #(:raster.ad.reverse/inexact-source-scan-checkpoint (meta %))
                                              (tree-seq coll? seq body))))
                        (original body))]
          (reason #(prepare source)))
        (is (true? (first @captured))
            "source numerical obligations are captured before either boundary rewrites products")))
    (is (= :ad-scan-carry-precision
           (reason #(reverse/transform-body source '[x]))))
    (doseq [construct [reverse/value+grad jvp/jvp]]
      (let [data (try (construct #'product-stored-loop) nil
                      (catch clojure.lang.ExceptionInfo e (ex-data e)))]
        (is (= :ad-scan-carry-precision (:reason data))
            "both modes retain the shared preparation boundary's precision refusal")
        (is (true? (:raster.ad.reverse/inexact-source-scan-checkpoint
                    (meta (:loop data))))
            "the refusal carries the original inexact-storage witness")))))

(deftest inactive-stored-carry-remains-ordinary-primal-code
  (is (= [2.0625 1.0] ((reverse/value+grad #'inactive-product-stored-loop) 2.0)))
  (is (= [2.0625 1.0] ((jvp/jvp #'inactive-product-stored-loop) 2.0 1.0))))

(deftm passive-double-scan-after-replay
  [h :- Double ys :- (Array double) zs :- (Array double) flag :- Long] :- Double
  (let [value (if (= flag 1)
                (par/reduce total 0.0 i 2 (n/+ total (n/* h (ra/aget ys i))))
                h)
        ignored (if (= flag 1)
                  (par/scan zs acc 0.5 j 2 double (n/* acc acc))
                  zs)]
    value))

(deftm passive-double-scan-overwrites-known-replay
  [h :- Double ys :- (Array double) flag :- Long] :- Double
  (let [value (if (= flag 1)
                (par/reduce total 0.0 i 2 (n/+ total (n/* h (ra/aget ys i))))
                h)
        ignored (par/scan ys acc 0.5 j 2 double (n/* acc acc))]
    value))

(deftest passive-scan-keeps-unused-result-writes-and-replay-guards
  (let [gradient (reverse/value+grad #'passive-double-scan-after-replay :wrt [0])
        source (double-array [1 2]) out (double-array [9 9])]
    (is (= [6.0 3.0 nil nil nil] (gradient 2.0 source out 1)))
    (is (= [0.25 0.0625] (mapv double out)) "unused scan result still writes its destination")
    (let [shared (double-array [1 2])]
      (is (= [2.0 1.0 nil nil nil] (gradient 2.0 shared shared 0)))
      (is (= [1.0 2.0] (mapv double shared)))
      (is (= :ad-replayed-array-alias
             (try (gradient 2.0 shared shared 1) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= [1.0 2.0] (mapv double shared)) "selected alias refusal precedes mutation")))
  (is (= :replayed-array-overwritten
         (try (reverse/value+grad #'passive-double-scan-overwrites-known-replay :wrt [0]) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))

(deftm scan-over-active-map [h :- Double] :- Double
  (let [out (double-array 2)
        mapped (par/map! out i 2 double (n/+ h (double i)))
        ignored (par/scan out acc 0.5 j 2 double (n/* acc acc))]
    (ra/aget out 1)))

(deftest passive-body-does-not-authorize-overwriting-active-storage
  (is (= :ad-active-storage-write
         (try (reverse/value+grad #'scan-over-active-map) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))

(defn- evaluate-gradient [program]
  (let [transformed (reverse/transform-body program '[x])]
    (eval (list 'fn* '[x]
                (list 'let* ['result transformed]
                      [(list 'clojure.core/nth 'result 0)
                       (list (list 'clojure.core/nth 'result 1) 1.0)])))))

(deftest reentrant-transform-normalizes-closed-products
  (let [program '(let* [tuple [x (* x x)]
                       first (clojure.core/nth tuple 0)
                       second (clojure.core/nth tuple 1)]
                   (+ first second))
        gradient (evaluate-gradient program)]
    (doseq [x [-2.0 0.0 1.5]]
      (is (= [(+ x (* x x)) [(+ 1.0 (* 2.0 x))]] (gradient x))))))

(deftest both-ad-boundaries-use-shared-value-normalization
  (let [source '(let* [value [x nil]
                      selected (clojure.core/nth value 0)]
                  (if (clojure.core/nil? selected) 9.0 (* selected selected)))
        expected (products/lower-local-products source)
        prepared (reverse/ad-prepare source)]
    ;; Unknown x is not assumed present: normalization does not erase the
    ;; dynamic branch merely because a closed product held the value.
    (is (some #(= 'clojure.core/nil? %)
              (tree-seq coll? seq expected)))
    (is (not-any? #(= 'clojure.core/nth %)
                  (tree-seq coll? seq prepared)))
    (is (some #(= 'clojure.core/nil? %)
              (tree-seq coll? seq prepared)))))

(deftest normalization-does-not-elide-source-evaluation
  (let [source '(let* [tuple [(do (swap! trace conj :first) x)
                             (do (swap! trace conj :second) (* x x))]
                      second (clojure.core/nth tuple 1)] second)
        prepared (reverse/ad-prepare source)
        execute (fn [program x]
                  ;; Prepared AD is canonical IR, not directly evaluable source.
                  ;; Exercise its actual source projection rather than defining a
                  ;; runtime implementation of the reserved completion term.
                  (eval (list 'let* ['trace '(atom []) 'x x]
                              [(completion/project-to-jvm-source program) '(deref trace)])))]
    (is (some completion/canonical-parts (tree-seq coll? seq prepared))
        "the oracle must cross the canonical completion boundary")
    (doseq [x [-2.0 1.5]]
      (is (= [(* x x) [:first :second]]
             (execute source x) (execute prepared x))))))

(deftest product-source-projection-preserves-a-failing-field-evaluation
  (let [source '(let* [tuple [(do (swap! trace conj :first) x)
                             (do (swap! trace conj :second)
                                 (throw (ex-info "field failure" {})))]
                      selected (clojure.core/nth tuple 0)] selected)
        execute (fn [program]
                  (let [trace (atom [])
                        f (eval (list 'fn ['trace 'x]
                                      (completion/project-to-jvm-source program)))]
                    (try [:value (f trace 1.5) @trace]
                         (catch clojure.lang.ExceptionInfo e
                           [:error (.getMessage e) @trace]))))]
    (is (= [:error "field failure" [:first :second]]
           (execute source) (execute (reverse/ad-prepare source))))))
