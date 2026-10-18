(ns raster.ad.generated-helper-owner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.backend.jvm.source-owner :as source-owner]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.core :as core]
            [raster.par]
            [raster.arrays]
            [raster.numeric]))

(defn- typed [name tag] (with-meta name {:raster.type/tag tag}))

(defn- helper-call [scope name arguments]
  (let [operation (symbol (str (ns-name scope)) (str name))
        source (core/resolve-deftm-var (ns-resolve scope name))
        receiver (symbol (str (ns-name scope)) (str (:name (meta source)) "-impl"))
        definition (entry/selected-definition operation receiver (count arguments))
        receiver (with-meta receiver {:tag (get-in definition [:entry :typed-iface])})]
    (core/ensure-bytecode-implementation! (:receiver-var definition))
    (with-meta (list* '.invk receiver arguments) {:raster.op/original operation})))

(defn- compile-program [params bindings result active]
  (#'reverse/make-runtime-value+grad-fn
   {:kind :generated-product-bindings :source-policy {:unchecked-math false}
    :bindings bindings :result result :active-inputs active}
   params {:generated-source-policy {:unchecked-math false}}))

(deftest prepared-program-retains-helper-ownership-until-compilation
  (let [scope (create-ns (gensym "raster.prepared-helper-owner."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm helper [x :- Double] :- Double (raster.numeric/* x x)))
        (let [x (typed 'x 'double) result (typed 'result 'double)
              call (helper-call scope 'helper [x])
              ;; Preparation must not compile/evaluate the outer program. The
              ;; existing helper producer may compile its private companion.
              prepared (with-redefs [source-owner/compile-function
                                     (fn [& _] (throw (AssertionError. "outer execution compiled during preparation")))]
                         (#'reverse/prepare-generated-value+grad
                          {:kind :generated-product-bindings :source-policy {:unchecked-math false}
                           :bindings [result call] :result result :active-inputs [x]}
                          [x] {:generated-source-policy {:unchecked-math false}}))
              {:keys [region companion-port]} (first (:call-regions prepared))
              capture (some #(when (= companion-port (:port %)) %) (get-in prepared [:source-owner :captures]))]
          (is (= 1 (count (:call-regions prepared))))
          (is (identical? call (:original region)))
          (is (identical? (:value capture) (get-in region [:helper :companion])))
          (is (= {:unchecked-math false} (get-in prepared [:source-owner :source-policy])))
          (is (= :unprojected-jvm-source-owner
                 (try (source-owner/assert-portable-source! prepared)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (let [f (#'reverse/compile-prepared-value+grad prepared)
                [value pb] (f 3.0)]
            (is (= 9.0 value))
            (is (= [12.0] (pb 2.0)))
            (is (= :ad-generated-seed-carrier
                   (try (pb (float 1.0))
                        (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
            (with-redefs-fn {(ns-resolve scope (second call)) (fn [& _] :stale)}
              #(do (is (thrown? clojure.lang.ExceptionInfo (f 3.0)))
                   (is (thrown? clojure.lang.ExceptionInfo (pb 1.0))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest indirect-source-inlining-cannot-drop-private-owner-metadata
  (let [scope (create-ns (gensym "raster.owner-inline-refusal."))]
    (try
      (let [v (intern scope 'owned
                      (with-meta (fn [x] x)
                        {:raster.core/deftm-walked-body ['x]
                         :raster.core/deftm-params ['x]
                         :raster.core/deftm-tags ['double]
                         ;; Even a malformed owner must not become unowned source.
                         :raster.compiler/source-owner nil}))]
        (is (= :unprojected-jvm-source-owner
               (try (#'inline/try-resolve-walked-body v)
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (= :unprojected-jvm-source-owner
               (try (#'inline/resolve-deftm-for-ad (symbol (str (ns-name scope)) "owned"))
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (= :unprojected-jvm-source-owner
               (try (reverse/value+grad v)
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
      (finally (remove-ns (ns-name scope))))))

(deftest flat-owned-preparation-keeps-runtime-guards
  (let [current? (atom true)
        entered (atom 0)
        guard (fn [] (when-not @current?
                       (throw (ex-info "owner changed" {:reason :test-owner-changed}))))
        prepared {:params ['x] :walked-body ['[x]]
                  :source-owner {:source-policy {:unchecked-math false} :source-ns *ns*
                                 :captures [] :reference-bindings {} :methods [] :guard guard}}
        f (#'reverse/compile-prepared-value+grad prepared)]
    (is (= [3.0] (f 3.0)))
    (reset! current? false)
    (is (= :test-owner-changed
           (try (f 4.0) (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    ;; Post-call fencing is independent of paired-result carrier validation.
    (reset! current? true)
    (let [port 'change-owner
          f (#'reverse/compile-prepared-value+grad
             (-> prepared
                 (assoc :walked-body [(list 'do (list port) '[x])])
                 (assoc-in [:source-owner :captures]
                           [{:port port :value #(do (swap! entered inc) (reset! current? false))}])))]
      (is (= :test-owner-changed
             (try (f 5.0) (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= 1 @entered)))))

(deftest paired-helper-unused-input-cotangents-stay-absent-inside-the-owner
  (let [scope (create-ns (gensym "raster.helper-sparse-inputs."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm keep-a [a :- Double b :- Double] :- Double a))
        (eval '(raster.core/deftm choose [a :- Double b :- Double flag :- Long] :- Double
                 (if (= flag 1) b a)))
        (let [x (typed 'x 'double) flag (typed 'flag 'long)
              b (typed 'b 'double) result (typed 'result 'double)
              only-a (compile-program [x]
                                      [b '(raster.numeric/sqrt x)
                                       result (helper-call scope 'keep-a [x b])] result [x])
              choose (compile-program [x flag]
                                      [b '(raster.numeric/sqrt x)
                                       result (helper-call scope 'choose [x b flag])] result [x])]
          (doseq [seed [0.0 1.0 2.0]]
            (doseq [pb [(second (only-a 0.0)) (second (choose 0.0 0))]]
              (is (= [seed] (pb seed))
                  "an unused helper input must not send present zero into sqrt"))
            (let [[value pb] (choose 0.0 1)
                  dx (first (pb seed))]
              (is (= 0.0 value))
              (if (zero? seed)
                (is (Double/isNaN dx))
                (is (= Double/POSITIVE_INFINITY dx)))))))
      (finally (remove-ns (ns-name scope))))))

(deftest generated-owner-composes-a-reduction-helper-and-its-array-pullback
  (let [scope (create-ns (gensym "raster.generated-helper-owner."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm helper
                 [a :- Double b :- Double xs :- (Array double) cnt :- Long] :- Double
                 (raster.par/reduce sum 0.0 i cnt
                   (raster.numeric/+ sum
                     (raster.numeric/* a (raster.numeric/+ b (raster.arrays/aget xs i)))))))
        (let [a (typed 'a 'double) b (typed 'b 'double)
              xs (typed 'xs 'doubles) cnt (typed 'cnt 'long) result (typed 'result 'double)
              call (helper-call scope 'helper [a b xs cnt])
              f (compile-program [a b xs cnt] [result call] result [a b xs])]
          (doseq [[av bv data] [[2.0 3.0 [1.0 2.0]] [-1.5 0.25 [4.0 -2.0 3.0]] [4.0 -3.0 []]]]
            (let [[value pb] (f av bv (double-array data) (long (count data)))
                  da (reduce + 0.0 (map #(+ bv %) data))]
              (is (= (* av da) value))
              (doseq [seed [0.0 1.0 2.0]]
                (let [[ga gb gx] (pb seed)]
                  (is (= (* seed da) ga))
                  (is (= (* seed av (count data)) gb))
                  (is (= (vec (repeat (count data) (* seed av))) (vec gx)))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest generated-owner-checks-helper-staleness-before-forward-and-delayed-backward
  (let [scope (create-ns (gensym "raster.generated-helper-staleness."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm helper [x :- Double] :- Double (raster.numeric/* x x)))
        (let [x (typed 'x 'double) result (typed 'result 'double)
              call (helper-call scope 'helper [x])
              f (compile-program [x] [result call] result [x])
              [_ pb] (f 3.0)
              receiver (ns-resolve scope (second call))]
          (is (= [6.0] (pb 1.0)))
          (with-redefs-fn {receiver (fn [& _] (throw (AssertionError. "stale helper must not execute")))}
            #(do
               (is (thrown? clojure.lang.ExceptionInfo (f 3.0)))
               (is (thrown? clojure.lang.ExceptionInfo (pb 1.0)))))))
      (finally (remove-ns (ns-name scope))))))

(deftest inactive-helpers-are-not-a-mutation-backdoor
  (let [scope (create-ns (gensym "raster.generated-helper-effects."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm helper [xs :- (Array double)] :- Double
                 (do (clojure.core/aset xs 0 7.0) 1.0)))
        (let [x (typed 'x 'double) xs (typed 'xs 'doubles)
              ignored (typed 'ignored 'double) result (typed 'result 'double)
              call (helper-call scope 'helper [xs])]
          (is (= :ad-paired-helper-effects
                 (try (compile-program [x xs] [ignored call result x] result [x]) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest paired-helper-input-edges-round-before-repeated-argument-fanin
  (let [scope (create-ns (gensym "raster.generated-helper-rounding."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm helper [a :- Double b :- Double] :- Double
                 (raster.numeric/+ a b)))
        (let [x (typed 'x 'float) result (typed 'result 'double)
              call (helper-call scope 'helper [x x])
              f (compile-program [x] [result call] result [x])
              [value pb] (f (float 2.0))
              seed (* 0.49 (double Float/MIN_VALUE))]
          (is (= 4.0 value))
          (is (pos? (float (+ seed seed))))
          (is (= [(float 0.0)] (pb seed)))
          (is (= [(float 2.0)] (pb 1.0)))
          (is (instance? Float (first (pb 1.0))))))
      (finally (remove-ns (ns-name scope))))))

(deftest paired-helper-narrowing-completes-both-primal-and-source-cotangent
  (let [scope (create-ns (gensym "raster.generated-helper-narrowing."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm helper [x :- Float] :- Float (raster.numeric/* x x)))
        (let [x (typed 'x 'double) result (typed 'result 'float)
              call (helper-call scope 'helper [x])
              f (compile-program [x] [result call] result [x])
              [value pb] (f 0.1)
              [gradient] (pb (float 1.0))]
          (is (instance? Float value))
          (is (= (float (* (float 0.1) (float 0.1))) value))
          (is (instance? Double gradient))
          (is (= (double (float (* 2.0 (float 0.1)))) gradient))
          (is (= [gradient] (pb (float 1.0))))))
      (finally (remove-ns (ns-name scope))))))

(deftest inactive-pure-helper-runs-under-the-same-owner-guard
  (let [scope (create-ns (gensym "raster.generated-helper-inactive."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm helper [x :- Double] :- Double (raster.numeric/* x x)))
        (let [x (typed 'x 'double) data (typed 'data 'double)
              constant (typed 'constant 'double) result (typed 'result 'double)
              call (helper-call scope 'helper [data])
              f (compile-program [x data] [constant call result '(raster.numeric/* x constant)] result [x])
              [value pb] (f 2.0 3.0)]
          (is (= 18.0 value))
          (is (= [9.0] (pb 1.0)))
          (with-redefs-fn {(ns-resolve scope (second call)) nil}
            #(is (thrown? clojure.lang.ExceptionInfo (f 2.0 3.0))))))
      (finally (remove-ns (ns-name scope))))))

(deftest branch-absence-does-not-execute-a-singular-helper-pullback
  (let [scope (create-ns (gensym "raster.generated-helper-absence."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm helper [x :- Double] :- Double (Math/log x)))
        (let [x (typed 'x 'double) y (typed 'y 'double)
              positive (typed 'positive 'boolean) selected (typed 'selected 'double)
              result (typed 'result 'double)
              call (helper-call scope 'helper [x])
              branch (list 'if positive (list 'let* [selected call] selected) y)
              f (compile-program [x y] [positive '(clojure.core/pos? y) result branch] result [x y])
              [value pb] (f 0.0 -1.0)]
          (is (= -1.0 value))
          (is (= [0.0 1.0] (pb 1.0)))
          (is (= [0.0 0.0] (pb 0.0)))
          (let [[_ present] (f 0.0 1.0)]
            (is (Double/isNaN (double (first (present 0.0))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest helper-result-cotangent-is-projected-before-the-saved-pullback
  (let [scope (create-ns (gensym "raster.generated-helper-result-edge."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm helper [x :- Double] :- Float x))
        (let [x (typed 'x 'double) y (typed 'y 'double)
              h (typed 'h 'float) result (typed 'result 'double)
              call (helper-call scope 'helper [x])
              f (compile-program [x y] [h call result '(raster.numeric/* h y)] result [x y])
              [value pb] (f 2.0 Math/PI)
              [dx dy] (pb 1.0)]
          (is (= (* 2.0 Math/PI) value))
          (is (not= Math/PI (double (float Math/PI))))
          (is (= (double (float Math/PI)) dx))
          (is (instance? Double dx))
          (is (= 2.0 dy))
          (is (= [dx dy] (pb 1.0)))))
      (finally (remove-ns (ns-name scope))))))

(deftest helper-result-fanin-is-projected-at-the-helper-tangent-boundary
  (let [scope (create-ns (gensym "raster.generated-helper-result-fanin."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm helper [x :- Double] :- Float x))
        (let [x (typed 'x 'double) y (typed 'y 'double) z (typed 'z 'double)
              h (typed 'h 'float) a (typed 'a 'double) b (typed 'b 'double)
              result (typed 'result 'double)
              call (helper-call scope 'helper [x])
              f (compile-program [x y z]
                                 [h call a '(raster.numeric/* h y) b '(raster.numeric/* h z)
                                  result '(raster.numeric/+ a b)] result [x y z])
              [value pb] (f 2.0 16777216.0 1.0)
              [dx dy dz] (pb 1.0)]
          (is (= 33554434.0 value))
          (is (= 16777216.0 dx))
          (is (not= 16777217.0 dx))
          (is (instance? Double dx))
          (is (= [2.0 2.0] [dy dz]))))
      (finally (remove-ns (ns-name scope))))))
