(ns raster.ad.normalized-program-sum-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-map-sum-test :as source]
            [raster.ad.normalized-map-sum-runtime-test :as sum-runtime]
            [raster.ad.normalized-program-bundle-test :as map-pilot]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- reason [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- build [p active] (#'source/build p active))
(defn- execute [b phase env]
  (canonical/validate! b)
  (let [p (get b phase)
        projection ((if (= 2 (count (soac/equations p)))
                      route/materialize-host-map-reduce route/materialize-host-map) p)
        f (binding [*ns* (:source-ns b)] (eval (list 'fn (:parameters projection) (:body projection))))]
    (apply f (mapv env (:inputs projection)))))
(defn- saved [b results size]
  (merge {'size (long size)}
         (select-keys (zipmap (soac/outputs (:primal b)) results)
                      (map :id (filter #(= :owned-array (:storage %)) (vals (:residual-bindings b)))))))
(defn- recertify [b] (assoc b :certificate (#'canonical/certificate b)))
(defn- recertify-evidence [b]
  (recertify (assoc-in b [:derivation :bundle :certificate]
                      (#'normalized/certificate (get-in b [:derivation :bundle])))))
(defn- change-reduction [p f]
  (apply list (assoc (vec p) 2 (assoc (vec (soac/equations p)) 1 (f (second (soac/equations p)))))))

(deftest exact-sum-programs-execute-with-scalar-seed-and-owned-map-residuals
  (doseq [[v dt array cast] [[#'source/weighted-double-sum :double double-array double]
                             [#'source/weighted-float-sum :float float-array float]]
          n [0 3] active '[xs weights]]
    (let [p (#'source/surface-source v dt) old (build p active) b (canonical/adapt-map-sum old)
          prepared (runtime/prepare old)
          x0 (vec (take n [-2.0 0.0 3.0])) w0 (vec (take n [0.5 2.0 -1.0]))
          xs (array x0) weights (array w0) env {'xs xs 'weights weights 'size (long n)}
          dx (array (repeat n 0.25)) seed (cast -0.5)
          results (execute b :primal env) residuals (saved b results n)
          tangent (first (execute b :linear (assoc residuals (get-in b [:interface :tangent-input :id]) dx)))
          [old-y pb] ((:vjp prepared) (mapv env (:inputs prepared)))
          old-gradient (nth (pb seed) (.indexOf ^java.util.List (:inputs prepared) active))
          gradient-env (assoc residuals (get-in b [:interface :cotangent-input :id]) seed)
          expected (mapv (fn [x w] (cast (* -0.5 (if (= active 'xs) (* 2.0 x w) (* x x))))) x0 w0)]
      (is (identical? p (:source b)))
      (is (= (v xs weights n) (first results) old-y))
      (is (= (class seed) (class (first results)) (class tangent)))
      (is (= [] (get-in b [:interface :primal-result :value :shape])
                (get-in b [:interface :tangent-result :value :shape])
                (get-in b [:interface :cotangent-input :value :shape])))
      (is (= [active] (get-in b [:interface :active-inputs])))
      (is (= :checked-scalar-map-sum (get-in b [:derivation :kind])))
      (is (empty? (:physical-bindings b)))
      (is (fingerprint/equivalent? (second (soac/equations p))
                                   (second (soac/equations (:primal b)))))
      (doseq [binding (vals (:residual-bindings b))]
        (is (= :owned (get-in binding [:value :ownership])))
        (is (= (second (first (soac/equations (:primal b)))) (get-in binding [:producer :equation]))))
      (dotimes [i n] (aset xs i (cast 99.0)) (aset weights i (cast -99.0)))
      (let [gradient (first (execute b :transpose gradient-env))]
        (is (= expected (vec gradient) (vec old-gradient)))
        (is (= (* seed tangent) (reduce + 0.0 (map * (seq dx) (seq gradient)))))
        (dotimes [i n] (aset gradient i (cast 77.0)))
        (is (= expected (vec (first (execute b :transpose gradient-env)))))))))

(deftest ordinary-weighted-objective-is-the-coordinate-fd-oracle
  (doseq [[v dt array cast h tolerance] [[#'source/weighted-double-sum :double double-array double 1.0e-5 1.0e-8]
                                        [#'source/weighted-float-sum :float float-array float 0.125 1.0e-6]]
          active '[xs weights]]
    (let [b (canonical/adapt-map-sum (build (#'source/surface-source v dt) active))
          env {'xs (array [-2.0 0.0 3.0]) 'weights (array [0.5 2.0 -1.0]) 'size 3}
          primal (execute b :primal env) residuals (saved b primal 3)
          gradient (first (execute b :transpose (assoc residuals (get-in b [:interface :cotangent-input :id]) (cast 0.5))))]
      (doseq [i (range 3)]
        (let [objective (fn [delta]
                          (let [changed (array (seq (get env active)))
                                _ (aset changed i (cast (+ (aget changed i) delta)))
                                e (assoc env active changed)]
                            (* 0.5 (v (get e 'xs) (get e 'weights) 3))))
              fd (/ (- (objective h) (objective (- h))) (* 2 h))]
          (is (< (Math/abs (- fd (aget gradient i))) tolerance)))))))

(deftest identity-transpose-is-array-free-and-ordered-fp-checkpoints-remain
  (doseq [[dt array cast bits large] [[:float float-array float #(Float/floatToRawIntBits %) 33554432.0]
                                     [:double double-array double #(Double/doubleToRawLongBits %) 18014398509481984.0]]
          n [0 3]]
    (let [p (#'sum-runtime/identity-source dt) b (canonical/adapt-map-sum (build p 'xs))
          x (array (take n [large 1.0 (- large)])) env {'xs x 'weights (array n) 'size (long n)}
          [y] (execute b :primal env)
          [tangent] (execute b :linear {'size (long n) (get-in b [:interface :tangent-input :id]) x})
          [g] (execute b :transpose {'size (long n) (get-in b [:interface :cotangent-input :id]) (cast -0.0)})]
      (is (empty? (:residual-bindings b)))
      (is (empty? (:arrays (soac/operation-parts (first (soac/equations (:transpose b)))))))
      (is (= (bits (cast 0.0)) (bits y) (bits tangent)))
      (is (= (vec (repeat n (bits (cast -0.0)))) (mapv bits g)))))
  (let [b (canonical/adapt-map-sum (build (#'source/source :float) 'xs))]
    (is (= :float (get-in b [:interface :cotangent-input :value :dtype])))))

(deftest both-certificates-cannot-authorize-a-changed-sum-contract
  (let [b (canonical/adapt-map-sum (build (#'source/source :double) 'xs))
        source-path [:derivation :bundle :sum-programs :source]]
    (doseq [change [#(assoc-in % [:interface :cotangent-input :value :dtype] :float)
                    #(assoc-in % [:interface :tangent-result :value :shape] '[size])
                    #(assoc-in % [:interface :cotangent-input :id] 'other)
                    #(assoc-in % [:interface :primal-result :value :ownership] :borrowed)]]
      (is (= :ad-program-bundle-correspondence (reason #(canonical/validate! (recertify (change b)))))))
    (is (= :ad-normalized-sum-interface
           (reason #(canonical/validate! (recertify-evidence
                                          (assoc-in b [:derivation :bundle :sum-programs :interface :seed] 'xs))))))
    (doseq [[change expected]
            [[(fn [eq] (apply list (assoc (vec eq) 3
                                         (apply list (assoc (vec (nth eq 3)) 4
                                                            (soac/lambda-form '[total element] '[(raster.numeric/* total element)]))))))
              :reduction-nonidentity-init]
             [(fn [eq] (apply list (update (vec eq) 3
                                          (fn [op] (apply list (update (vec op) 1 assoc :identities [1.0]))))))
              :reduction-nonidentity-init]
             [(fn [eq] (apply list (update (vec eq) 3
                                          (fn [op] (apply list (update (vec op) 1 assoc :algebra
                                                                      [(assoc (first (:algebra (second op))) :element 'other)]))))))
              :ad-normalized-sum-algebra]]]
      (is (= expected (reason #(canonical/validate!
                               (recertify-evidence (update-in b source-path change-reduction change)))))))
    (is (= :ad-program-bundle-scope (reason #(canonical/adapt-map (get-in b [:derivation :bundle])))))
    (is (= :ad-program-bundle-scope (reason #(canonical/validate!
                                            (recertify (assoc-in b [:derivation :kind] :checked-scalar-map))))))))

(deftest sum-adapter-does-not-reinterpret-map-evidence-or-runtime
  (let [old (#'map-pilot/simple :double {})]
    (is (= :ad-program-bundle-scope (reason #(canonical/adapt-map-sum old))))
    (is (identical? old (normalized/validate! old))))
  (let [b (canonical/adapt-map-sum (build (#'source/source :double) 'xs))
        revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions) current (revisions)
        executions (atom 0) original route/materialize-host-map-reduce]
    (with-redefs-fn {revisions (constantly (update current :compiler-revision inc))
                    #'route/materialize-host-map-reduce (fn [p] (swap! executions inc) (original p))}
      (fn []
        (is (= :ad-normalized-stale-rules
               (reason #(execute b :primal {'xs (double-array 1) 'weights (double-array 1) 'size 1}))))
        (is (zero? @executions))))))
