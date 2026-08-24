(ns raster.compiler.reference.attention-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.attention-ad :as attention-ad]
            [raster.compiler.reference.attention :as reference]))

(def ^:private dims
  {:batch-size 3 :q-heads 2 :kv-heads 1 :qk-head-dim 2 :value-head-dim 2
   :page-size 2 :physical-pages 3})

(defn- query
  []
  (attention/packed-query-batch
   {:values 'q :row-offsets 'q-offsets :positions 'q-positions :total-tokens 3}))

(defn- route
  [kind]
  (case kind
    :dense-paged
    (attention/dense-paged-route
     {:page-table 'pages :lengths 'lengths :start-positions 'starts
      :pages-per-sequence 2})

    :csr-paged
    (attention/csr-paged-route
     {:page-offsets 'page-offsets :page-indices 'page-indices
      :last-page-lengths 'lasts :start-positions 'starts :page-index-capacity 5})))

(defn- problem
  [kind]
  (attention/make
   (merge dims
          {:id [:attention kind] :query (query) :k-pages 'k :v-pages 'v :output 'output
           :route (route kind) :scale (/ 1.0 (Math/sqrt 2.0))
           :k-layout :kv-head-major :v-layout :page-major
           :visibility (attention/visibility
                        {:causal? true :window-left 2 :window-right 0})})))

(defn- data
  [kind]
  (merge
   {'q [0.12 -0.31, 0.27 0.08, -0.19 0.41,
        0.33 -0.22, -0.14 0.29, 0.38 0.17]
    'k [0.11 -0.07, 0.23 0.31,
        -0.29 0.13, 0.37 -0.17,
        0.19 0.43, -0.41 0.05]
    'v [0.17 -0.23, 0.31 0.09,
        -0.27 0.35, 0.41 -0.11,
        0.07 0.29, -0.33 0.21]
    'q-offsets [0 2 2 3]
    'q-positions [1 2 2]
    'starts [0 0 0]
    'd-output [0.21 -0.14, -0.32 0.17,
               0.09 0.28, -0.16 -0.23,
               0.34 0.12, -0.27 0.19]}
   (case kind
     :dense-paged {'pages [1 2, -1 -1, 1 0] 'lengths [3 0 3]}
     :csr-paged {'page-offsets [0 2 2 4]
                 'page-indices [1 2 1 0 -1]
                 'lasts [1 0 1]})))

(defn- vjp
  [kind & overrides]
  (attention-ad/make
   (merge {:id [:attention-vjp kind] :primal (problem kind)
           :output-cotangent 'd-output
           :cotangents {:query 'd-q :key 'd-k :value 'd-v}}
          (apply hash-map overrides))))

(defn- close?
  ([a b] (close? a b 1.0e-9))
  ([a b tolerance]
   (< (Math/abs (- (double a) (double b))) tolerance)))

(defn- arrays-close?
  [a b tolerance]
  (and (= (count a) (count b))
       (every? true? (map #(close? %1 %2 tolerance) a b))))

(defn- objective
  [problem values]
  (reduce + 0.0 (map * (reference/reference-forward problem values) (get values 'd-output))))

(defn- numerical-gradient
  [problem values buffer-id epsilon]
  (let [primal (vec (get values buffer-id))]
    (mapv
     (fn [index]
       (let [x (double (nth primal index))
             plus (assoc primal index (+ x epsilon))
             minus (assoc primal index (- x epsilon))]
         (/ (- (objective problem (assoc values buffer-id plus))
               (objective problem (assoc values buffer-id minus)))
            (* 2.0 epsilon))))
     (range (count primal)))))

(deftest dense-and-csr-routes-have-identical-forward-and-vjp-semantics
  (let [dense-data (data :dense-paged)
        csr-data (data :csr-paged)
        dense-output (reference/reference-forward (problem :dense-paged) dense-data)
        csr-output (reference/reference-forward (problem :csr-paged) csr-data)
        dense-grads (reference/reference-vjp (vjp :dense-paged) dense-data)
        csr-grads (reference/reference-vjp (vjp :csr-paged) csr-data)]
    (is (arrays-close? dense-output csr-output 1.0e-12))
    (doseq [role [:query :key :value]]
      (is (arrays-close? (get dense-grads role) (get csr-grads role) 1.0e-12)
          (str "route-independent " role " cotangent")))))

(deftest routed-vjp-matches-finite-differences-with-gqa-windows-and-empty-packed-row
  (doseq [kind [:dense-paged :csr-paged]
          :let [problem (problem kind)
                values (data kind)
                analytical (reference/reference-vjp (vjp kind) values)]]
    (testing (name kind)
      (doseq [[role buffer-id] [[:query 'q] [:key 'k] [:value 'v]]]
        (let [numerical (numerical-gradient problem values buffer-id 1.0e-6)]
          (is (arrays-close? (get analytical role) numerical 2.0e-8)
              (str role " finite-difference agreement")))))))

(deftest shared-pages-sum-gradient-contributions-and-saved-lse-is-equivalent
  (let [kind :csr-paged
        values (data kind)
        full (reference/reference-vjp (vjp kind) values)
        dy (vec (get values 'd-output))
        first-only (assoc values 'd-output (into (subvec dy 0 8) (repeat 4 0.0)))
        third-only (assoc values 'd-output (into (vec (repeat 8 0.0)) (subvec dy 8 12)))
        first-grad (reference/reference-vjp (vjp kind) first-only)
        third-grad (reference/reference-vjp (vjp kind) third-only)
        {:keys [lse]} (reference/reference-forward-with-state (problem kind) values)
        saved-vjp (vjp kind :softmax-state-mode :saved-lse :saved-lse 'lse)
        saved (reference/reference-vjp saved-vjp (assoc values 'lse lse))]
    (doseq [role [:key :value]]
      (is (arrays-close? (get full role)
                         (mapv + (get first-grad role) (get third-grad role))
                         1.0e-12)
          (str role " accumulates both logical users of shared page 1")))
    (doseq [role [:query :key :value]]
      (is (arrays-close? (get full role) (get saved role) 1.0e-12)
          (str "saved LSE and recomputation agree for " role)))))

(deftest selective-query-vjp-does-not-materialize-detached-cache-gradients
  (let [kind :dense-paged
        query-vjp (attention-ad/make
                   {:id :query-only :primal (problem kind) :output-cotangent 'd-output
                    :active-values #{:query} :cotangents {:query 'd-q}})
        gradients (reference/reference-vjp query-vjp (data kind))]
    (is (= #{:query} (set (keys gradients))))
    (is (= 12 (count (:query gradients))))))
