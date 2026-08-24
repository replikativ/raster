(ns raster.compiler.ir.attention-ad-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.attention-ad :as attention-ad]))

(defn- problem
  [& overrides]
  (attention/make
   (merge
    {:id :attention
     :query (attention/packed-query-batch
             {:values 'q :row-offsets 'q-offsets :positions 'q-positions :total-tokens 3})
     :k-pages 'k :v-pages 'v :output 'output
     :route (attention/dense-paged-route
             {:page-table 'pages :lengths 'lengths :start-positions 'starts
              :pages-per-sequence 2})
     :batch-size 3 :q-heads 2 :kv-heads 1 :qk-head-dim 2 :value-head-dim 3
     :page-size 2 :physical-pages 3 :scale (/ 1.0 (Math/sqrt 2.0))
     :visibility (attention/visibility {:causal? true :window-left 2 :window-right 0})}
    (apply hash-map overrides))))

(defn- full-vjp
  [& overrides]
  (attention-ad/make
   (merge {:id :attention-vjp :primal (problem) :output-cotangent 'd-output
           :cotangents {:query 'd-q :key 'd-k :value 'd-v}}
          (apply hash-map overrides))))

(deftest semantic-vjp-separates-activity-metadata-and-accumulation
  (let [vjp (full-vjp)
        contract (attention-ad/differentiation-contract vjp)
        specs (attention-ad/buffer-specs vjp)]
    (is (attention-ad/attention-vjp? vjp))
    (is (= #{:query :key :value} (:active-values contract)))
    (is (= {:query 'q :key 'k :value 'v} (:differentiable contract)))
    (is (= {:query-row-offsets 'q-offsets
            :query-positions 'q-positions
            :route '[pages lengths starts]}
           (:nondifferentiable contract)))
    (is (= {:query :write :key :routed-sum :value :routed-sum}
           (:cotangent-accumulation contract)))
    (is (false? (:inference-cache-detached? contract)))
    (is (= [3 2 2] (get-in specs ['d-q :shape])))
    (is (= [1 3 2 2] (get-in specs ['d-k :shape])))
    (is (= [1 3 2 3] (get-in specs ['d-v :shape])))
    (is (= :input (get-in specs ['d-output :role])))
    (is (nil? (get specs 'output)))))

(deftest selective-activity-makes-inference-cache-detachment-explicit
  (let [vjp (attention-ad/make
             {:id :query-only-vjp :primal (problem) :output-cotangent 'd-output
              :active-values #{:query} :cotangents {:query 'd-q}})
        contract (attention-ad/differentiation-contract vjp)]
    (is (= #{:query} (:active-values vjp)))
    (is (:inference-cache-detached? contract))
    (is (= {:query :write} (:cotangent-accumulation contract)))
    (is (nil? (get (attention-ad/buffer-specs vjp) 'd-k)))))

(deftest checkpoint-policy-and-ad-legality-fail-loud
  (testing "saved LSE is an explicit backward input"
    (let [vjp (full-vjp :softmax-state-mode :saved-lse :saved-lse 'lse)]
      (is (= {:mode :saved-lse :buffer 'lse}
             (:softmax-state (attention-ad/differentiation-contract vjp))))
      (is (= [3 2] (get-in (attention-ad/buffer-specs vjp) ['lse :shape])))))
  (testing "discrete metadata cannot be marked active"
    (is (= :attention-invalid-vjp-activity
           (try
             (full-vjp :active-values #{:query :route})
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "activity and cotangent outputs agree exactly"
    (is (= :attention-vjp-cotangent-activity-mismatch
           (try
             (full-vjp :active-values #{:query :key})
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "quantized storage needs a deliberate gradient rule"
    (is (= :attention-quantized-vjp-undeclared
           (try
             (full-vjp
              :primal (problem :k-dtype :byte
                               :k-format {:dtype :byte :quantization :int8 :group-size 32}))
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "derivative buffers cannot masquerade as primals"
    (is (= :attention-vjp-primal-buffer-alias
           (try
             (full-vjp :output-cotangent 'output)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
