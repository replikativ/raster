(ns raster.compiler.ir.program-stage-test
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.program-stage :as stage]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]
            [raster.dl.attention :as attention]
            [raster.dl.nn :as nn]))

(deftm staged-attention!
  [q :- (Array float) qr :- (Array float) k :- (Array float) v :- (Array float)
   kc :- (Array float) vc :- (Array float) at :- (Array float) sc :- (Array float)
   output :- (Array float)
   pos :- (Array long) context-length :- (Array long)
   max-position :- Long scale :- Float] :- Void
  (nn/residual-add! q q qr 4)
  (attention/kv-append-buf! k kc 4 pos)
  (attention/kv-append-buf! v vc 4 pos)
  (attention/gqa-decode-attention-buf!
   qr kc vc at sc context-length 1 1 1 4 max-position scale)
  (nn/residual-add! at q output 4))

(defn- descriptor []
  (pipeline/compile-gpu-program #'staged-attention! :ze:0 :dtype :float))

(defn- projection-bindings
  [descriptor]
  (let [symbols (into #{} (mapcat (comp keys stage/step-accesses)) (:steps descriptor))]
    (into {} (map (fn [symbol] [symbol (keyword (name symbol))])) symbols)))

(deftest semantic-stage-is-selected-by-declared-effects-not-step-or-kernel-names
  (let [program (descriptor)
        selected (stage/select program {:id :routed-cache-attention
                                        :state #{'kc 'vc}
                                        :outputs #{'at}
                                        :attributes {:replacement :routed}})
        parts (stage/partition program selected)]
    (is (stage/program-stage? selected))
    (is (= :routed-cache-attention (:id selected)))
    (is (set/subset? #{'k 'v 'qr 'pos 'context-length} (:inputs selected)))
    (is (= #{'kc 'vc} (:state selected)))
    (is (= #{'at} (:outputs selected)))
    (is (seq (get-in parts [:before :steps])))
    (is (seq (get-in parts [:selected :steps])))
    (is (seq (get-in parts [:after :steps])))
    (is (= (count (:steps program))
           (reduce + (map #(count (:steps %)) (vals parts)))))
    (is (= {:id :routed-cache-attention :part :selected
            :source-range [(:start selected) (:end selected)]
            :inputs (:inputs selected) :outputs #{'at} :state #{'kc 'vc}
            :internal #{}
            :attributes {:replacement :routed}}
           (:stage (:selected parts))))
    (testing "a stage output may escape the complete descriptor rather than feed a later step"
      (let [tail (stage/select program {:id :external-output :outputs #{'output}})]
        (is (= (count (:steps program)) (:end tail)))
        (is (nil? (:after (stage/partition program tail))))))
    (testing "the projected descriptors compose through the ordinary public LinkPlan"
      (let [nodes [(link/node {:id :q :dtype :float :shape [4] :device :ze:0
                               :role :input :source (float-array 4)})
                   (link/node {:id :qr :dtype :float :shape [4] :device :ze:0
                               :role :internal})
                   (link/node {:id :k :dtype :float :shape [4] :device :ze:0
                               :role :input :source (float-array 4)})
                   (link/node {:id :v :dtype :float :shape [4] :device :ze:0
                               :role :input :source (float-array 4)})
                   (link/node {:id :kc :dtype :float :shape [32] :device :ze:0
                               :role :state :source (float-array 32)})
                   (link/node {:id :vc :dtype :float :shape [32] :device :ze:0
                               :role :state :source (float-array 32)})
                   (link/node {:id :at :dtype :float :shape [4] :device :ze:0
                               :role :internal})
                   (link/node {:id :sc :dtype :float :shape [8] :device :ze:0
                               :role :internal})
                   (link/node {:id :output :dtype :float :shape [4] :device :ze:0
                               :role :output})
                   (link/node {:id :pos :dtype :long :shape [1] :device :ze:0
                               :role :input :source (long-array [0])})
                   (link/node {:id :context-length :dtype :long :shape [1] :device :ze:0
                               :role :input :source (long-array [1])})]
            scalars {'max-position 8 'scale (float 0.5)}
            instances (mapv (fn [[id descriptor]]
                              (link/instance {:id id :descriptor descriptor
                                              :bindings (projection-bindings descriptor)
                                              :scalars scalars}))
                            [[:pre (:before parts)]
                             [:replaceable (:selected parts)]
                             [:post (:after parts)]])]
        (is (link/link-plan?
             (link/make {:id :staged-attention :target :ze:0 :nodes nodes
                         :instances instances :outputs [:output]})))))))

(deftest staging-fails-loudly-on-ambiguous-or-leaking-effect-contracts
  (let [program (descriptor)]
    (testing "an anchor may not be selected by first/last name-search heuristics"
      (let [at-step (some #(when (= :write (get (stage/step-accesses %) 'at)) %) (:steps program))
            ambiguous (update program :steps conj at-step)]
        (is (= :program-stage-ambiguous-anchor
               (:reason
                (ex-data
                 (try
                   (stage/select ambiguous {:id :ambiguous
                                            :state #{'kc 'vc} :outputs #{'at}})
                   (catch clojure.lang.ExceptionInfo error error))))))))
    (testing "every external write crossing the interval is part of the semantic contract"
      (is (= :program-stage-unclassified-anchor
             (:reason
              (ex-data
               (try
                 (stage/select program {:id :missing-state
                                        :state #{'kc} :outputs #{'at}
                                        :anchors #{'kc 'vc 'at}})
                 (catch clojure.lang.ExceptionInfo error error)))))))))
