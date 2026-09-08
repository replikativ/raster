(ns raster.compiler.passes.parallel.staged-contraction-admission
  "Shared, non-emitting admission for the generated packed staged schedule."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.contract-stages :as stages]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.passes.parallel.staged-contraction-schedule :as schedule]))

(defn decline! [rule message data]
  (throw (ex-info message (assoc data :reason :staged-kernel-body-declined
                                 :missing-rule rule))))

(defn declined? [e]
  (= :staged-kernel-body-declined (:reason (ex-data e))))

(defn- require! [condition rule data]
  (when-not condition (decline! rule "typed staged schedule is not proved" data)))

(defn- extent! [amap domain]
  (let [pairs (vec (mapcat identity (:groups amap)))
        axes (mapv first pairs)]
    (require! (and (seq pairs) (= (count axes) (count (set axes)))
                   (every? (fn [[a n :as pair]]
                             (and (= 2 (count pair)) (contains? domain a)
                                  (= n (get domain a)))) pairs))
              :operand-domain {:map amap :domain domain})
    (let [n (reduce *' 1 (map second pairs))]
      (require! (<= n Integer/MAX_VALUE) :address-range {:map amap :elements n})
      (long n))))

(defn analyze!
  "Check the packed schedule domain without constructing KernelBody or emitting code."
  [source & {:keys [workgroup-size] :or {workgroup-size 64}}]
  (when-not (facts/facts? source)
    (throw (ex-info "staged body requires verified contraction facts" {:reason :raster/bug})))
  (let [{:keys [free-axes contract-axes out]} source
        stage-list (:stages source)
        [outer inner] stage-list
        axes (vec (concat free-axes contract-axes))
        axis-ids (mapv first axes)
        domain (into {} axes)
        operands (vec (get-in source [:opts :operands]))
        lifts (stages/lift-operands stage-list)
        arrays (vec (concat operands lifts))
        array-ids (mapv :sym arrays)
        legality (stages/stages-legal? stage-list contract-axes)
        reduction (facts/scalar-reduction-view source)]
    (require! (:ok legality) :stage-legality legality)
    ;; Decode is semantic evidence on canonical facts, not necessarily copied into the optional
    ;; scheduling declarations. A body-replacing packed leaf must not discard either source.
    (require! (and (not (seq (get-in source [:opts :decode])))
                   (not-any? :decode (:operands source)))
              :decoded-operands {:operands (:operands source)})
    (require! (and (= 2 (count stage-list)) (= :int (dtype/canon (:dtype inner)))
                   (= :float (dtype/canon (:dtype outer)))
                   (= :float (or (:out-dtype source) :float))
                   (nil? (:epilogue source))
                   (contains? '#{+ clojure.core/+ raster.numeric/+} (:combine reduction))
                   (constant/zero-value? (:neutral reduction)))
              :stage-contract {:stages stage-list :reduction reduction})
    (require! (and (seq free-axes) (= (count axis-ids) (count (set axis-ids)))
                   (every? symbol? axis-ids)
                   (every? #(and (integer? %) (pos? %) (<= % Integer/MAX_VALUE))
                           (map second axes))
                   (integer? workgroup-size) (<= 1 workgroup-size 256)
                   (zero? (bit-and workgroup-size (dec workgroup-size))))
              :iteration-domain {:axes axes :workgroup-size workgroup-size})
    (require! (and (= (count array-ids) (count (set array-ids)))
                   (every? symbol? array-ids) (symbol? out)
                   (= (count (concat axis-ids array-ids [out '_nseg 'inner]))
                      (count (set (concat axis-ids array-ids [out '_nseg 'inner]))))
                   (every? #(and (= :float (:dtype %)) (nil? (:decode %))) lifts))
              :parameter-identities {:axes axis-ids :arrays arrays :output out})
    (let [plan (schedule/inner-dp4a-plan
                {:stages stage-list :body (:body source) :operands operands
                 :dtype (:dtype source)})
          _ (require! (:ok plan) :packed-admission plan)
          n (reduce *' 1 (map second free-axes))
          _ (require! (<= (*' workgroup-size (quot (+ n (dec workgroup-size)) workgroup-size))
                          Integer/MAX_VALUE)
                      :launch-range {:elements n})
          sizes (merge (into {} (map (fn [{:keys [sym map]}] [sym (extent! map domain)]) operands))
                       (into {} (map (fn [{:keys [sym map]}]
                                       [sym (extent! map (dissoc domain (:axis inner)))]) lifts)))
          ;; The stage contract gives lift reads their declared maps. Admit scalar factors only;
          ;; arbitrary calls, local binders and casts need their own retained region contract.
          factors (stages/linear-in-inner (:lift outer) 'inner)
          lift-ids (set (map :sym lifts))
          _ (require! (every? #(or (number? %)
                                  (and (descriptor/aget-call? %)
                                       (contains? lift-ids (descriptor/aget-array-sym %)))) factors)
                      :lift-region {:lift (:lift outer)})
]
      {:free-axes free-axes
       :out out
       :stage-list stage-list
       :outer outer
       :inner inner
       :axis-ids axis-ids
       :operands operands
       :lifts lifts
       :array-ids array-ids
       :legality legality
       :plan plan
       :n n
       :sizes sizes})))

