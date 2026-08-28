(ns raster.compiler.passes.parallel.segmented-weighted-reduction-body
  "Apply a verified cooperative schedule to a segmented weighted reduction.

  The first production storage row is routed paged K/V with interval or CSR membership. The
  resulting KernelBody contains all scalar/control, memory, loop-carried online state, and
  subgroup-reduction structure. It contains no OpenCL spelling and preserves the source plan's
  buffer identities so the ordered ABI can be projected independently."
  (:require [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.segmented-weighted-reduction-schedule :as schedule]))

(defn- lit [value type]
  (body/literal value type))

(defn- expr [op type & arguments]
  (body/scalar-expression op type arguments))

(defn- select-expr [condition if-true if-false type]
  (body/scalar-expression :select type [condition if-true if-false]))

(defn- and-expr [left right]
  (select-expr left right (lit false :predicate) :predicate))

(defn- or-expr [left right]
  (select-expr left (lit true :predicate) right :predicate))

(defn- not-expr [value]
  (select-expr value (lit false :predicate) (lit true :predicate) :predicate))

(defn- true-expr []
  (expr :eq :predicate (lit 0 :int) (lit 0 :int)))

(defn- all-expr
  [values]
  (reduce and-expr (lit true :predicate) values))

(defn- compute [id type expression]
  (body/->ScalarCompute (body/value id type) expression))

(defn- load-value
  ([id type buffer coordinates]
   (load-value id type buffer coordinates nil nil))
  ([id type buffer coordinates predicate other]
   (body/->ScalarLoad (body/value id type) buffer (vec coordinates)
                      predicate other :cached)))

(defn- cast-expr [value type]
  (body/cast-expression value type :exact :exact))

(defn- half-or-float->float [value type]
  (if (= :float type)
    (expr :+ :float value (lit 0.0 :float))
    (cast-expr value :float)))

(defn- output-value [value type]
  (if (= :float type)
    (expr :+ :float value (lit 0.0 :float))
    (body/cast-expression value :half :nearest-even :ieee)))

(defn- role-map
  [{:keys [query route k-pages v-pages output visibility]}]
  (merge {(:values query) :query
          (:row-offsets query) :query-rows
          (:positions query) :query-positions
          k-pages :key-cache
          v-pages :value-cache
          (:start-positions route) :kv-start-positions
          output :result}
         (if (attention/dense-paged-route? route)
           {(:page-table route) :page-routing
            (:lengths route) :kv-lengths}
           {(:page-offsets route) :page-row-offsets
            (:page-indices route) :page-routing
            (:last-page-lengths route) :last-page-lengths})
         (when (attention/csr-visibility? visibility)
           {(:row-offsets visibility) :attention-row-offsets
            (:key-indices visibility) :attention-key-indices})))

(defn- parameters
  [plan problem]
  (let [specs (attention/buffer-specs problem)
        roles (role-map problem)
        ids (conj (swr/ordered-input-ids plan) (get-in plan [:output :id]))]
    (mapv (fn [id]
            (let [{:keys [dtype shape role]} (get specs id)]
              (body/->KernelParameter
               id role dtype shape :global (layout/row-major shape dtype) (get roles id role))))
          ids)))

(defn partial-buffer-ids
  "Stable graph-local identities derived from the semantic reduction identity.

  Derivation matters when several tiled reductions are linked into one larger program: private
  buffers from distinct reductions must never become equal merely because they share a phase."
  [plan]
  (let [plan-id (:id (swr/validate! plan))]
    {:valid [:segmented-weighted-reduction plan-id :partial-valid]
     :maximum [:segmented-weighted-reduction plan-id :partial-maximum]
     :denominator [:segmented-weighted-reduction plan-id :partial-denominator]
     :weighted-values [:segmented-weighted-reduction plan-id :partial-weighted-values]}))

(defn- checked-elements
  [owner extents]
  (try
    (reduce (fn [result extent]
              (Math/multiplyExact (long result) (long extent)))
            1 extents)
    (catch ArithmeticException cause
      (throw (ex-info "tiled weighted-reduction temporary extent exceeds signed 64-bit capacity"
                      {:reason :segmented-weighted-reduction-partial-extent-overflow
                       :owner owner :extents (vec extents)}
                      cause)))))

(defn partial-buffer-specs
  "Graph-private storage required by a tiled online weighted-reduction schedule."
  [plan scheduled]
  (let [plan (swr/validate! plan)
        scheduled (schedule/validate! scheduled)
        problem (attention/validate! (:source-operation plan))]
    (when-not (schedule/tiled? scheduled)
      (throw (ex-info "partial buffers require a tiled weighted-reduction schedule"
                      {:reason :segmented-weighted-reduction-partial-schedule
                       :schedule scheduled})))
    (let [tile-count (get-in scheduled [:membership-tiling :tile-count])
          ids (partial-buffer-ids plan)
          prefix [(get-in problem [:query :total-tokens]) (:q-heads problem) tile-count]
          scalar-elements (checked-elements :scalar-state prefix)
          weighted-elements (checked-elements :weighted-values
                                               (conj prefix (:value-head-dim problem)))]
      [{:id (:valid ids) :dtype :int :shape prefix :elements scalar-elements
        :role :partial-valid}
       {:id (:maximum ids) :dtype :float :shape prefix :elements scalar-elements
        :role :partial-maximum}
       {:id (:denominator ids) :dtype :float :shape prefix :elements scalar-elements
        :role :partial-denominator}
       {:id (:weighted-values ids) :dtype :float
        :shape (conj prefix (:value-head-dim problem))
        :elements weighted-elements
        :role :partial-weighted-values}])))

(defn- partial-parameters
  [plan problem scheduled]
  (into (mapv #(assoc % :kind :input) (butlast (parameters plan problem)))
        (map (fn [{:keys [id dtype shape role]}]
               (body/->KernelParameter id :output dtype shape :global
                                       (layout/row-major shape dtype) role)))
        (partial-buffer-specs plan scheduled)))

(defn- merge-parameters
  [plan problem scheduled]
  (let [{:keys [dtype shape role]} (get (attention/buffer-specs problem) (:output problem))]
    (conj
     (mapv (fn [{:keys [id dtype shape role]}]
             (body/->KernelParameter id :input dtype shape :global
                                     (layout/row-major shape dtype) role))
           (partial-buffer-specs plan scheduled))
     (body/->KernelParameter (:output problem) :output dtype shape :global
                             (layout/row-major shape dtype) role))))

(defn- query-metadata-operations
  [{:keys [query batch-size]}]
  (let [offsets (:row-offsets query)
        total (:total-tokens query)]
    [(load-value 'first-query-offset :int offsets [0])
     (load-value 'final-query-offset :int offsets [batch-size])
     (compute 'first-query-offset-valid :predicate
              (expr :eq :predicate 'first-query-offset (lit 0 :int)))
     (compute 'final-query-offset-valid :predicate
              (expr :eq :predicate 'final-query-offset (lit total :int)))
     (compute 'query-metadata-initial-valid :predicate
              (and-expr 'first-query-offset-valid 'final-query-offset-valid))
     (body/->ForLoop
      (body/value 'query-row :int) 0 batch-size 1
      [(body/->LoopArg (body/value 'query-valid-state :predicate)
                       'query-metadata-initial-valid)
       (body/->LoopArg (body/value 'query-batch-state :int) (lit -1 :int))]
      [(load-value 'query-row-start :int offsets ['query-row])
       (load-value 'query-row-end :int offsets
                   [(body/expression :add 'query-row 1)])
       (compute 'query-row-start-valid :predicate
                (expr :ge :predicate 'query-row-start (lit 0 :int)))
       (compute 'query-row-order-valid :predicate
                (expr :ge :predicate 'query-row-end 'query-row-start))
       (compute 'query-row-end-valid :predicate
                (expr :le :predicate 'query-row-end (lit total :int)))
       (compute 'query-row-valid :predicate
                (all-expr ['query-row-start-valid 'query-row-order-valid
                           'query-row-end-valid]))
       (compute 'query-next-valid :predicate
                (and-expr 'query-valid-state 'query-row-valid))
       (compute 'query-after-row-start :predicate
                (expr :ge :predicate 'query-token 'query-row-start))
       (compute 'query-before-row-end :predicate
                (expr :lt :predicate 'query-token 'query-row-end))
       (compute 'query-in-row :predicate
                (and-expr 'query-after-row-start 'query-before-row-end))
       (compute 'query-next-batch :int
                (select-expr 'query-in-row 'query-row 'query-batch-state :int))
       (body/->Yield ['query-next-valid 'query-next-batch])]
      [(body/value 'query-metadata-valid :predicate)
       (body/value 'query-batch :int)]
      {:uniform-iter-args #{'query-valid-state 'query-batch-state}})]))

(defn- dense-route-operations
  [{:keys [route page-size]}]
  (let [capacity (* (:pages-per-sequence route) page-size)]
    [(load-value 'kv-length :int (:lengths route) ['safe-query-batch])
     (load-value 'kv-start-position :int (:start-positions route) ['safe-query-batch])
     (compute 'kv-length-nonnegative :predicate
              (expr :ge :predicate 'kv-length (lit 0 :int)))
     (compute 'kv-length-bounded :predicate
              (expr :le :predicate 'kv-length (lit capacity :int)))
     (compute 'kv-start-nonnegative :predicate
              (expr :ge :predicate 'kv-start-position (lit 0 :int)))
     (compute 'kv-start-long :long (cast-expr 'kv-start-position :long))
     (compute 'kv-length-long :long (cast-expr 'kv-length :long))
     (compute 'kv-end-long :long (expr :+ :long 'kv-start-long 'kv-length-long))
     (compute 'kv-end-bounded :predicate
              (expr :le :predicate 'kv-end-long (lit 2147483648 :long)))
     (compute 'route-valid :predicate
              (all-expr ['kv-length-nonnegative 'kv-length-bounded
                         'kv-start-nonnegative 'kv-end-bounded]))
     (compute 'safe-kv-length :int
              (expr :min :int
                    (expr :max :int 'kv-length (lit 0 :int))
                    (lit capacity :int)))]))

(defn- csr-route-operations
  [{:keys [route page-size]}]
  (let [capacity (:page-index-capacity route)
        token-capacity (* capacity page-size)]
    [(load-value 'page-offset-zero :int (:page-offsets route) [0])
     (load-value 'page-begin :int (:page-offsets route) ['safe-query-batch])
     (load-value 'page-end :int (:page-offsets route)
                 [(body/expression :add 'safe-query-batch 1)])
     (load-value 'final-page-length :int (:last-page-lengths route) ['safe-query-batch])
     (load-value 'kv-start-position :int (:start-positions route) ['safe-query-batch])
     (compute 'page-zero-valid :predicate
              (expr :eq :predicate 'page-offset-zero (lit 0 :int)))
     (compute 'page-begin-nonnegative :predicate
              (expr :ge :predicate 'page-begin (lit 0 :int)))
     (compute 'page-order-valid :predicate
              (expr :ge :predicate 'page-end 'page-begin))
     (compute 'page-end-bounded :predicate
              (expr :le :predicate 'page-end (lit capacity :int)))
     (compute 'kv-start-nonnegative :predicate
              (expr :ge :predicate 'kv-start-position (lit 0 :int)))
     (compute 'routed-page-count :int (expr :- :int 'page-end 'page-begin))
     (compute 'has-routed-pages :predicate
              (expr :gt :predicate 'routed-page-count (lit 0 :int)))
     (compute 'empty-last-page-valid :predicate
              (expr :eq :predicate 'final-page-length (lit 0 :int)))
     (compute 'last-page-positive :predicate
              (expr :ge :predicate 'final-page-length (lit 1 :int)))
     (compute 'last-page-bounded :predicate
              (expr :le :predicate 'final-page-length (lit page-size :int)))
     (compute 'nonempty-last-page-valid :predicate
              (and-expr 'last-page-positive 'last-page-bounded))
     (compute 'last-page-valid :predicate
              (select-expr 'has-routed-pages 'nonempty-last-page-valid
                           'empty-last-page-valid :predicate))
     (compute 'computed-kv-length :int
              (expr :+ :int
                    (expr :* :int
                          (expr :- :int 'routed-page-count (lit 1 :int))
                          (lit page-size :int))
                    'final-page-length))
     (compute 'kv-length :int
              (select-expr 'has-routed-pages 'computed-kv-length (lit 0 :int) :int))
     (compute 'kv-start-long :long (cast-expr 'kv-start-position :long))
     (compute 'kv-length-long :long (cast-expr 'kv-length :long))
     (compute 'kv-end-long :long (expr :+ :long 'kv-start-long 'kv-length-long))
     (compute 'kv-end-bounded :predicate
              (expr :le :predicate 'kv-end-long (lit 2147483648 :long)))
     (compute 'route-valid :predicate
              (all-expr ['page-zero-valid 'page-begin-nonnegative 'page-order-valid
                         'page-end-bounded 'kv-start-nonnegative 'last-page-valid
                         'kv-end-bounded]))
     (compute 'safe-page-begin :int
              (expr :min :int
                    (expr :max :int 'page-begin (lit 0 :int))
                    (lit (dec capacity) :int)))
     (compute 'safe-kv-length :int
              (expr :min :int
                    (expr :max :int 'kv-length (lit 0 :int))
                    (lit token-capacity :int)))]))

(defn- interval-membership-operations
  [{:keys [visibility]}]
  (let [{:keys [causal? window-left window-right]} visibility
        begin-expression
        (if (some? window-left)
          (expr :min :long
                (expr :max :long
                      (expr :- :long
                            (expr :- :long 'query-position-long (lit window-left :long))
                            'kv-start-long)
                      (lit 0 :long))
                'safe-kv-length-long)
          (lit 0 :long))
        end-expression
        (cond
          causal?
          (expr :max :long
                (expr :min :long 'safe-kv-length-long
                      (expr :+ :long
                            (expr :- :long 'query-position-long 'kv-start-long)
                            (lit 1 :long)))
                (lit 0 :long))

          (some? window-right)
          (expr :max :long
                (expr :min :long 'safe-kv-length-long
                      (expr :+ :long
                            (expr :- :long
                                  (expr :+ :long 'query-position-long
                                        (lit window-right :long))
                                  'kv-start-long)
                            (lit 1 :long)))
                (lit 0 :long))

          :else 'safe-kv-length-long)]
    [(compute 'safe-kv-length-long :long (cast-expr 'safe-kv-length :long))
     (compute 'attention-begin :long
              (expr :+ :long begin-expression (lit 0 :long)))
     (compute 'attention-end :long
              (expr :+ :long end-expression (lit 0 :long)))
     (compute 'membership-valid :predicate (true-expr))]))

(defn- csr-membership-operations
  [{:keys [visibility]}]
  (let [capacity (:key-index-capacity visibility)]
    [(load-value 'attention-offset-zero :int (:row-offsets visibility) [0])
     (load-value 'raw-attention-begin :int (:row-offsets visibility) ['query-token])
     (load-value 'raw-attention-end :int (:row-offsets visibility)
                 [(body/expression :add 'query-token 1)])
     (compute 'attention-zero-valid :predicate
              (expr :eq :predicate 'attention-offset-zero (lit 0 :int)))
     (compute 'attention-begin-nonnegative :predicate
              (expr :ge :predicate 'raw-attention-begin (lit 0 :int)))
     (compute 'attention-order-valid :predicate
              (expr :ge :predicate 'raw-attention-end 'raw-attention-begin))
     (compute 'attention-end-bounded :predicate
              (expr :le :predicate 'raw-attention-end (lit capacity :int)))
     (compute 'membership-valid :predicate
              (all-expr ['attention-zero-valid 'attention-begin-nonnegative
                         'attention-order-valid 'attention-end-bounded]))
     (compute 'attention-begin :int
              (expr :min :int
                    (expr :max :int 'raw-attention-begin (lit 0 :int))
                    (lit capacity :int)))
     (compute 'attention-end :int
              (expr :min :int
                    (expr :max :int 'raw-attention-end (lit 0 :int))
                    (lit capacity :int)))]))

(defn- position-visible-expression
  [{:keys [causal? window-left window-right]}]
  (all-expr
   (cond-> []
     causal?
     (conj (expr :le :predicate 'kv-position 'query-position-long))

     (some? window-left)
     (conj (expr :ge :predicate 'kv-position
                 (expr :- :long 'query-position-long (lit window-left :long))))

     (some? window-right)
     (conj (expr :le :predicate 'kv-position
                 (expr :+ :long 'query-position-long (lit window-right :long)))))))

(defn- cache-coordinates
  [layout kv-head physical-page page-token component]
  (case layout
    :kv-head-major [kv-head physical-page page-token component]
    :page-major [physical-page page-token kv-head component]))

(defn- member-token-operations
  [{:keys [visibility]} member]
  (if (attention/csr-visibility? visibility)
    [(load-value 'logical-token :int (:key-indices visibility) [member])
     (compute 'logical-token-nonnegative :predicate
              (expr :ge :predicate 'logical-token (lit 0 :int)))
     (compute 'logical-token-bounded :predicate
              (expr :lt :predicate 'logical-token 'safe-kv-length))
     (compute 'logical-token-valid :predicate
              (and-expr 'logical-token-nonnegative 'logical-token-bounded))
     (compute 'logical-token-long :long (cast-expr 'logical-token :long))]
    [(compute 'logical-token-long :long
              (expr :+ :long member (lit 0 :long)))
     (compute 'logical-token-valid :predicate (true-expr))]))

(defn- physical-page-operations
  [{:keys [route page-size physical-pages]}]
  (if (attention/dense-paged-route? route)
    [(compute 'logical-page :long
              (expr :quot :long 'logical-token-long (lit page-size :long)))
     (compute 'safe-logical-page :long
              (expr :min :long
                    (expr :max :long 'logical-page (lit 0 :long))
                    (lit (dec (:pages-per-sequence route)) :long)))
     (load-value 'physical-page :int (:page-table route)
                 ['safe-query-batch 'safe-logical-page])
     (compute 'page-token :long
              (expr :- :long 'logical-token-long
                    (expr :* :long 'safe-logical-page (lit page-size :long))))
     (compute 'physical-page-nonnegative :predicate
              (expr :ge :predicate 'physical-page (lit 0 :int)))
     (compute 'physical-page-bounded :predicate
              (expr :lt :predicate 'physical-page (lit physical-pages :int)))]
    (let [capacity (:page-index-capacity route)]
      [(compute 'logical-page :long
                (expr :quot :long 'logical-token-long (lit page-size :long)))
       (compute 'safe-page-begin-long :long (cast-expr 'safe-page-begin :long))
       (compute 'raw-page-index :long
                (expr :+ :long 'safe-page-begin-long 'logical-page))
       (compute 'safe-page-index :long
                (expr :min :long
                      (expr :max :long 'raw-page-index (lit 0 :long))
                      (lit (dec capacity) :long)))
       (load-value 'physical-page :int (:page-indices route) ['safe-page-index])
       (compute 'page-token :long
                (expr :- :long 'logical-token-long
                      (expr :* :long 'logical-page (lit page-size :long))))
       (compute 'physical-page-nonnegative :predicate
                (expr :ge :predicate 'physical-page (lit 0 :int)))
       (compute 'physical-page-bounded :predicate
                (expr :lt :predicate 'physical-page (lit physical-pages :int)))])))

(defn- dot-operations
  [{:keys [query k-pages qk-head-dim q-dtype k-dtype k-layout] :as problem}
   subgroup-size]
  [(body/->ForLoop
    (body/value 'qk-component :int) 'lane qk-head-dim subgroup-size
    [(body/->LoopArg (body/value 'partial-dot-state :float) (lit 0.0 :float))]
    [(load-value 'query-element q-dtype (:values query)
                 ['query-token 'query-head 'qk-component])
     (load-value 'key-element k-dtype k-pages
                 (cache-coordinates k-layout 'kv-head 'safe-physical-page
                                    'page-token 'qk-component))
     (compute 'query-float :float (half-or-float->float 'query-element q-dtype))
     (compute 'key-float :float (half-or-float->float 'key-element k-dtype))
     (compute 'qk-product :float (expr :* :float 'query-float 'key-float))
     (compute 'partial-dot-next :float
              (expr :+ :float 'partial-dot-state 'qk-product))
     (body/->Yield ['partial-dot-next])]
    [(body/value 'partial-dot :float)]
    {})
   (body/->Collective
    (body/value 'dot :float) :reduce :subgroup subgroup-size 'partial-dot :+ nil
    (body/full-participation) :implementation-defined)
   (compute 'logit :float (expr :* :float 'dot (lit (:scale problem) :float)))])

(defn- value-load-operations
  [{:keys [v-pages v-dtype v-layout]} slots]
  (mapcat
   (fn [{:keys [safe-id mask-id]}]
     [(load-value (symbol (str "value-element-" (name safe-id))) v-dtype v-pages
                  (cache-coordinates v-layout 'kv-head 'safe-physical-page
                                     'page-token safe-id)
                  mask-id (lit 0.0 v-dtype))
      (compute (symbol (str "value-float-" (name safe-id))) :float
               (half-or-float->float
                (symbol (str "value-element-" (name safe-id))) v-dtype))])
   slots))

(defn- online-update-region
  [slots]
  (let [next-accs (mapv :next-id slots)]
    [(compute 'maximum-is-nan :predicate
              (body/scalar-expression :isnan :predicate ['maximum-state]))
     (compute 'logit-is-nan :predicate
              (body/scalar-expression :isnan :predicate ['logit]))
     (compute 'online-state-is-nan :predicate
              (or-expr 'maximum-is-nan 'logit-is-nan))
     (body/->IfRegion
      'online-state-is-nan
      [(body/->Yield [(lit Double/NaN :float)
                      (lit Double/NaN :float)
                      (lit Double/NaN :float)])]
      [(compute 'next-maximum-valid :float (expr :max :float 'maximum-state 'logit))
       (compute 'old-weight-valid :float
                (expr :exp :float (expr :- :float 'maximum-state 'next-maximum-valid)))
       (compute 'new-weight-valid :float
                (expr :exp :float (expr :- :float 'logit 'next-maximum-valid)))
       (body/->Yield ['next-maximum-valid 'old-weight-valid 'new-weight-valid])]
      [(body/value 'next-maximum :float)
       (body/value 'old-weight :float)
       (body/value 'new-weight :float)])
     (compute 'next-denominator :float
              (expr :+ :float
                    (expr :* :float 'denominator-state 'old-weight)
                    'new-weight))
     (vec
      (mapcat
       (fn [{:keys [binding-id safe-id next-id]}]
         [(compute next-id :float
                   (expr :+ :float
                         (expr :* :float binding-id 'old-weight)
                         (expr :* :float
                               (symbol (str "value-float-" (name safe-id)))
                               'new-weight)))])
       slots))
     (body/->Yield (vec (concat ['member-valid-next 'next-maximum 'next-denominator]
                                next-accs)))]))

(defn- flatten-operations
  [operations]
  (vec (mapcat #(if (and (vector? %) (not (record? %))) % [%]) operations)))

(defn- membership-loop
  [problem schedule slots lower upper]
  (let [csr? (attention/csr-visibility? (:visibility problem))
        member (if csr? 'membership-edge 'membership-token)
        member-type (if csr? :int :long)
        acc-bindings (mapv :binding-id slots)
        acc-results (mapv :result-id slots)
        state-bindings (vec (concat [(body/value 'member-valid-state :predicate)
                                     (body/value 'maximum-state :float)
                                     (body/value 'denominator-state :float)]
                                    (map #(body/value % :float) acc-bindings)))
        state-initials (vec (concat ['initial-valid
                                     (lit -3.402823466e38 :float)
                                     (lit 0.0 :float)]
                                    (repeat (count slots) (lit 0.0 :float))))
        state-results (vec (concat [(body/value 'final-valid :predicate)
                                    (body/value 'final-maximum :float)
                                    (body/value 'final-denominator :float)]
                                   (map #(body/value % :float) acc-results)))
        token-ops (member-token-operations problem member)
        page-ops (physical-page-operations problem)
        filter-expr (position-visible-expression
                     (attention/position-filter (:visibility problem)))
        dot-ops (dot-operations problem (:workgroup-size schedule))
        value-ops (value-load-operations problem slots)
        update-region (flatten-operations (online-update-region slots))
        unchanged (vec (concat ['member-valid-next 'maximum-state 'denominator-state]
                               acc-bindings))
        body-ops
        (flatten-operations
         (concat
          token-ops
          [(compute 'kv-position :long
                    (expr :+ :long 'kv-start-long 'logical-token-long))
           (compute 'position-visible :predicate
                    (and-expr filter-expr (true-expr)))]
          page-ops
          [(compute 'physical-page-valid :predicate
                    (and-expr 'physical-page-nonnegative 'physical-page-bounded))
           (compute 'member-applies :predicate
                    (all-expr ['member-valid-state 'logical-token-valid
                               'position-visible 'physical-page-valid]))
           (compute 'member-invalid :predicate
                    (or-expr (not-expr 'logical-token-valid)
                             (and-expr 'position-visible
                                       (not-expr 'physical-page-valid))))
           (compute 'member-valid-next :predicate
                    (and-expr 'member-valid-state (not-expr 'member-invalid)))
           (compute 'safe-physical-page :int
                    (expr :min :int
                          (expr :max :int 'physical-page (lit 0 :int))
                          (lit (dec (:physical-pages problem)) :int)))]
          dot-ops value-ops
          [(body/->IfRegion
            'member-applies update-region
            [(body/->Yield unchanged)]
            (vec (concat [(body/value 'member-valid-result :predicate)
                          (body/value 'maximum-result :float)
                          (body/value 'denominator-result :float)]
                         (map #(body/value % :float)
                              (map :iteration-result-id slots)))))
           (body/->Yield
            (vec (concat ['member-valid-result 'maximum-result 'denominator-result]
                         (map :iteration-result-id slots))))]))]
    (body/->ForLoop
     (body/value member member-type) lower upper 1
     (mapv body/->LoopArg state-bindings state-initials)
     body-ops state-results {})))

(defn- component-slots
  [schedule]
  (mapv (fn [slot]
          {:slot slot
           :id (symbol (str "value-component-" slot))
           :safe-id (symbol (str "safe-value-component-" slot))
           :mask-id (keyword (str "active-value-component-" slot))
           :binding-id (symbol (str "weighted-value-state-" slot))
           :next-id (symbol (str "weighted-value-next-" slot))
           :iteration-result-id (symbol (str "weighted-value-iteration-" slot))
           :result-id (symbol (str "weighted-value-result-" slot))})
        (range (get-in schedule [:value-mapping :components-per-lane]))))

(defn- slot-indices
  [problem scheduled slots]
  (mapcat (fn [{:keys [slot id safe-id]}]
            [(body/->IndexCompute
              id (body/expression :add 'lane (* slot (:workgroup-size scheduled))))
             (body/->IndexCompute
              safe-id (body/expression :min id (dec (:value-head-dim problem))))])
          slots))

(defn- slot-masks
  [problem slots]
  (mapv (fn [{:keys [id mask-id]}]
          (body/->Mask mask-id [(body/predicate :lt id (:value-head-dim problem))]))
        slots))

(defn- lane-zero-mask []
  (body/->Mask :lane-zero [(body/predicate :eq 'lane 0)]))

(defn- output-operations
  [problem slots]
  (flatten-operations
   (concat
    [(compute 'denominator-is-zero :predicate
              (expr :eq :predicate 'final-denominator (lit 0.0 :float)))]
    (mapcat
     (fn [{:keys [slot id mask-id result-id]}]
       (let [normalized (symbol (str "normalized-value-" slot))
             valid-value (symbol (str "valid-output-value-" slot))
             stored (symbol (str "stored-output-value-" slot))]
         [(compute normalized :float
                   (select-expr 'denominator-is-zero (lit 0.0 :float)
                                (expr :div :float result-id 'final-denominator) :float))
          (compute valid-value :float
                   (select-expr 'final-valid normalized (lit Double/NaN :float) :float))
          (compute stored (:output-dtype problem)
                   (output-value valid-value (:output-dtype problem)))
          (body/->ScalarStore (:output problem)
                              ['query-token 'query-head id] stored mask-id)]))
     slots))))

(defn- lowering-row!
  [plan scheduled problem]
  (let [visibility-kind (attention/visibility-kind (:visibility problem))
        expected-traversal (if (= :csr visibility-kind)
                             :csr-row
                             :contiguous-interval)]
    (when-not
     (and (swr/online-softmax-algebra? plan)
          (= :attention (get-in plan [:provenance :semantic-op]))
          (= :canonical-segmented-weighted-reduction
             (get-in plan [:provenance :lowering]))
          (= (attention/ordered-input-buffer-ids problem)
             (swr/ordered-input-ids plan))
          (= (:value-head-dim problem) (get-in plan [:value :components]))
          (= (:qk-head-dim problem) (get-in plan [:score :axis :extent]))
          (= (get-in plan [:score :axis :name])
             (get-in scheduled [:score-reduction :axis]))
          (= (:accumulator-dtype plan)
             (get-in scheduled [:numerical-mode :score-accumulate])
             (get-in scheduled [:numerical-mode :state-accumulate]))
          (= (:value-head-dim problem)
             (get-in scheduled [:value-mapping :components]))
          (= :routed-paged-kv (get-in scheduled [:attributes :storage-kind]))
          (= (attention/route-kind (:route problem))
             (get-in scheduled [:attributes :route-kind]))
          (= visibility-kind (get-in scheduled [:attributes :visibility-kind]))
          (= expected-traversal (:membership-traversal scheduled))
          (= (:output problem) (get-in plan [:output :id])))
      (throw (ex-info
              "scheduled weighted-reduction body does not match its semantic plan"
              {:reason :segmented-weighted-reduction-body-plan-mismatch
               :plan-id (:id plan)
               :schedule scheduled
               :operation-id (:id problem)})))
    [plan scheduled problem]))

(defn- sequential-lowering-row!
  [plan scheduled problem]
  (lowering-row! plan scheduled problem)
  (when (schedule/tiled? scheduled)
    (throw (ex-info "single-body weighted reduction cannot consume a tiled schedule"
                    {:reason :segmented-weighted-reduction-body-schedule-phase
                     :expected :sequential :schedule scheduled})))
  [plan scheduled problem])

(defn- tiled-lowering-row!
  [plan scheduled problem]
  (lowering-row! plan scheduled problem)
  (when-not (schedule/tiled? scheduled)
    (throw (ex-info "partial/merge bodies require a tiled weighted-reduction schedule"
                    {:reason :segmented-weighted-reduction-body-schedule-phase
                     :expected :tiled :schedule scheduled})))
  [plan scheduled problem])

(defn lower-routed-paged
  "Construct the verified KernelBody for the routed paged online schedule.

  This pass accepts the generic SWR plan and schedule values. The initial storage lowering is
  deliberately strict: it recognizes only the already-validated routed-paged descriptor emitted
  by canonical attention lowering and otherwise fails before target emission."
  [plan scheduled]
  (let [plan (swr/validate! plan)
        scheduled (schedule/validate! scheduled)
        problem (attention/validate! (:source-operation plan))
        _ (sequential-lowering-row! plan scheduled problem)
        subgroup-size (:workgroup-size scheduled)
        slots (component-slots scheduled)
        query-ops (query-metadata-operations problem)
        route-ops (if (attention/dense-paged-route? (:route problem))
                    (dense-route-operations problem)
                    (csr-route-operations problem))
        membership-ops (if (attention/interval-visibility? (:visibility problem))
                         (interval-membership-operations problem)
                         (csr-membership-operations problem))
        slot-indices (slot-indices problem scheduled slots)
        masks (slot-masks problem slots)
        initial-ops
        (flatten-operations
         (concat
          query-ops
          [(load-value 'query-position :int (get-in problem [:query :positions])
                       ['query-token])
           (compute 'query-position-valid :predicate
                    (expr :ge :predicate 'query-position (lit 0 :int)))
           (compute 'query-batch-valid :predicate
                    (expr :ge :predicate 'query-batch (lit 0 :int)))
           (compute 'safe-query-batch :int
                    (expr :min :int
                          (expr :max :int 'query-batch (lit 0 :int))
                          (lit (dec (:batch-size problem)) :int)))
           (compute 'query-position-long :long (cast-expr 'query-position :long))]
          route-ops membership-ops
          [(compute 'initial-valid :predicate
                    (all-expr ['query-metadata-valid 'query-position-valid
                               'query-batch-valid 'route-valid 'membership-valid]))
           (compute 'kv-head :int
                    (expr :quot :int 'query-head
                          (lit (quot (:q-heads problem) (:kv-heads problem)) :int)))
           (membership-loop problem scheduled slots 'attention-begin 'attention-end)]
          (output-operations problem slots)))
        launch (launch/spec {:workgroup-size [subgroup-size 1]
                             :group-count [(:q-heads problem)
                                           (get-in problem [:query :total-tokens])]})]
    (body/make
     {:id [:segmented-weighted-reduction-body (:id plan) scheduled]
      :parameters (parameters plan problem)
      :stable-reads (mapv body/stable-read (swr/ordered-input-ids plan))
      :indices (vec (concat [(body/->IndexBinding 'query-head :group 0)
                             (body/->IndexBinding 'query-token :group 1)
                             (body/->IndexBinding 'lane :lane 0)]
                            slot-indices))
      :masks masks
      :operations initial-ops
      :schedule (assoc scheduled :subgroup-size subgroup-size)
      :launch launch
      :provenance {:operation-id (:id problem)
                   :semantic-op :segmented-weighted-reduction
                   :algebra-plan-id (:id plan)
                   :lowering :scheduled-kernel-body}
      :attributes {:storage-kind :routed-paged-kv
                   :route-kind (attention/route-kind (:route problem))
                   :visibility-kind (attention/visibility-kind (:visibility problem))}})))

(defn- tile-bound-operations
  [problem scheduled]
  (let [csr? (attention/csr-visibility? (:visibility problem))
        type (if csr? :int :long)
        tile-size (get-in scheduled [:membership-tiling :tile-size])
        offset (if csr? 'history-tile-offset-int 'history-tile-offset)]
    (vec
     (concat
      [(compute 'history-tile-offset-int :int
                (expr :* :int 'history-tile (lit tile-size :int)))]
      (when-not csr?
        [(compute 'history-tile-offset :long
                  (cast-expr 'history-tile-offset-int :long))])
      [(compute 'membership-count type
                (expr :- type 'attention-end 'attention-begin))
       (compute 'tile-relative-begin type
                (expr :min type 'membership-count offset))
       (compute 'tile-membership-begin type
                (expr :+ type 'attention-begin 'tile-relative-begin))
       (compute 'tile-membership-remaining type
                (expr :- type 'membership-count 'tile-relative-begin))
       (compute 'tile-membership-width type
                (expr :min type 'tile-membership-remaining (lit tile-size type)))
       (compute 'tile-membership-end type
                (expr :+ type 'tile-membership-begin 'tile-membership-width))]))))

(defn- partial-store-operations
  [partial-ids slots]
  (flatten-operations
   (concat
    [(compute 'partial-valid-int :int
              (select-expr 'final-valid (lit 1 :int) (lit 0 :int) :int))
     (body/->ScalarStore (:valid partial-ids)
                         ['query-token 'query-head 'history-tile]
                         'partial-valid-int :lane-zero)
     (body/->ScalarStore (:maximum partial-ids)
                         ['query-token 'query-head 'history-tile]
                         'final-maximum :lane-zero)
     (body/->ScalarStore (:denominator partial-ids)
                         ['query-token 'query-head 'history-tile]
                         'final-denominator :lane-zero)]
    (mapcat
     (fn [{:keys [id mask-id result-id]}]
       [(body/->ScalarStore (:weighted-values partial-ids)
                            ['query-token 'query-head 'history-tile id]
                            result-id mask-id)])
     slots))))

(defn lower-routed-paged-partial
  "Lower each statically bounded membership tile to a private mergeable online state."
  [plan scheduled]
  (let [plan (swr/validate! plan)
        scheduled (schedule/validate! scheduled)
        problem (attention/validate! (:source-operation plan))
        _ (tiled-lowering-row! plan scheduled problem)
        subgroup-size (:workgroup-size scheduled)
        tile-count (get-in scheduled [:membership-tiling :tile-count])
        slots (component-slots scheduled)
        partial-ids (partial-buffer-ids plan)
        query-ops (query-metadata-operations problem)
        route-ops (if (attention/dense-paged-route? (:route problem))
                    (dense-route-operations problem)
                    (csr-route-operations problem))
        membership-ops (if (attention/interval-visibility? (:visibility problem))
                         (interval-membership-operations problem)
                         (csr-membership-operations problem))
        initial-ops
        (flatten-operations
         (concat
          query-ops
          [(load-value 'query-position :int (get-in problem [:query :positions])
                       ['query-token])
           (compute 'query-position-valid :predicate
                    (expr :ge :predicate 'query-position (lit 0 :int)))
           (compute 'query-batch-valid :predicate
                    (expr :ge :predicate 'query-batch (lit 0 :int)))
           (compute 'safe-query-batch :int
                    (expr :min :int
                          (expr :max :int 'query-batch (lit 0 :int))
                          (lit (dec (:batch-size problem)) :int)))
           (compute 'query-position-long :long (cast-expr 'query-position :long))]
          route-ops membership-ops (tile-bound-operations problem scheduled)
          [(compute 'initial-valid :predicate
                    (all-expr ['query-metadata-valid 'query-position-valid
                               'query-batch-valid 'route-valid 'membership-valid]))
           (compute 'kv-head :int
                    (expr :quot :int 'query-head
                          (lit (quot (:q-heads problem) (:kv-heads problem)) :int)))
           (membership-loop problem scheduled slots
                            'tile-membership-begin 'tile-membership-end)]
          (partial-store-operations partial-ids slots)))
        launch (launch/spec {:workgroup-size [subgroup-size 1 1]
                             :group-count [(:q-heads problem)
                                           (get-in problem [:query :total-tokens])
                                           tile-count]})]
    (body/make
     {:id [:segmented-weighted-reduction-partial-body (:id plan) scheduled]
      :parameters (partial-parameters plan problem scheduled)
      :stable-reads (mapv body/stable-read (swr/ordered-input-ids plan))
      :indices (vec (concat [(body/->IndexBinding 'query-head :group 0)
                             (body/->IndexBinding 'query-token :group 1)
                             (body/->IndexBinding 'history-tile :group 2)
                             (body/->IndexBinding 'lane :lane 0)]
                            (slot-indices problem scheduled slots)))
      :masks (into [(lane-zero-mask)] (slot-masks problem slots))
      :operations initial-ops
      :schedule (assoc scheduled :subgroup-size subgroup-size :phase :partial)
      :launch launch
      :provenance {:operation-id (:id problem)
                   :semantic-op :segmented-weighted-reduction
                   :algebra-plan-id (:id plan)
                   :lowering :scheduled-partial-kernel-body}
      :attributes {:storage-kind :routed-paged-kv
                   :route-kind (attention/route-kind (:route problem))
                   :visibility-kind (attention/visibility-kind (:visibility problem))
                   :history-tiles tile-count}})))

(defn- merge-update-region
  [slots]
  (let [next-accs (mapv :next-id slots)]
    [(compute 'merge-maximum-is-nan :predicate
              (body/scalar-expression :isnan :predicate ['maximum-state]))
     (compute 'tile-maximum-is-nan :predicate
              (body/scalar-expression :isnan :predicate ['tile-maximum]))
     (compute 'merge-state-is-nan :predicate
              (or-expr 'merge-maximum-is-nan 'tile-maximum-is-nan))
     (body/->IfRegion
      'merge-state-is-nan
      [(body/->Yield [(lit Double/NaN :float)
                      (lit Double/NaN :float)
                      (lit Double/NaN :float)])]
      [(compute 'merged-maximum-valid :float
                (expr :max :float 'maximum-state 'tile-maximum))
       (compute 'old-state-weight-valid :float
                (expr :exp :float
                      (expr :- :float 'maximum-state 'merged-maximum-valid)))
       (compute 'tile-state-weight-valid :float
                (expr :exp :float
                      (expr :- :float 'tile-maximum 'merged-maximum-valid)))
       (body/->Yield ['merged-maximum-valid
                      'old-state-weight-valid 'tile-state-weight-valid])]
      [(body/value 'next-maximum :float)
       (body/value 'old-weight :float)
       (body/value 'new-weight :float)])
     (compute 'next-denominator :float
              (expr :+ :float
                    (expr :* :float 'denominator-state 'old-weight)
                    (expr :* :float 'tile-denominator 'new-weight)))
     (vec
      (mapcat
       (fn [{:keys [binding-id next-id tile-value-id]}]
         [(compute next-id :float
                   (expr :+ :float
                         (expr :* :float binding-id 'old-weight)
                         (expr :* :float tile-value-id 'new-weight)))])
       slots))
     (body/->Yield
      (vec (concat ['next-valid 'next-maximum 'next-denominator] next-accs)))]))

(defn- merge-loop
  [scheduled partial-ids slots]
  (let [acc-bindings (mapv :binding-id slots)
        acc-results (mapv :result-id slots)
        state-bindings (vec (concat [(body/value 'merge-valid-state :predicate)
                                     (body/value 'maximum-state :float)
                                     (body/value 'denominator-state :float)]
                                    (map #(body/value % :float) acc-bindings)))
        state-initials (vec (concat [(true-expr)
                                     (lit -3.402823466e38 :float)
                                     (lit 0.0 :float)]
                                    (repeat (count slots) (lit 0.0 :float))))
        state-results (vec (concat [(body/value 'final-valid :predicate)
                                    (body/value 'final-maximum :float)
                                    (body/value 'final-denominator :float)]
                                   (map #(body/value % :float) acc-results)))
        tile-loads
        (flatten-operations
         (concat
          [(load-value 'tile-valid-int :int (:valid partial-ids)
                       ['query-token 'query-head 'merge-tile])
           (load-value 'tile-maximum :float (:maximum partial-ids)
                       ['query-token 'query-head 'merge-tile])
           (load-value 'tile-denominator :float (:denominator partial-ids)
                       ['query-token 'query-head 'merge-tile])
           (compute 'tile-valid :predicate
                    (expr :eq :predicate 'tile-valid-int (lit 1 :int)))
           (compute 'next-valid :predicate
                    (and-expr 'merge-valid-state 'tile-valid))]
          (mapcat
           (fn [{:keys [id mask-id tile-value-id]}]
             [(load-value tile-value-id :float (:weighted-values partial-ids)
                          ['query-token 'query-head 'merge-tile id]
                          mask-id (lit 0.0 :float))])
           slots)
          (merge-update-region slots)))]
    (body/->ForLoop
     (body/value 'merge-tile :int) 0
     (get-in scheduled [:membership-tiling :tile-count]) 1
     (mapv body/->LoopArg state-bindings state-initials)
     tile-loads state-results
     {:uniform-iter-args #{'merge-valid-state 'maximum-state 'denominator-state}})))

(defn lower-routed-paged-merge
  "Merge private tile states in increasing tile order and materialize the semantic output."
  [plan scheduled]
  (let [plan (swr/validate! plan)
        scheduled (schedule/validate! scheduled)
        problem (attention/validate! (:source-operation plan))
        _ (tiled-lowering-row! plan scheduled problem)
        subgroup-size (:workgroup-size scheduled)
        partial-ids (partial-buffer-ids plan)
        slots (mapv #(assoc % :tile-value-id
                            (symbol (str "tile-weighted-value-" (:slot %))))
                    (component-slots scheduled))
        launch (launch/spec {:workgroup-size [subgroup-size 1]
                             :group-count [(:q-heads problem)
                                           (get-in problem [:query :total-tokens])]})]
    (body/make
     {:id [:segmented-weighted-reduction-merge-body (:id plan) scheduled]
      :parameters (merge-parameters plan problem scheduled)
      :stable-reads (mapv (comp body/stable-read :id)
                          (partial-buffer-specs plan scheduled))
      :indices (vec (concat [(body/->IndexBinding 'query-head :group 0)
                             (body/->IndexBinding 'query-token :group 1)
                             (body/->IndexBinding 'lane :lane 0)]
                            (slot-indices problem scheduled slots)))
      :masks (slot-masks problem slots)
      :operations (flatten-operations
                   [(merge-loop scheduled partial-ids slots)
                    (output-operations problem slots)])
      :schedule (assoc scheduled :subgroup-size subgroup-size :phase :merge)
      :launch launch
      :provenance {:operation-id (:id problem)
                   :semantic-op :segmented-weighted-reduction
                   :algebra-plan-id (:id plan)
                   :lowering :scheduled-merge-kernel-body}
      :attributes {:storage-kind :private-online-state
                   :merge-kind (get-in scheduled [:state :merge :kind])
                   :merge-order (get-in scheduled [:state :merge :order])
                   :history-tiles (get-in scheduled [:membership-tiling :tile-count])}})))
