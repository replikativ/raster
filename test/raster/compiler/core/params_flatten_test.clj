(ns raster.compiler.core.params-flatten-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.params-flatten :as pf]))

(deftest hmap-mandatory-shapes
  (testing "(HMap :mandatory {...}) form"
    (is (= '{:a Long :b Double}
           (pf/hmap-mandatory '(HMap :mandatory {:a Long :b Double})))))
  (testing "(HMap {...}) shorthand"
    (is (= '{:a Long}
           (pf/hmap-mandatory '(HMap {:a Long})))))
  (testing "(HMap :mandatory {...} :complete? true) keeps mandatory only"
    (is (= '{:a Long}
           (pf/hmap-mandatory '(HMap :mandatory {:a Long} :complete? true))))))

(deftest validate-pytree-spec-accepts-supported-shapes
  (testing "Shorthand and explicit forms with closed semantics are accepted"
    (is (= '(HMap {:a Long}) (pf/validate-pytree-spec! '(HMap {:a Long}))))
    (is (= '(HMap :mandatory {:a Long})
           (pf/validate-pytree-spec! '(HMap :mandatory {:a Long}))))
    (is (= '(HMap :mandatory {:a Long} :complete? true)
           (pf/validate-pytree-spec! '(HMap :mandatory {:a Long} :complete? true))))
    (is (= '(HVec [Long Double])
           (pf/validate-pytree-spec! '(HVec [Long Double])))))
  (testing "Recurses into nested HMap/HVec children"
    (is (= '(HMap :mandatory {:layers (HVec [(HMap :mandatory {:W Long})])})
           (pf/validate-pytree-spec!
             '(HMap :mandatory {:layers (HVec [(HMap :mandatory {:W Long})])}))))))

(deftest validate-pytree-spec-rejects-optional-keys
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":optional"
        (pf/validate-pytree-spec!
          '(HMap :mandatory {:a Long} :optional {:b Long})))))

(deftest validate-pytree-spec-rejects-complete-false
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":complete\? true"
        (pf/validate-pytree-spec!
          '(HMap :mandatory {:a Long} :complete? false)))))

(deftest validate-pytree-spec-rejects-absent-keys
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":absent-keys"
        (pf/validate-pytree-spec!
          '(HMap :mandatory {:a Long} :absent-keys #{:b})))))

(deftest validate-pytree-spec-rejects-unknown-options
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unsupported options"
        (pf/validate-pytree-spec!
          '(HMap :mandatory {:a Long} :weird-opt 42)))))

(deftest validate-pytree-spec-rejects-malformed-hvec
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"HVec"
        (pf/validate-pytree-spec! '(HVec))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"HVec"
        (pf/validate-pytree-spec! '(HVec [Long] :something-else)))))

(deftest validate-pytree-spec-rejects-nested-bad-shape
  (testing "Nested HMap with :optional fails — recursion catches it"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #":optional"
          (pf/validate-pytree-spec!
            '(HMap :mandatory {:layers (HVec [(HMap :optional {:W Long})])}))))))

(deftest prepare-deftm-rejects-bad-spec
  (testing "prepare-deftm wraps validation error with the offending arg name"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"arg `w`.*:optional"
          (pf/prepare-deftm
            '[w]
            '[(Params (HMap :mandatory {:a Long} :optional {:b Long}))]
            '(use w))))))

;; ----------------------------------------------------------------------
;; Identity-collision detection (weight tying — Phase 1: error on collision)
;; ----------------------------------------------------------------------

(deftest assert-no-identity-collisions-passes-distinct-leaves
  (testing "Distinct array instances at every leaf — no collision"
    (let [spec '(HMap :mandatory {:W (Param (Array double))
                                  :b (Param (Array double))})
          value {:W (double-array [1 2]) :b (double-array [3])}]
      (is (nil? (pf/assert-no-identity-collisions! spec value))))))

(deftest assert-no-identity-collisions-detects-shared-array
  (testing "Same array instance at two HMap positions raises with both paths"
    (let [shared (double-array [1 2 3])
          spec '(HMap :mandatory {:W1 (Param (Array double))
                                  :W2 (Param (Array double))})
          value {:W1 shared :W2 shared}]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
            #":W1.*:W2"
            (pf/assert-no-identity-collisions! spec value))))))

(deftest assert-no-identity-collisions-detects-cross-subtree-share
  (testing "Same array instance in different sub-trees is detected"
    (let [shared (double-array [1 2])
          spec '(HMap :mandatory {:enc (HMap :mandatory {:W (Param (Array double))})
                                  :dec (HMap :mandatory {:W (Param (Array double))})})
          value {:enc {:W shared} :dec {:W shared}}]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
            #"share JVM identity"
            (pf/assert-no-identity-collisions! spec value))))))

(deftest assert-no-identity-collisions-detects-share-in-hvec
  (testing "Same array at two HVec positions is detected"
    (let [shared (double-array [1])
          spec '(HMap :mandatory {:layers (HVec [(HMap :mandatory {:W (Param (Array double))})
                                                  (HMap :mandatory {:W (Param (Array double))})])})
          value {:layers [{:W shared} {:W shared}]}]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
            #":layers 0 :W.*:layers 1 :W"
            (pf/assert-no-identity-collisions! spec value))))))

(deftest assert-no-identity-collisions-allows-equal-but-distinct-arrays
  (testing "Two arrays with equal contents but distinct identity are fine"
    (let [spec '(HMap :mandatory {:W1 (Param (Array double))
                                  :W2 (Param (Array double))})
          value {:W1 (double-array [1 2 3])
                 :W2 (double-array [1 2 3])}]   ;; distinct array, same contents
      (is (nil? (pf/assert-no-identity-collisions! spec value))))))

(deftest assert-no-identity-collisions-skips-primitives
  (testing "Boxed numbers/booleans aren't treated as identity-shared"
    ;; Two pytree positions with the same long literal could be cached and
    ;; share identity (Long.valueOf), but they don't represent shared mutable
    ;; state, so identity collision must be ignored for primitives.
    (let [spec '(HMap :mandatory {:a Long :b Long})
          n (Long/valueOf 42)
          value {:a n :b n}]
      (is (nil? (pf/assert-no-identity-collisions! spec value))))))

(deftest flatten-spec-canonical-order
  (testing "Flat HMap leaves emit in sorted-key order"
    (let [leaves (pf/flatten-spec '(HMap :mandatory {:b (Param Long)
                                                     :a (Param Double)}))]
      (is (= [[:a] [:b]] (mapv :path leaves)))))
  (testing "HVec leaves emit in index order, sub-HMaps recurse"
    (let [leaves (pf/flatten-spec
                   '(HMap :mandatory {:layers (HVec [(HMap :mandatory {:W (Param Long)})
                                                     (HMap :mandatory {:W (Param Long)})])}))]
      (is (= [[:layers 0 :W] [:layers 1 :W]] (mapv :path leaves)))))
  (testing "Wrapper kind is preserved on leaves"
    (let [leaves (pf/flatten-spec '(HMap :mandatory {:p (Param Double)
                                                     :f (Frozen Double)
                                                     :c Double}))]
      (is (= {:p :param, :f :frozen, :c :plain}
             (into {} (map (fn [{:keys [path kind]}] [(first path) kind])) leaves))))))

(deftest path->sym-stable
  (is (= 'w (pf/path->sym 'w [])))
  (is (= 'w__a (pf/path->sym 'w [:a])))
  (is (= 'w__layers__0__Wq (pf/path->sym 'w [:layers 0 :Wq]))))

(deftest extract-access-step-shapes
  (is (= '[w :W] (pf/extract-access-step '(:W w))))
  (is (= '[w :W] (pf/extract-access-step '(get w :W))))
  (is (= '[v 3]  (pf/extract-access-step '(nth v 3))))
  (is (nil? (pf/extract-access-step '(some-fn w :W))))
  (is (nil? (pf/extract-access-step 'plain-sym)))
  (is (nil? (pf/extract-access-step '(get w "string-key")))))

(deftest rewrite-keyword-access
  (let [env  '{w {:root w :path []
                  :spec (HMap :mandatory {:W1 (Param (Array double))
                                          :b1 (Param (Array double))})}}
        {:keys [body]} (pf/rewrite-body '(linear x (:W1 w) (:b1 w)) env)]
    (is (= '(linear x w__W1 w__b1) body))))

(deftest rewrite-get-form
  (let [env  '{w {:root w :path []
                  :spec (HMap :mandatory {:W (Param (Array double))})}}
        {:keys [body]} (pf/rewrite-body '(linear x (get w :W)) env)]
    (is (= '(linear x w__W) body))))

(deftest rewrite-nth-form
  (let [env  '{layers {:root w :path [:layers]
                       :spec (HVec [(HMap :mandatory {:Wq (Param (Array double))})])}}
        {:keys [body]} (pf/rewrite-body '(:Wq (nth layers 0)) env)]
    (is (= 'w__layers__0__Wq body))))

(deftest rewrite-let-aliased-subtree
  (let [env '{w {:root w :path []
                 :spec (HMap :mandatory
                             {:layers (HVec [(HMap :mandatory
                                                   {:Wq (Param (Array double))
                                                    :Wk (Param (Array double))})])})}}
        {:keys [body leaves]}
        (pf/rewrite-body
          '(let [layer (nth (:layers w) 0)
                 h     (attn x (:Wq layer) (:Wk layer))]
             h)
          env)]
    (let [flat-syms (->> body flatten (filter symbol?) set)]
      (is (contains? flat-syms 'w__layers__0__Wq) "Wq access resolved via alias")
      (is (contains? flat-syms 'w__layers__0__Wk) "Wk access resolved via alias"))
    (is (= #{'w__layers__0__Wq 'w__layers__0__Wk}
           (into #{} (map :sym) leaves)))))

(deftest prepare-deftm-flat-mlp
  (let [result (pf/prepare-deftm
                 '[w x]
                 '[(Params (HMap :mandatory {:W1 (Param (Array double))
                                             :b1 (Param (Array double))
                                             :W2 (Param (Array double))
                                             :b2 (Param (Array double))}))
                   (Array double)]
                 '(let [h (linear x (:W1 w) (:b1 w))]
                    (linear h (:W2 w) (:b2 w))))]
    (testing "Pytree arg expanded to flat positional args in canonical order"
      (is (= '[w__W1 w__W2 w__b1 w__b2 x] (:params result))))
    (testing "Flat annotations preserve Param wrapper, x untouched"
      (is (= '[(Param (Array double)) (Param (Array double))
               (Param (Array double)) (Param (Array double))
               (Array double)]
             (:annotations result))))
    (testing "Body references flat args directly"
      (is (= '(let [h (linear x w__W1 w__b1)] (linear h w__W2 w__b2))
             (:body result))))
    (testing "Treedef carries spec + leaves"
      (let [td (-> result :treedefs (get 'w))]
        (is (= '(HMap :mandatory {:W1 (Param (Array double))
                                  :b1 (Param (Array double))
                                  :W2 (Param (Array double))
                                  :b2 (Param (Array double))})
               (:spec td)))
        (is (= '[w__W1 w__W2 w__b1 w__b2] (mapv :sym (:leaves td))))))))

(deftest prepare-deftm-non-pytree-args-untouched
  (let [result (pf/prepare-deftm
                 '[a b]
                 '[(Array double) Long]
                 '(some-fn a b))]
    (is (= '[a b] (:params result)))
    (is (= '[(Array double) Long] (:annotations result)))
    (is (= '(some-fn a b) (:body result)))
    (is (empty? (:treedefs result)))))

(deftest runtime-flatten-roundtrip
  (let [spec  '(HMap :mandatory {:W1 (Param (Array double))
                                 :b1 (Param (Array double))
                                 :W2 (Param (Array double))
                                 :b2 (Param (Array double))})
        value {:W1 (double-array [1 2]) :b1 (double-array [3])
               :W2 (double-array [4 5 6]) :b2 (double-array [7])}
        flat  (pf/flatten-value spec value)]
    (testing "flatten emits leaves in canonical (sorted-key) order"
      (is (= 4 (count flat)))
      (is (= [1.0 2.0] (vec (nth flat 0))) ":W1 first")
      (is (= [4.0 5.0 6.0] (vec (nth flat 1))) ":W2 second"))
    (testing "unflatten round-trips"
      (let [restored (pf/unflatten-value spec flat)]
        (is (= (set (keys value)) (set (keys restored))))
        (doseq [k (keys value)]
          (is (= (vec (get value k)) (vec (get restored k)))))))))

(deftest runtime-flatten-nested
  (let [spec  '(HMap :mandatory
                     {:embed (Param (Array double))
                      :layers (HVec [(HMap :mandatory {:W (Param (Array double))})
                                     (HMap :mandatory {:W (Param (Array double))})])})
        value {:embed  (double-array [0.0])
               :layers [{:W (double-array [1.0])}
                        {:W (double-array [2.0])}]}
        flat  (pf/flatten-value spec value)]
    (is (= 3 (count flat)))
    (is (= [[0.0] [1.0] [2.0]] (mapv vec flat)))
    (let [restored (pf/unflatten-value spec flat)]
      (is (= [0.0] (vec (:embed restored))))
      (is (= [1.0] (vec (-> restored :layers (nth 0) :W))))
      (is (= [2.0] (vec (-> restored :layers (nth 1) :W)))))))

(deftest filter-by-kind-partitions
  (let [spec   '(HMap :mandatory {:W   (Param (Array double))
                                  :ln  (Frozen (Array double))})
        leaves (pf/flatten-spec spec)
        value  {:W (double-array [1]) :ln (double-array [2])}
        flat   (pf/flatten-value spec value)
        params (pf/filter-by-kind :param  leaves flat)
        frozen (pf/filter-by-kind :frozen leaves flat)]
    (is (= 1 (count params)))
    (is (= 1 (count frozen)))
    (is (= [1.0] (vec (first params))))
    (is (= [2.0] (vec (first frozen))))))

(deftest let-binding-with-subtree-value-dropped
  (testing "Sub-tree alias bindings are dropped from the let — no dangling pytree-arg refs"
    (let [{:keys [body]}
          (pf/rewrite-body
            '(let [layer (nth (:layers w) 0)
                   h     (use (:Wq layer))]
               h)
            '{w {:root w :path []
                 :spec (HMap :mandatory
                             {:layers (HVec [(HMap :mandatory {:Wq (Param (Array double))})])})}})]
      ;; The 'layer' binding should be gone; only 'h' remains.
      (let [bindings (second body)]
        (is (= 1 (/ (count bindings) 2)))
        (is (= 'h (nth bindings 0))
            "only the surviving non-alias binding is in the let")
        (is (= '(use w__layers__0__Wq) (nth bindings 1))
            "the leaf access through the (now-absent) alias still resolves via env")))))

(deftest pytree-arg-as-value-errors
  (testing "Passing the pytree arg as a value (e.g. (some-fn w)) is rejected"
    (is (thrown-with-msg?
          clojure.lang.ExceptionInfo #"referenced as a value"
          (pf/prepare-deftm
            '[w]
            '[(Params (HMap :mandatory {:W (Param (Array double))}))]
            '(some-fn w))))))

(deftest flat-view-detection
  (testing "flat-view-form? recognizes both qualified and unqualified"
    (is (pf/flat-view-form? '(flat-view vg :spec-of #'foo)))
    (is (pf/flat-view-form? '(raster.tree/flat-view vg)))
    (is (not (pf/flat-view-form? '(some-other-fn vg))))
    (is (not (pf/flat-view-form? 'plain-symbol)))))

(deftest path->flat-idx-canonical-order
  (testing "Computes canonical flat-index from a leaf path"
    (let [spec '(HMap :mandatory {:b (Param Long) :a (Param Long)})]
      ;; sorted: :a (0), :b (1)
      (is (= 0 (pf/path->flat-idx spec [:a])))
      (is (= 1 (pf/path->flat-idx spec [:b])))
      (is (nil? (pf/path->flat-idx spec [:nonexistent])))))
  (testing "Nested HMap+HVec paths"
    (let [spec '(HMap :mandatory {:layers (HVec [(HMap :mandatory {:Wq (Param Long)})
                                                  (HMap :mandatory {:Wq (Param Long)})])})]
      (is (= 0 (pf/path->flat-idx spec [:layers 0 :Wq])))
      (is (= 1 (pf/path->flat-idx spec [:layers 1 :Wq]))))))

(deftest rewrite-flat-view-leaf-access
  (testing "Path access on a flat-view binding emits (nth flat-source flat-idx)"
    (let [spec '(HMap :mandatory {:W (Param (Array double)) :b (Param (Array double))})
          {:keys [body]} (pf/rewrite-body
                           `(~'let [~'grads (~'flat-view ~'vg :spec ~spec :starting-at 1)
                                    ~'x (~'use (:W ~'grads))
                                    ~'y (~'use (:b ~'grads))]
                              [~'x ~'y])
                           {})]
      ;; :W is sorted-first → flat-idx 0 → starting-at 1 = nth idx 1
      ;; :b is sorted-second → flat-idx 1 → starting-at 1 = nth idx 2
      (let [bindings (second body)]
        ;; bindings: [grads vg, x (use (nth vg 1)), y (use (nth vg 2))]
        (is (= '(use (clojure.core/nth vg 1)) (nth bindings 3)))
        (is (= '(use (clojure.core/nth vg 2)) (nth bindings 5)))))))

(deftest rewrite-flat-view-nested-path
  (testing "Nested path access through a flat-view resolves correctly"
    (let [spec '(HMap :mandatory {:layers (HVec [(HMap :mandatory {:Wq (Param (Array double))
                                                                    :Wk (Param (Array double))})])
                                  :embed (Param (Array double))})
          {:keys [body]} (pf/rewrite-body
                           `(~'let [~'grads (~'flat-view ~'vg :spec ~spec)
                                    ~'layer0 (~'nth (:layers ~'grads) 0)
                                    ~'Wq (:Wq ~'layer0)
                                    ~'Wk (:Wk ~'layer0)
                                    ~'emb (:embed ~'grads)]
                              [~'Wq ~'Wk ~'emb])
                           {})]
      ;; sorted leaves: [:embed], [:layers 0 :Wk], [:layers 0 :Wq]
      ;; flat indices  :  0,         1,             2
      (let [nth-calls (atom [])]
        (clojure.walk/postwalk
         (fn [x]
           (when (and (seq? x) (= 'clojure.core/nth (first x)) (= 'vg (second x)))
             (swap! nth-calls conj (last x)))
           x)
         body)
        (is (= #{0 1 2} (set @nth-calls)))))))

(deftest walk-form-detection
  (testing "walk-form? recognizes both qualified and unqualified"
    (is (pf/walk-form? '(walk! :param f [w g] lr)))
    (is (pf/walk-form? '(raster.tree/walk! :any f [w] x y)))
    (is (not (pf/walk-form? '(other-fn :param))))))

(deftest walk-expansion-emits-per-leaf-calls
  (testing "Filters by kind, emits a splice-statements marker with per-leaf calls"
    (let [spec '(HMap :mandatory {:W (Param (Array double))
                                   :b (Param (Array double))
                                   :ln (Frozen (Array double))})
          env {'w {:root 'w :path [] :spec spec}
               'g {:root 'g :path [] :spec spec}}
          {:keys [body]} (pf/rewrite-body
                           '(walk! :param adam-step! [w g] lr eps)
                           env)]
      ;; The expansion is a splice marker — its rest contains the per-leaf
      ;; calls. When this appears at let-body position, the let-handler
      ;; splices its contents into the parent body.
      (is (pf/splice-statements? body))
      (is (= '[(adam-step! w__W g__W lr eps)
               (adam-step! w__b g__b lr eps)]
             (vec (rest body)))))))

(deftest walk-expansion-spliced-into-let-body
  (testing "walk! used as a let-body form is spliced as statement-position effects"
    (let [spec '(HMap :mandatory {:W (Param (Array double))
                                   :b (Param (Array double))})
          env {'w {:root 'w :path [] :spec spec}
               'm {:root 'm :path [] :spec spec}}
          {:keys [body]} (pf/rewrite-body
                           (list 'let
                                 ['grads (list 'flat-view 'vg :spec spec :starting-at 1)]
                                 (list 'walk! :param 'step! ['w 'grads 'm])
                                 'loss)
                           env)]
      ;; body is (let [grads vg] (step! w__W (nth vg 1) m__W) (step! w__b (nth vg 2) m__b) loss)
      ;; The walk! statement-position form has been spliced to two calls.
      (let [body-forms (drop 2 body)]   ;; skip 'let and bindings vec
        (is (= 3 (count body-forms)))
        (is (= 'loss (last body-forms)))
        (let [calls (take 2 body-forms)
              has-nth-vg? (fn [call]
                            (some (fn [arg]
                                    (and (seq? arg)
                                         (= 'clojure.core/nth (first arg))
                                         (= 'vg (second arg))))
                                  call))]
          (is (every? has-nth-vg? calls)))))))

;; ----------------------------------------------------------------------
;; Helpers for tests that need a fake defmodel-typed callee var.
;; Intern a real var with defmodel-style metadata so resolve + splice see it.
;; ----------------------------------------------------------------------

(defn- fake-callee-var!
  "Intern a fake callee var into the test ns with defmodel-style metadata.
  Returns the var. Cleanup happens via with-fake-callees."
  [sym original-args treedefs]
  (let [flat-sym (symbol (str sym "--flat"))
        flat-var (intern *ns* flat-sym (fn [& _] :flat-result))
        v        (intern *ns* sym (fn [& _] :wrapper-result))]
    (alter-meta! v assoc
                 :raster.params/flat-var flat-var
                 :raster.params/original-args original-args
                 :raster.params/treedefs treedefs)
    v))

(defn- with-fake-callees [syms thunk]
  (try (thunk)
       (finally
         (doseq [s syms]
           (ns-unmap *ns* s)
           (ns-unmap *ns* (symbol (str s "--flat")))))))

(deftest walk-arity-mismatch-errors-at-macro-time
  (testing "If f's known arity doesn't match (pytree-args + extras), error clearly"
    (with-fake-callees ['ariy-step!]
      (fn []
        ;; Define a fake callee with original-args of length 4 (acc-style 4-arg)
        (fake-callee-var!
          'ariy-step!
          '[w g lr eps]
          {})    ;; no treedefs — just a deftm-shaped arglist
        (let [spec '(HMap :mandatory {:W (Param (Array double))
                                      :b (Param (Array double))})
              env {'w {:root 'w :path [] :spec spec}
                   'g {:root 'g :path [] :spec spec}}]
          ;; Caller emits (ariy-step! w-leaf g-leaf lr) → arity 3, not 4
          (is (thrown-with-msg? clojure.lang.ExceptionInfo
                #"arity 4.*arity 3"
                (pf/rewrite-body
                  '(walk! :param ariy-step! [w g] lr)
                  env))))))))

(deftest walk-arity-match-passes
  (testing "Matching arity passes through cleanly"
    (with-fake-callees ['ariy-ok-step!]
      (fn []
        (fake-callee-var!
          'ariy-ok-step!
          '[w g lr eps]
          {})
        (let [spec '(HMap :mandatory {:W (Param (Array double))})
              env {'w {:root 'w :path [] :spec spec}
                   'g {:root 'g :path [] :spec spec}}
              {:keys [body]}
              (pf/rewrite-body
                '(walk! :param ariy-ok-step! [w g] lr eps)
                env)]
          ;; No throw — body is a splice-statements form
          (is (pf/splice-statements? body)))))))

(deftest walk-arity-skipped-when-no-info
  (testing "Unresolved f-form skips the check (no false positive)"
    (let [spec '(HMap :mandatory {:W (Param (Array double))})
          env {'w {:root 'w :path [] :spec spec}}
          ;; 'no-such-fn doesn't resolve → arities is nil → check is skipped
          {:keys [body]}
          (pf/rewrite-body
            '(walk! :param no-such-fn [w])
            env)]
      (is (pf/splice-statements? body)))))

(deftest scan-vec-form-detection
  (is (pf/scan-vec-form? '(scan-vec f init xs)))
  (is (pf/scan-vec-form? '(raster.tree/scan-vec f init xs extras)))
  (is (not (pf/scan-vec-form? '(reduce f init xs)))))

(deftest scan-vec-expands-to-let-chain
  (testing "Unrolls into N applications of f over a known-length HVec"
    (let [layer-spec '(HMap :mandatory {:W (Param (Array double))})
          w-spec    (list 'HMap :mandatory
                          {:layers (list 'HVec [layer-spec layer-spec])})
          env {'w {:root 'w :path [] :spec w-spec}}
          {:keys [body]} (pf/rewrite-body
                           (list 'scan-vec 'block 'h-init (list :layers 'w))
                           env)]
      (is (= 'let* (first body)))
      ;; Bindings: scan_acc__0, h-init, scan_acc__1, (block ...), scan_acc__2, (block ...)
      ;; = 6 entries for 2 layers
      (is (= 6 (count (second body))))
      ;; Last acc returned
      (is (= 'scan_acc__2 (last body)))
      ;; Each block call has the layer reconstructed as map literal
      (let [bindings (second body)
            call-1 (nth bindings 3)
            call-2 (nth bindings 5)]
        (is (and (seq? call-1) (= 'block (first call-1))))
        (is (and (seq? call-2) (= 'block (first call-2))))
        ;; The 3rd arg (after f and acc) is the layer map literal
        (is (map? (nth call-1 2)))
        (is (= #{:W} (set (keys (nth call-1 2)))))))))

(deftest scan-vec-leaf-elements-pass-through
  (testing "When element is a leaf (e.g. (HVec [(Array double) ...])), passes through"
    (let [w-spec '(HMap :mandatory {:rows (HVec [(Array double) (Array double)])})
          env {'w {:root 'w :path [] :spec w-spec}}
          {:keys [body]} (pf/rewrite-body
                           (list 'scan-vec 'sum 'init (list :rows 'w))
                           env)
          bindings (second body)
          call-1 (nth bindings 3)]
      ;; For leaf elements, the element passes through as a direct path access
      ;; which the walker rewrites — in this case to a flat-arg symbol.
      (is (= 'w__rows__0 (nth call-1 2))))))

(deftest scan-vec-splices-when-f-is-a-defmodel
  (testing "scan-vec emits direct calls to f--flat with leaves spliced — no per-iteration HMap"
    (with-fake-callees ['scan-block]
      (fn []
        (fake-callee-var!
          'scan-block
          '[acc layer batch]
          '{layer {:spec (HMap :mandatory {:W (Param (Array double))
                                           :b (Param (Array double))})
                   :leaves [{:path [:W] :sym layer__W :kind :param :type (Array double)}
                            {:path [:b] :sym layer__b :kind :param :type (Array double)}]}})
        (let [layer-spec '(HMap :mandatory {:W (Param (Array double))
                                            :b (Param (Array double))})
              w-spec    (list 'HMap :mandatory
                              {:layers (list 'HVec [layer-spec layer-spec])})
              env {'w {:root 'w :path [] :spec w-spec}}
              {:keys [body]} (pf/rewrite-body
                               (list 'scan-vec 'scan-block 'h-init (list :layers 'w) 'batch)
                               env)
              ns-name (str (.name *ns*))
              flat-sym (symbol ns-name "scan-block--flat")
              bindings (second body)
              call-1 (nth bindings 3)
              call-2 (nth bindings 5)]
          (testing "Each iteration calls f--flat directly, with no map literal"
            (is (= flat-sym (first call-1)))
            (is (= flat-sym (first call-2)))
            ;; Body args: acc, layer__W resolved leaf, layer__b resolved leaf, batch
            (is (every? (complement map?) call-1)
                "No HMap literal anywhere in the spliced call")
            (is (every? (complement map?) call-2)))
          (testing "Per-leaf args resolve to outer flat-arg syms via the walker"
            ;; (:W (nth (:layers w) 0)) → w__layers__0__W
            (is (= 'w__layers__0__W (nth call-1 2)))
            (is (= 'w__layers__0__b (nth call-1 3)))
            (is (= 'w__layers__1__W (nth call-2 2)))
            (is (= 'w__layers__1__b (nth call-2 3)))
            (is (= 'batch (nth call-1 4))
                "Extras are appended after spliced leaves")))))))

(deftest scan-vec-falls-back-when-f-not-defmodel
  (testing "If f doesn't resolve to a Params-typed defmodel, falls back to map reconstruction"
    ;; 'plain-fn is just a symbol — resolve returns nil — splice path skipped.
    (let [w-spec '(HMap :mandatory
                        {:layers (HVec [(HMap :mandatory {:W (Param (Array double))})])})
          env {'w {:root 'w :path [] :spec w-spec}}
          {:keys [body]} (pf/rewrite-body
                           (list 'scan-vec 'plain-fn 'h-init (list :layers 'w))
                           env)
          bindings (second body)
          call-1 (nth bindings 3)]
      ;; Fallback path emits a map literal as the elt arg
      (is (= 'plain-fn (first call-1)))
      (is (map? (nth call-1 2)) "Fallback emits map literal"))))

(deftest prepare-deftm-frozen-tracked
  (let [result (pf/prepare-deftm
                 '[w]
                 '[(Params (HMap :mandatory {:trainable (Param (Array double))
                                             :fixed     (Frozen (Array double))}))]
                 '(plus (:trainable w) (:fixed w)))]
    (is (= '[w__fixed w__trainable] (:params result)))
    (is (= '[(Frozen (Array double)) (Param (Array double))] (:annotations result))
        "Wrapper kinds are preserved per-leaf in annotations")
    (let [leaves (-> result :treedefs (get 'w) :leaves)]
      (is (= {:fixed :frozen, :trainable :param}
             (into {} (map (fn [l] [(first (:path l)) (:kind l)])) leaves))))))

;; ----------------------------------------------------------------------
;; Cross-deftm pytree call splicing
;;
;; When a defmodel-typed callee is invoked from another defmodel body, the
;; pre-flatten walker should rewrite the call to invoke the callee's --flat
;; var directly, splicing each pytree arg into its leaves at the call site.
;; This eliminates runtime Map reconstruction at deftm-to-deftm boundaries.
;; ----------------------------------------------------------------------

(deftest splice-cross-deftm-call-rewrites-pytree-arg
  (testing "Splice rewrites (callee pytree-arg ...) to (callee--flat leaf1 leaf2 ...)"
    (with-fake-callees ['inner-block]
      (fn []
        (fake-callee-var!
          'inner-block
          '[layer x batch d]
          '{layer {:spec (HMap :mandatory {:W (Param (Array double))
                                           :b (Param (Array double))})
                   :leaves [{:path [:W] :sym layer__W :kind :param :type (Array double)}
                            {:path [:b] :sym layer__b :kind :param :type (Array double)}]}})
        (let [spliced (pf/splice-cross-deftm-call
                        '(inner-block (:l w) x batch d)
                        {})
              ns-name (str (.name *ns*))]
          (is (= (list (symbol ns-name "inner-block--flat")
                       '(:W (:l w))
                       '(:b (:l w))
                       'x 'batch 'd)
                 spliced)
              "Pytree arg is spliced into per-leaf path-access expressions"))))))

(deftest splice-cross-deftm-call-passes-through-non-pytree-args
  (testing "Non-pytree args pass through untouched, in order"
    (with-fake-callees ['mixed-callee]
      (fn []
        (fake-callee-var!
          'mixed-callee
          '[scalar w other]
          '{w {:spec (HMap :mandatory {:W (Param (Array double))})
               :leaves [{:path [:W] :sym w__W :kind :param :type (Array double)}]}})
        (let [spliced (pf/splice-cross-deftm-call
                        '(mixed-callee 42 (:layer x) other-sym)
                        {})
              ns-name (str (.name *ns*))]
          (is (= (list (symbol ns-name "mixed-callee--flat")
                       42
                       '(:W (:layer x))
                       'other-sym)
                 spliced)))))))

(deftest splice-cross-deftm-call-noop-for-plain-fn
  (testing "Splice returns nil for calls to functions without defmodel metadata"
    (is (nil? (pf/splice-cross-deftm-call '(some-fn x y) {})))
    (is (nil? (pf/splice-cross-deftm-call '(clojure.core/+ 1 2) {})))
    (is (nil? (pf/splice-cross-deftm-call '(unresolvable-symbol-xyz x) {})))))

(deftest splice-cross-deftm-call-noop-on-arity-mismatch
  (testing "Splice does not fire if arg count doesn't match callee's original arity"
    (with-fake-callees ['arity-callee]
      (fn []
        (fake-callee-var!
          'arity-callee
          '[layer x]
          '{layer {:spec (HMap :mandatory {:W (Param (Array double))})
                   :leaves [{:path [:W] :sym layer__W :kind :param :type (Array double)}]}})
        ;; Wrong arity — returns nil, falls through to generic seq walk
        (is (nil? (pf/splice-cross-deftm-call
                    '(arity-callee (:l w) x extra-arg)
                    {})))))))

(deftest cross-deftm-splice-via-prepare-deftm
  (testing "prepare-deftm rewrites a cross-deftm call to use the callee's flat-var"
    (with-fake-callees ['composite-block]
      (fn []
        (fake-callee-var!
          'composite-block
          '[layer x batch d]
          '{layer {:spec (HMap :mandatory {:W (Param (Array double))
                                           :b (Param (Array double))})
                   :leaves [{:path [:W] :sym layer__W :kind :param :type (Array double)}
                            {:path [:b] :sym layer__b :kind :param :type (Array double)}]}})
        (let [{:keys [body params]}
              (pf/prepare-deftm
                '[w x batch d]
                '[(Params (HMap :mandatory {:l (HMap :mandatory {:W (Param (Array double))
                                                                  :b (Param (Array double))})}))
                  (Array double) Long Long]
                '(composite-block (:l w) x batch d))
              ns-name (str (.name *ns*))]
          (is (= '[w__l__W w__l__b x batch d] params)
              "Outer pytree arg gets flattened into positional args")
          (is (= (list (symbol ns-name "composite-block--flat")
                       'w__l__W 'w__l__b 'x 'batch 'd)
                 body)
              "Body calls the callee's --flat var with leaves spliced and resolved to outer flat-args"))))))

(deftest splice-lifts-unsafe-pytree-arg-to-let-binding
  (testing "Side-effecting / non-trivial pytree arg is lifted to a temp binding so it evaluates once"
    (with-fake-callees ['side-effect-callee]
      (fn []
        (fake-callee-var!
          'side-effect-callee
          '[layer x]
          '{layer {:spec (HMap :mandatory {:W (Param (Array double))
                                           :b (Param (Array double))})
                   :leaves [{:path [:W] :sym layer__W :kind :param :type (Array double)}
                            {:path [:b] :sym layer__b :kind :param :type (Array double)}]}})
        (let [spliced (pf/splice-cross-deftm-call
                        '(side-effect-callee (some-side-effect arg) x)
                        {})
              ns-name (str (.name *ns*))]
          (is (and (seq? spliced) (= 'let* (first spliced)))
              "Result wraps the call in a let* so the unsafe arg is evaluated once")
          (let [bindings (second spliced)
                tmp-sym  (first bindings)
                bind-val (second bindings)
                inner-call (nth spliced 2)]
            (is (= '(some-side-effect arg) bind-val)
                "Original unsafe expression is bound to the temp")
            (is (= (list (symbol ns-name "side-effect-callee--flat")
                         (list :W tmp-sym)
                         (list :b tmp-sym)
                         'x)
                   inner-call)
                "Splice references the temp symbol in each leaf access")))))))

(deftest cross-deftm-splice-with-let-aliased-subtree
  (testing "Splicing works when the pytree arg is a let-aliased sub-tree"
    (with-fake-callees ['alias-block]
      (fn []
        (fake-callee-var!
          'alias-block
          '[layer x]
          '{layer {:spec (HMap :mandatory {:W (Param (Array double))})
                   :leaves [{:path [:W] :sym layer__W :kind :param :type (Array double)}]}})
        (let [{:keys [body]}
              (pf/prepare-deftm
                '[w x]
                '[(Params (HMap :mandatory {:l (HMap :mandatory {:W (Param (Array double))})}))
                  (Array double)]
                '(let [layer (:l w)]
                   (alias-block layer x)))
              ns-name (str (.name *ns*))]
          ;; Sub-tree binding 'layer' is dropped by update-let-env (recorded in
          ;; env), the call site is spliced, leaf access resolves via env.
          (is (= (list 'let []
                       (list (symbol ns-name "alias-block--flat") 'w__l__W 'x))
                 body)))))))
