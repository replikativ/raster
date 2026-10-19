(ns raster.ad.owned-active-index-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as reverse]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(deftm packed-prior [theta :- (Array double) count :- Long stride :- Long offset :- Long] :- Double
  (par/reduce acc 0.0 i count
    (let [v (aget theta (+ offset (* stride (long i))))]
      (n/+ acc (n/* v v)))))
(deftm let-bound-packed-prior [theta :- (Array double) count :- Long] :- Double
  (let [total (par/reduce acc 0.0 i count
                (let [v (aget theta (+ 1 (long i)))] (n/+ acc (n/* v v))))]
    (n/* 2.0 total)))

(deftm packed-prior-gradient [theta :- (Array double) count :- Long stride :- Long offset :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'packed-prior :wrt [0]) theta count stride offset)
        gradient (nth result 1)]
    (par/map [i 7] (double (if (zero? i) (nth result 0) (aget gradient (dec i)))))))
(deftm let-bound-packed-gradient [theta :- (Array double) count :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'let-bound-packed-prior :wrt [0]) theta count)
        gradient (nth result 1)]
    (par/map [i 4] (double (if (zero? i) (nth result 0) (aget gradient (dec i)))))))

(deftm effectful-packed-prior [theta :- (Array double) events :- (Array long) count :- Long] :- Double
  (par/reduce acc 0.0 i count
    (n/+ acc (aget theta (do (aset events 0 (inc (aget events 0))) i)))))
(deftm active-index-prior [theta :- (Array double) count :- Long] :- Double
  (par/reduce acc 0.0 i count
    (n/+ acc (aget theta (long (aget theta i))))))
(deftm packed-float-prior [theta :- (Array float) count :- Long] :- Double
  (par/reduce acc 0.0 i count
    (let [v (double (aget theta (+ 1 (long i))))] (n/+ acc (n/* v v)))))
(deftm packed-product [theta :- (Array double) count :- Long stride :- Long] :- Double
  (par/reduce acc 1.0 i count (n/* acc (aget theta (* stride (long i))))))

(use-fixtures :once (fn [run] (n/* 1.0 1.0) (n/+ 1.0 1.0) (run)))

(deftest inactive-owned-indices-scatter-active-array-cotangents
  (let [prepared (reverse/prepare-owned-bytecode-value+grad #'packed-prior [0])
        gradient (#'reverse/compile-prepared-value+grad prepared)]
    (doseq [count [0 1 3] stride [0 1 2] offset [0 1]]
      (let [theta (double-array [0.2 -0.3 0.7 1.2 -0.8 0.4])
            indices (map #(+ offset (* stride %)) (range count))
            expected (reduce (fn [g i] (update g i + (* 2.0 (aget theta i))))
                             (vec (repeat 6 0.0)) indices)
            [value actual inactive-count inactive-stride inactive-offset]
            (gradient theta count stride offset)]
        (is (= (reduce + 0.0 (map #(let [v (aget theta %)] (* v v)) indices)) value))
        (is (= expected (vec actual)))
        (is (every? nil? [inactive-count inactive-stride inactive-offset]))))
    (is (= [4.0 [4.0] nil nil nil]
           (let [[v g & rest] (gradient (double-array [2.0]) 1 1 4294967296)]
             (into [v (vec g)] rest))))
    (is (thrown? ArrayIndexOutOfBoundsException (gradient (double-array [2.0]) 1 1 -1)))
    (is (thrown? NullPointerException (gradient nil 1 1 0)))))

(deftest empty-reduction-does-not-evaluate-array-reads
  (let [gradient (#'reverse/compile-prepared-value+grad
                   (reverse/prepare-owned-bytecode-value+grad #'packed-prior [0]))]
    (doseq [count [-1 0]]
      (is (= [0.0 [] nil nil nil]
             (let [[v g & inactive] (gradient (double-array 0) count 1 -1)]
               (into [v (vec g)] inactive)))))))

(deftest let-bound-reduction-retains-its-emitted-object-result-carrier
  (let [gradient (#'reverse/compile-prepared-value+grad
                   (reverse/prepare-owned-bytecode-value+grad #'let-bound-packed-prior [0]))
        [value actual] (gradient (double-array [8.0 0.5 -0.3]) 2)]
    (is (= 0.6799999999999999 value))
    (is (= [0.0 2.0 -1.2] (vec actual)))))

(deftest invoked-aot-preserves-let-bound-reduction-result-completion
  (let [aot (pipeline/compile-aot #'let-bound-packed-gradient)
        runtime (reverse/value+grad #'let-bound-packed-prior :wrt [0])]
    (doseq [count [-1 0 1 2]]
      (let [theta (double-array [8.0 0.5 -0.3])
            [value gradient] (runtime theta count)
            expected (into [(let-bound-packed-prior theta count)]
                           (for [j (range 3)]
                             (let [plus (aclone theta) minus (aclone theta)]
                               (aset plus j (+ (aget plus j) 1e-6))
                               (aset minus j (- (aget minus j) 1e-6))
                               (/ (- (let-bound-packed-prior plus count)
                                     (let-bound-packed-prior minus count)) 2e-6))))]
        (is (= (into [value] gradient) (vec (aot theta count))))
        (is (every? true? (map #(< (Math/abs (double (- %1 %2))) 1e-8)
                                 (into [value] gradient) expected)))))))

(deftest packed-active-array-runtime-and-invoked-aot-agree-with-finite-differences
  (let [aot (pipeline/compile-aot #'packed-prior-gradient)
        runtime (reverse/value+grad #'packed-prior :wrt [0])]
    (doseq [theta [[0.2 -0.3 0.7 1.2 -0.8 0.4] [0.4 0.1 -0.7 0.2 0.8 1.4]]
            stride [0 1 2]]
      (let [args [(double-array theta) 3 stride 1]
            [value gradient] (apply runtime args)
            actual (into [value] gradient)
            expected (into [(apply packed-prior args)]
                           (for [j (range 6)]
                             (let [plus (double-array (update theta j + 1e-6))
                                   minus (double-array (update theta j - 1e-6))]
                               (/ (- (packed-prior plus 3 stride 1)
                                     (packed-prior minus 3 stride 1)) 2e-6))))]
        (is (= actual (vec (apply aot args))))
        (is (every? true? (map #(< (Math/abs (double (- %1 %2))) 1e-8) actual expected)))))))

(deftest contradictory-logical-and-physical-carry-declarations-still-refuse
  ;; The current frontend stamps this Float-array reduction Float despite its
  ;; actual Double carry. This extension must not guess a tangent declaration
  ;; from the physical slot and silently admit that separate frontend gap.
  (let [source (raster.core/resolve-deftm-var #'packed-float-prior)
        receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))]
    (raster.core/ensure-current-owned-bytecode! receiver)
    (let [emission (:raster.core/bytecode-emission (meta receiver))
          original (first (vals (:reduction-catalog emission)))
          observed (bytecode/retained-reduction-emission emission original)]
      (is (= 'float (:raster.type/tag (meta original))))
      (is (= :double (get-in observed [:loop :locals (second (get-in observed [:loop :variables])) :type])))
      (is (= :ad-unclosed-bytecode-reduction-entry
             (try (reverse/prepare-owned-bytecode-value+grad #'packed-float-prior [0])
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))))

(deftest nonadditive-array-captures-use-the-same-saved-pullback-rules
  (let [gradient (#'reverse/compile-prepared-value+grad
                   (reverse/prepare-owned-bytecode-value+grad #'packed-product [0]))]
    (doseq [theta [[0.5 0.25 -0.5] [0.0 0.25 -0.5]] stride [0 1]]
      (let [[value actual] (gradient (double-array theta) 3 stride)
            expected (mapv (fn [j]
                             (/ (- (packed-product (double-array (update theta j + 1e-6)) 3 stride)
                                   (packed-product (double-array (update theta j - 1e-6)) 3 stride)) 2e-6))
                           (range 3))]
        (is (= (packed-product (double-array theta) 3 stride) value))
        (is (every? true? (map #(< (Math/abs (double (- %1 %2))) 1e-8) actual expected)))))))

(deftest saved-pullback-does-not-reread-mutated-active-array-values
  (let [prepared (reverse/prepare-owned-bytecode-value+grad #'packed-prior [0])
        {:keys [params tags source-ns program helpers]} prepared
        prefix (vec (take (* 2 (count helpers)) (second (first (:walked-body prepared)))))
        entry (get-in prepared [:source-owner :entry])
        pair (bytecode/compile-typed-impl!
               (str "raster.compiled.OwnedPackedPair_" (clojure.lang.RT/nextID))
               params tags 'Object [(list 'let* prefix (:form program))] source-ns
               (:interface-class entry) nil nil (mapv :spec (:input-regions entry)))
        theta (double-array [0.2 -0.3 0.7 1.2 -0.8 0.4])
        [value pullback] (pair theta 3 0 1)]
    (is (= 0.27 value))
    (aset theta 1 99.0)
    (is (= [0.0 -1.7999999999999998 0.0 0.0 0.0 0.0]
           (vec (first (pullback 1.0)))))))

(deftest array-activity-does-not-authorize-active-or-effectful-index-computation
  (doseq [v [#'effectful-packed-prior #'active-index-prior]]
    (is (= :ad-unclosed-bytecode-body-region
           (try (reverse/prepare-owned-bytecode-value+grad v [0])
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))

(deftest prepared-array-step-keeps-absent-cotangents-on-zero-trips
  (with-bindings {#'reverse/*gensym-counter* (atom 0)
                  #'reverse/*source-memory-context* {:internal-region? true :values []}}
    (let [acc (with-meta 'carry {:raster.type/tag 'double})
          theta (with-meta 'theta {:raster.type/tag 'doubles})
          read (with-meta 'read_value {:raster.type/tag 'double})
          square (with-meta 'square {:raster.type/tag 'double})
          result (with-meta 'step_result {:raster.type/tag 'double})
          program (#'reverse/derive-flat-product-program
                    [read (list 'clojure.core/aget theta 'i)
                     square (list 'clojure.core/* read read)
                     result (list 'clojure.core/+ acc square)] result [acc theta])
          spec {:acc-sym acc :init-expr 0.0 :idx-sym 'i :bound-expr 'size
                :body-expr '(throw (ex-info "unowned source used" {}))
                :agets [] :scalar-bindings [] :body-result nil :free-syms #{theta}
                :typed-double? false :replay-values #{} :step-program program}
          record (#'reverse/assemble-ordered-reduce spec [theta])
          bindings (vec (concat (:forward-bindings record)
                                [(:d-acc-sym record) 1.0]
                                (mapcat identity (:backward-maps record))
                                (mapcat identity (:backward-reduces record))))
          invoke (eval (list 'fn [theta 'size]
                            (list 'let* bindings
                                  [(:reduce-result-sym record) (first (:d-scalar-syms record))])))]
      (doseq [size [-1 0]] (is (= [0.0 nil] (invoke nil size))))
      (is (= [0.25 [1.0]]
             (let [[v g] (invoke (double-array [0.5]) 1)] [v (vec g)])))
      (doseq [invalid [(dissoc spec :step-program)
                       (assoc spec :typed-double? true)
                       (assoc-in spec [:step-program :writes] #{theta})
                       (assoc-in spec [:step-program :active-inputs] [acc])]]
        (is (thrown? clojure.lang.ExceptionInfo (#'reverse/assemble-ordered-reduce invalid [theta])))))))
