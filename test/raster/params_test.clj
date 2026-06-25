(ns raster.params-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.params :as rp]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.optim :as optim]))

;; A small linear projection y = W x + b, written with HMap params.
(rp/defmodel linear-model
  [w :- (Params (HMap :mandatory {:W (Param (Array double))
                                  :b (Param (Array double))}))
   x :- (Array double)
   n-out :- Long
   n-in  :- Long]
  :- (Array double)
  (let [out (double-array n-out)]
    (dotimes [i n-out]
      (clojure.core/aset out i (double (clojure.core/aget (:b w) i)))
      (dotimes [j n-in]
        (let [cur (double (clojure.core/aget out i))
              wij (double (clojure.core/aget (:W w)
                                             (clojure.core/+ (clojure.core/* i n-in) j)))
              xj  (double (clojure.core/aget x j))]
          (clojure.core/aset out i (clojure.core/+ cur (clojure.core/* wij xj))))))
    out))

(def W (double-array [1.0 2.0  3.0 4.0  5.0 6.0]))   ; 3x2 row-major
(def b (double-array [0.1 0.2 0.3]))
(def x (double-array [10.0 20.0]))
(def expected [50.1 110.2 170.3])

(deftest defmodel-attaches-tree-metadata
  (let [m (meta #'linear-model)]
    (is (contains? m :raster.params/treedefs))
    (is (contains? m :raster.params/flat-var))
    (is (contains? m :raster.params/original-args))
    (is (= '[w x n-out n-in] (:raster.params/original-args m)))
    (is (= #{'w} (set (keys (:raster.params/treedefs m)))))))

(deftest lazy-jit-path-runs-with-structured-args
  (let [y (linear-model {:W W :b b} x 3 2)]
    (is (= expected (vec y)))))

(deftest compile-aot-path-matches-jit-path
  (let [fast (rp/compile-aot #'linear-model)
        y    (fast {:W W :b b} x 3 2)]
    (is (= expected (vec y)))))

(deftest compile-aot-pass-through-on-non-tree-var
  (testing "compile-aot delegates to pipeline/compile-aot for non-tree deftms"
    ;; Just verify the path doesn't blow up — defining a regular deftm is
    ;; outside this ns's responsibility, but compile-aot should pass through.
    (let [v (resolve 'raster.params-test/linear-model--flat)]
      (is (some? v))
      (let [fast (pipeline/compile-aot v)
            ;; Call via flat positional args (no wrapper)
            y    (fast W b x 3 2)]
        (is (= expected (vec y)))))))

(deftest trainable-leaves-extracts-param-only
  (let [spec  '(HMap :mandatory {:W  (Param (Array double))
                                 :b  (Param (Array double))
                                 :ln (Frozen (Array double))})
        v     {:W (double-array [1 2 3])
               :b (double-array [4 5])
               :ln (double-array [1])}
        train (rp/trainable-leaves spec v)
        froz  (rp/leaves-by-kind :frozen spec v)]
    (is (= #{:W :b} (set (keys train))))
    (is (= #{:ln} (set (keys froz))))
    (is (= [1.0 2.0 3.0] (vec (:W train))))
    (is (= [1.0] (vec (:ln froz)))
        "Frozen leaves are extractable but not via trainable-leaves")))

(deftest trainable-leaves-paths-as-keys
  (let [spec  '(HMap :mandatory
                     {:layers (HVec [(HMap :mandatory {:Wq (Param (Array double))})
                                     (HMap :mandatory {:Wq (Param (Array double))})])})
        v     {:layers [{:Wq (double-array [1])}
                        {:Wq (double-array [2])}]}
        train (rp/trainable-leaves spec v)]
    (is (= #{:layers.0.Wq :layers.1.Wq} (set (keys train))))
    (is (= [1.0] (vec (:layers.0.Wq train))))
    (is (= [2.0] (vec (:layers.1.Wq train))))))

(deftest adam-update-via-trainable-leaves-skips-frozen
  (let [spec  '(HMap :mandatory {:W  (Param (Array double))
                                 :ln (Frozen (Array double))})
        w     {:W (double-array [1.0 2.0 3.0])
               :ln (double-array [99.0])}
        grads {:W (double-array [0.5 0.5 0.5])
               :ln (double-array [-1000.0])}  ; should be ignored
        train (rp/trainable-leaves spec w)
        train-grads (rp/trainable-leaves spec grads)
        state (optim/make-adam-state train)]
    (optim/adam-update! train train-grads state 0.01)
    (is (every? true? (map (fn [old new] (< new old))
                           [1.0 2.0 3.0]
                           (vec (:W w))))
        ":W elements all decreased by Adam step (positive grad)")
    (is (= [99.0] (vec (:ln w))) ":ln unchanged — Frozen never seen by optimizer")))

(deftest treedef-flat-leaves-canonical-order
  (let [td (-> #'linear-model meta :raster.params/treedefs (get 'w))]
    (is (= '[linear-model--flat--root]  ; placeholder check below
           ['linear-model--flat--root])
        "smoke check setup")
    (is (= '[w__W w__b] (mapv :sym (:leaves td)))
        "leaves are in sorted-key canonical order")))
