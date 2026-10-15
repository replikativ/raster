(ns raster.ad.normalized-map-chain-runtime-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.numeric]
            [raster.par]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- recertify [owner]
  (assoc owner :certificate (#'canonical/certificate owner)))

(defn- typed-chain [extent]
  (frontend/form->program
   (walk/postwalk-replace
    {'chain-extent extent}
    '(let* [squared (raster.par/pmap i chain-extent float
                                     (raster.numeric/* (clojure.core/aget x i)
                                                       (clojure.core/aget x i)))
            scaled (raster.par/pmap j chain-extent float
                                    (raster.numeric/* (clojure.core/aget squared j)
                                                      (clojure.core/aget q j)))]
           scaled))
   {:dtype :float :array-types {'x :float 'q :float}
    :scalar-types (if (symbol? extent) {extent :long} {})}))

(defn- typed-three-boundary-chain [extent dt]
  (frontend/form->program
   (walk/postwalk-replace
    {'chain-extent extent 'lane-dtype (if (= dt :float) 'float 'double)}
    '(let* [paired (raster.par/pmap i chain-extent lane-dtype
                                    (raster.numeric/* (clojure.core/aget x i)
                                                      (clojure.core/aget q i)))
            result (raster.par/pmap j chain-extent lane-dtype
                                    (raster.numeric/* (clojure.core/aget paired j)
                                                      (clojure.core/aget r j)))]
           result))
   {:dtype dt :array-types {'x dt 'q dt 'r dt}
    :scalar-types (if (symbol? extent) {extent :long} {})}))

(defn- typed-three-node-reuse [extent]
  (frontend/form->program
   (walk/postwalk-replace
    {'chain-extent extent}
    '(let* [paired (raster.par/pmap i chain-extent float
                                    (raster.numeric/- (clojure.core/aget x i)
                                                      (clojure.core/aget q i)))
            reused (raster.par/pmap j chain-extent float
                                    (raster.numeric/* (clojure.core/aget paired j)
                                                      (clojure.core/aget x j)))
            result (raster.par/pmap k chain-extent float
                                    (raster.numeric/* (clojure.core/aget reused k)
                                                      (clojure.core/aget r k)))]
           result))
   {:dtype :float :array-types {'x :float 'q :float 'r :float}
    :scalar-types (if (symbol? extent) {extent :long} {})}))

(defn- typed-diamond [extent]
  (frontend/form->program
   (walk/postwalk-replace
    {'diamond-extent extent}
    '(let* [squared (raster.par/pmap i diamond-extent float
                                     (raster.numeric/* (clojure.core/aget x i)
                                                       (clojure.core/aget x i)))
            shifted (raster.par/pmap j diamond-extent float
                                     (raster.numeric/+ (clojure.core/aget squared j)
                                                       (clojure.core/float 1.0)))
            fourth (raster.par/pmap k diamond-extent float
                                    (raster.numeric/* (clojure.core/aget squared k)
                                                      (clojure.core/aget squared k)))
            joined (raster.par/pmap l diamond-extent float
                                    (raster.numeric/+ (clojure.core/aget shifted l)
                                                      (clojure.core/aget fourth l)))]
           joined))
   {:dtype :float :array-types {'x :float}
    :scalar-types (if (symbol? extent) {extent :long} {})}))

(defn- scalar-capture-chain []
  (frontend/form->program
   '(let* [squared (raster.par/pmap i n float
                                    (raster.numeric/* (clojure.core/aget x i)
                                                      (clojure.core/aget x i)))
           scaled (raster.par/pmap j n float
                                   (raster.numeric/* (clojure.core/aget squared j) scale))]
          scaled)
   {:dtype :float :array-types {'x :float}
    :scalar-types {'n :long 'scale :float}}))

(defn- four-map-spine [extent]
  (frontend/form->program
   (walk/postwalk-replace
    {'spine-extent extent}
    '(let* [squared (raster.par/pmap i spine-extent float
                                     (raster.numeric/* (clojure.core/aget x i)
                                                       (clojure.core/aget x i)))
            fourth (raster.par/pmap j spine-extent float
                                    (raster.numeric/* (clojure.core/aget squared j)
                                                      (clojure.core/aget squared j)))
            scaled (raster.par/pmap k spine-extent float
                                    (raster.numeric/* (clojure.core/aget fourth k) scale))
            result (raster.par/pmap l spine-extent float
                                    (raster.numeric/* (clojure.core/aget scaled l)
                                                      (clojure.core/aget q l)))]
           result))
   {:dtype :float :array-types {'x :float 'q :float}
    :scalar-types (merge {'scale :float} (if (symbol? extent) {extent :long} {}))}))

(defn- evidence
  ([source] (evidence source 'x))
  ([source active]
   (normalized/linearize-map-chain-program
    source (the-ns 'raster.ad.normalized-map-chain-runtime-test)
    {:active-input active})))

(defn- arguments [prepared environment]
  (mapv environment (:inputs prepared)))

(defn- fbits [x] (Float/floatToRawIntBits (float x)))
(defn- fmul [left right] (float (* (float left) (float right))))
(defn- fadd [left right] (float (+ (float left) (float right))))

(defn- alter-facts [program f]
  (apply list (assoc (vec program) 1 (f (soac/facts program)))))

(deftest canonical-owner-retains-source-abi-and-rederived-execution-scope
  (let [source (typed-chain 'n)
        reordered (soac/make (update (soac/facts source) :inputs #(vec (reverse %)))
                             (soac/equations source) (soac/outputs source))
        owner (canonical/adapt-map (evidence reordered 'q))
        prepared (runtime/prepare-owner owner)
        source-ids (mapv second (soac/equations reordered))
        execution-id (second (first (soac/equations (:execution-source owner))))
        execution-lambda (:lambda (soac/operation-parts
                                   (first (soac/equations (:execution-source owner)))))
        lambda-layout (soac/lambda-parts execution-lambda)
        outer-scalars (set (concat (:parameters lambda-layout)
                                   (map :id (:locals lambda-layout))))
        coefficients (set (keys (:residual-bindings owner)))
        certified-scalars (:scalar-values
                           (normalized/execution-map-scope
                            (get-in owner [:derivation :bundle])))
        coefficient (first coefficients)
        changed-coefficient (with-meta coefficient
                              (assoc (meta coefficient) :raster.type/tag 'double))]
    (is (identical? reordered (:source owner)))
    (is (= (vec (reverse (:inputs (soac/facts source))))
           (mapv :id (get-in owner [:interface :inputs]))))
    (is (= 2 (count (soac/equations (:source owner)))))
    (is (= '[map] (mapv soac/operation-kind
                         (soac/equations (:execution-source owner)))))
    (is (every? (partial #'canonical/exact-scalar-member? certified-scalars) coefficients))
    (is (not (#'canonical/exact-scalar-member? certified-scalars changed-coefficient))
        "ordinary symbol equality cannot erase retained coefficient type evidence")
    (is (some (complement outer-scalars) coefficients)
        "ANF coefficients are certified by the scalar primal, not guessed from outer syntax")
    (doseq [[coefficient binding] (:residual-bindings owner)]
      (is (= source-ids (get-in binding [:origin :source-equations])))
      (is (= execution-id (get-in binding [:origin :execution-equation])))
      (is (= coefficient (get-in binding [:origin :scalar-value])))
      (is (= :owned-array (:storage binding)))
      (is (= :owned (get-in binding [:value :ownership]))))
    (is (identical? owner (canonical/validate! owner)))
    (let [xs (float-array [2.0 -3.0])
          qs (float-array [0.5 4.0])
          args (arguments prepared {'n (long 2) 'x xs 'q qs})
          [output pullback] ((:vjp prepared) args)
          slots (pullback (float-array [-0.25 2.0]))
          q-slot (.indexOf ^java.util.List (:inputs prepared) 'q)
          x-slot (.indexOf ^java.util.List (:inputs prepared) 'x)]
      (is (= (mapv fbits [2.0 36.0]) (mapv fbits output)))
      (is (= (mapv fbits [-1.0 18.0]) (mapv fbits (nth slots q-slot))))
      (is (nil? (nth slots x-slot))))))

(deftest two-stage-runtime-snapshots-both-nonlinear-stages
  (let [checked (evidence (typed-chain 'n))
        prepared (runtime/prepare checked)
        xs0 (mapv float [4097.0 -3.0 0.5])
        qs0 (mapv float [0.25 -2.0 4.0])
        dx0 (mapv float [0.5 -0.25 2.0])
        dy0 (mapv float [-0.5 1.25 3.0])
        xs (float-array xs0) qs (float-array qs0)
        dx (float-array dx0) dy (float-array dy0)
        args (arguments prepared {'n (long 3) 'x xs 'q qs})
        [output pullback] ((:vjp prepared) args)
        [jvp-output tangent] ((:jvp prepared) args dx)
        primal (mapv (fn [x q] (float (* (float (* x x)) q))) xs0 qs0)
        jvp (mapv (fn [x q d]
                    (let [producer (float (* (float (* (float 2.0) x)) d))]
                      (float (* producer q))))
                  xs0 qs0 dx0)
        gradient (mapv (fn [x q seed]
                         (let [consumer-pull (float (* q seed))]
                           (float (* (float (* (float 2.0) x)) consumer-pull))))
                       xs0 qs0 dy0)]
    (is (= (mapv fbits primal) (mapv fbits output) (mapv fbits jvp-output)))
    (is (= (mapv fbits jvp) (mapv fbits tangent)))
    (dotimes [i 3]
      (aset xs i (float 99.0))
      (aset qs i (float 88.0))
      (aset output i (float -77.0)))
    (let [slots (pullback dy)
          x-slot (.indexOf ^java.util.List (:inputs prepared) 'x)
          q-slot (.indexOf ^java.util.List (:inputs prepared) 'q)
          grad (nth slots x-slot)]
      (is (= (mapv fbits gradient) (mapv fbits grad)))
      (is (nil? (nth slots q-slot)))
      (is (every? nil? (keep-indexed #(when (not= %1 x-slot) %2) slots)))
      (dotimes [i 3] (aset grad i (float 42.0)))
      (is (= (mapv fbits gradient) (mapv fbits (nth (pullback dy) x-slot)))))))

(deftest two-stage-chain-reuses-fused-proof-for-reversed-multiple-activity
  (let [source (typed-chain 'n)
        checked (normalized/linearize-map-chain-program
                 source (the-ns 'raster.ad.normalized-map-chain-runtime-test)
                 {:active-inputs '[q x]})
        owner (canonical/adapt-map checked)
        prepared (runtime/prepare-owner owner)
        xs0 (mapv float [4097.0 -3.0 0.5 1.0e-20])
        qs0 (mapv float [0.25 -2.0 4.0 1.0])
        dq0 (mapv float [-0.5 0.75 2.0 0.0])
        dx0 (mapv float [0.25 -0.5 1.5 1.0e-25])
        dy0 (mapv float [-0.5 1.25 3.0 1.0e-25])
        xs (float-array xs0) qs (float-array qs0)
        dq (float-array dq0) dx (float-array dx0) dy (float-array dy0)
        args (arguments prepared {'n (long 4) 'x xs 'q qs})
        [output pullback] ((:vjp prepared) args)
        [jvp-output tangent] ((:jvp prepared) args [dq dx])
        primal (mapv (fn [x q] (fmul (fmul x x) q)) xs0 qs0)
        jvp (mapv (fn [x q dq dx]
                    (let [square (fmul x x)
                          square-tangent (fadd (fmul dx x) (fmul dx x))]
                      (fadd (fmul square-tangent q) (fmul dq square))))
                  xs0 qs0 dq0 dx0)
        gx (mapv (fn [x q seed]
                   (let [consumer-pull (fmul seed q)]
                     (fadd (fmul consumer-pull x) (fmul consumer-pull x))))
                 xs0 qs0 dy0)
        gq (mapv (fn [x seed] (fmul seed (fmul x x))) xs0 dy0)]
    (is (= '[q x] (get-in owner [:interface :active-inputs])))
    (is (= (mapv fbits primal) (mapv fbits output) (mapv fbits jvp-output)))
    (is (= (mapv fbits jvp) (mapv fbits tangent)))
    (is (= 2 (fbits (nth jvp 3))) "ordered square JVP retains two subnormal contributions")
    (is (= 2 (fbits (nth gx 3))) "ordered square VJP retains two subnormal contributions")
    (dotimes [i 4]
      (aset xs i (float 99.0))
      (aset qs i (float -88.0))
      (aset output i (float 77.0)))
    (let [slots (pullback dy)
          x-slot (.indexOf ^java.util.List (:inputs prepared) 'x)
          q-slot (.indexOf ^java.util.List (:inputs prepared) 'q)]
      (is (= (mapv fbits gx) (mapv fbits (nth slots x-slot))))
      (is (= (mapv fbits gq) (mapv fbits (nth slots q-slot))))
      (is (every? nil? (keep-indexed #(when (not (contains? #{x-slot q-slot} %1)) %2) slots))))))

(deftest three-boundary-node-composition-preserves-ordered-float-checkpoints
  (let [source (typed-three-boundary-chain 'n :float)
        checked (normalized/linearize-map-chain-program
                 source (the-ns 'raster.ad.normalized-map-chain-runtime-test)
                 {:active-inputs '[r q x]})
        owner (canonical/adapt-map checked)
        prepared (runtime/prepare-owner owner)
        xs0 (mapv float [0.1 -3.0 1.0e-20])
        qs0 (mapv float [0.2 4.0 1.0])
        rs0 (mapv float [-0.3 0.5 2.0])
        dx0 (mapv float [0.25 -0.5 1.0e-25])
        dq0 (mapv float [-0.75 0.125 0.0])
        dr0 (mapv float [0.5 1.25 0.0])
        dy0 (mapv float [0.1 -0.75 1.0e-25])
        xs (float-array xs0) qs (float-array qs0) rs (float-array rs0)
        args (arguments prepared {'n (long 3) 'x xs 'q qs 'r rs})
        [output pullback] ((:vjp prepared) args)
        [jvp-output tangent] ((:jvp prepared) args
                              [(float-array dr0) (float-array dq0) (float-array dx0)])
        primal (mapv #(fmul (fmul %1 %2) %3) xs0 qs0 rs0)
        tangent-oracle
        (mapv (fn [x q r dx dq dr]
                (let [paired (fmul x q)
                      dpaired (fadd (fmul dx q) (fmul x dq))]
                  (fadd (fmul dpaired r) (fmul paired dr))))
              xs0 qs0 rs0 dx0 dq0 dr0)
        ga (mapv fmul dy0 rs0)
        gx (mapv fmul ga qs0)
        gq (mapv fmul ga xs0)
        gr (mapv #(fmul %1 (fmul %2 %3)) dy0 xs0 qs0)]
    (is (= '[r q x] (get-in owner [:interface :active-inputs])))
    (is (= (mapv fbits primal) (mapv fbits output) (mapv fbits jvp-output)))
    (is (= (mapv fbits tangent-oracle) (mapv fbits tangent)))
    (dotimes [i 3]
      (aset xs i (float 91.0)) (aset qs i (float 92.0)) (aset rs i (float 93.0)))
    (let [slots (pullback (float-array dy0))
          positions (zipmap (:inputs prepared) (range))]
      (is (= (mapv fbits gx) (mapv fbits (nth slots (positions 'x)))))
      (is (= (mapv fbits gq) (mapv fbits (nth slots (positions 'q)))))
      (is (= (mapv fbits gr) (mapv fbits (nth slots (positions 'r))))))))

(deftest inactive-primal-node-snapshots-its-output-for-r-only-ad
  (doseq [[dt make-array epsilon]
          [[:float float-array 2.0e-5]
           [:double double-array 1.0e-10]]]
    (let [source (typed-three-boundary-chain 'n dt)
          checked (normalized/linearize-map-chain-program
                   source (the-ns 'raster.ad.normalized-map-chain-runtime-test)
                   {:active-input 'r})
          nodes (get-in checked [:map-programs :proof :nodes])
          owner (canonical/adapt-map checked)
          prepared (runtime/prepare-owner owner)
          xs0 [0.25 -2.0 3.0] qs0 [-0.5 4.0 0.125] rs0 [2.0 -0.75 1.5]
          xs (make-array xs0) qs (make-array qs0) rs (make-array rs0)
          args (arguments prepared {'n (long 3) 'x xs 'q qs 'r rs})
          [output pullback] ((:vjp prepared) args)
          [_ tangent] ((:jvp prepared) args (make-array [0.5 -1.0 2.0]))
          primal (mapv #(* %1 %2 %3) xs0 qs0 rs0)
          dr-oracle (mapv #(* %1 %2 %3) xs0 qs0 [0.5 -1.0 2.0])]
      (is (= [false true] (mapv #(contains? % :scalar-derivation) nodes)))
      (is (contains? (first nodes) :primal-derivation))
      (is (= {} (:stage-interface (first nodes))))
      (is (every? true? (map #(<= (Math/abs (- (double %1) (double %2))) epsilon)
                              primal output)))
      (is (every? true? (map #(<= (Math/abs (- (double %1) (double %2))) epsilon)
                              dr-oracle tangent)))
      (dotimes [i 3]
        (if (= dt :float)
          (do (aset-float xs i (float 0.0))
              (aset-float qs i (float 0.0))
              (aset-float rs i (float 0.0)))
          (do (aset-double xs i (double 0.0))
              (aset-double qs i (double 0.0))
              (aset-double rs i (double 0.0)))))
      (let [slots (pullback (make-array [1.0 -0.5 2.0]))
            r-slot (.indexOf ^java.util.List (:inputs prepared) 'r)
            expected (mapv #(* %1 %2 %3) [1.0 -0.5 2.0] xs0 qs0)]
        (is (every? true? (map #(<= (Math/abs (- (double %1) (double %2))) epsilon)
                                expected (nth slots r-slot))))
        (is (every? nil? (keep-indexed #(when (not= %1 r-slot) %2) slots)))))))

(deftest three-boundary-node-composition-retains-zero-cardinality-ports
  (let [prepared (runtime/prepare
                  (normalized/linearize-map-chain-program
                   (typed-three-boundary-chain 0 :float)
                   (the-ns 'raster.ad.normalized-map-chain-runtime-test)
                   {:active-inputs '[r x]}))
        empty-array (float-array 0)
        args (arguments prepared {'x empty-array 'q (float-array 0) 'r (float-array 0)})
        [output pullback] ((:vjp prepared) args)
        [_ tangent] ((:jvp prepared) args [(float-array 0) (float-array 0)])
        slots (pullback (float-array 0))]
    (is (zero? (alength output)))
    (is (zero? (alength tangent)))
    (is (= 2 (count (filter some? slots))))
    (is (every? zero? (map alength (filter some? slots))))))

(deftest passive-external-reuse-allows-three-primal-nodes-but-active-reuse-refuses
  (let [source (typed-three-node-reuse 'n)
        checked (normalized/linearize-map-chain-program
                 source (the-ns 'raster.ad.normalized-map-chain-runtime-test)
                 {:active-input 'r})
        nodes (get-in checked [:map-programs :proof :nodes])
        owner (canonical/adapt-map checked)
        bindings (:residual-bindings owner)
        primal (get-in checked [:map-programs :primal])]
    (is (= 3 (count nodes)))
    (is (= [false false true] (mapv #(contains? % :scalar-derivation) nodes)))
    (is (= :ad-normalized-map-chain-boundary
           (reason #(normalized/linearize-map-chain-program
                     source (the-ns 'raster.ad.normalized-map-chain-runtime-test)
                     {:active-input 'x}))))
    (doseq [[[node-id _] binding] bindings]
      (is (= node-id (get-in binding [:origin :execution-equation])))
      (is (= [node-id] (get-in binding [:origin :source-equations])))
      (is (= 1 (count (filter #(some #{(:id binding)} (nth % 2))
                             (soac/equations primal)))))
      (is (= (get-in binding [:producer :equation])
             (second (first (filter #(some #{(:id binding)} (nth % 2))
                                    (soac/equations primal)))))))))

(deftest inactive-node-raw-call-remains-in-the-execution-staleness-fence
  (let [checked (normalized/linearize-map-chain-program
                 (typed-three-node-reuse 'n)
                 (the-ns 'raster.ad.normalized-map-chain-runtime-test)
                 {:active-input 'r})
        prepared (runtime/prepare checked)
        args (arguments prepared {'n (long 1) 'x (float-array [2.0])
                                  'q (float-array [0.5]) 'r (float-array [3.0])})]
    (with-redefs [raster.numeric/- (fn [& _] (float 99.0))]
      (is (= :ad-normalized-execution-callee
             (reason #((:jvp prepared) args (float-array [1.0]))))))))

(deftest closed-diamond-runtime-obeys-fused-float-order-and-snapshots
  (let [prepared (runtime/prepare (evidence (typed-diamond 'n)))
        xs0 (mapv float [0.1 -3.0 4097.0])
        dx0 (mapv float [0.25 0.75 -0.5])
        dy0 (mapv float [0.1 1.25 -0.5])
        xs (float-array xs0)
        dx (float-array dx0)
        dy (float-array dy0)
        args (arguments prepared {'n (long 3) 'x xs})
        [output pullback] ((:vjp prepared) args)
        [jvp-output tangent] ((:jvp prepared) args dx)
        primal (mapv (fn [x]
                       (fadd (fadd (fmul x x) (float 1.0))
                             (fmul (fmul x x) (fmul x x))))
                     xs0)
        jvp (mapv (fn [x d]
                    (let [left-da (fadd (fmul d x) (fmul d x))
                          right-a (fmul x x)
                          right-da (fadd (fmul d x) (fmul d x))]
                      (fadd left-da
                            (fadd (fmul right-da right-a)
                                  (fmul right-da right-a)))))
                  xs0 dx0)
        gradient (mapv (fn [x seed]
                         (let [a (fmul x x)
                               t (fmul (fmul seed a) x)
                               u (fmul seed x)]
                           (fadd (fadd (fadd (fadd (fadd t t) t) t) u) u)))
                       xs0 dy0)]
    (is (= (mapv fbits primal) (mapv fbits output) (mapv fbits jvp-output)))
    (is (= (mapv fbits jvp) (mapv fbits tangent)))
    (dotimes [i 3]
      (aset xs i (float 99.0))
      (aset output i (float -77.0)))
    (let [slots (pullback dy)
          x-slot (.indexOf ^java.util.List (:inputs prepared) 'x)]
      (is (= (mapv fbits gradient) (mapv fbits (nth slots x-slot))))
      (is (every? nil? (keep-indexed #(when (not= x-slot %1) %2) slots)))))
  (let [prepared (runtime/prepare (evidence (typed-diamond 0)))
        args (arguments prepared {'x (float-array 0)})
        [output pullback] ((:vjp prepared) args)
        [jvp-output tangent] ((:jvp prepared) args (float-array 0))
        slots (pullback (float-array 0))]
    (is (= [] (vec output) (vec jvp-output) (vec tangent)))
    (is (= [] (vec (nth slots (.indexOf ^java.util.List (:inputs prepared) 'x)))))))

(deftest four-stage-runtime-preserves-checkpoints-order-captures-and-snapshots
  (let [source (four-map-spine 'n)
        reordered (soac/make (update (soac/facts source) :inputs #(vec (reverse %)))
                             (soac/equations source) (soac/outputs source))
        checked (evidence reordered)
        owner (canonical/adapt-map checked)
        residual-plan (canonical/execution-residual-plan owner)
        prepared (runtime/prepare-owner owner)
        xs0 (mapv float [4097.0 -3.0 0.5])
        qs0 (mapv float [0.5 -2.0 4.0])
        dx0 (mapv float [0.25 -0.5 2.0])
        dy0 (mapv float [-0.5 1.25 3.0])
        scale (float 0.25)
        xs (float-array xs0) qs (float-array qs0)
        dx (float-array dx0) dy (float-array dy0)
        args (arguments prepared {'n (long 3) 'x xs 'q qs 'scale scale})
        [output pullback] ((:vjp prepared) args)
        [jvp-output tangent] ((:jvp prepared) args dx)
        forward (fn [x q]
                  (let [a (float (* x x))
                        b (float (* a a))
                        c (float (* b scale))]
                    [a b c (float (* c q))]))
        tangent-oracle
        (fn [x q d]
          (let [[a] (forward x q)
                da (float (* (float (* (float 2.0) x)) d))
                db (float (* (float (* (float 2.0) a)) da))
                dc (float (* db scale))]
            (float (* dc q))))
        gradient-oracle
        (fn [x q seed]
          (let [[a] (forward x q)
                dc (float (* q seed))
                db (float (* scale dc))
                da (float (* (float (* (float 2.0) a)) db))]
            (float (* (float (* (float 2.0) x)) da))))
        primal (mapv (fn [x q] (last (forward x q))) xs0 qs0)
        expected-tangent (mapv tangent-oracle xs0 qs0 dx0)
        expected-gradient (mapv gradient-oracle xs0 qs0 dy0)
        source-equations (mapv second (soac/equations source))]
    (is (= 4 (count source-equations)))
    (is (= 1 (count (soac/equations (:execution-source owner)))))
    (is (= (set (keys (:residual-bindings owner)))
           (set (concat (keys (:owned-arrays residual-plan))
                        (keys (:immutable-scalars residual-plan))))))
    (is (seq (:owned-arrays residual-plan)))
    (is (seq (:immutable-scalars residual-plan)))
    (doseq [[coefficient binding] (:residual-bindings owner)]
      (is (identical? binding
                      (get (case (:storage binding)
                             :owned-array (:owned-arrays residual-plan)
                             :immutable-scalar (:immutable-scalars residual-plan))
                           coefficient))))
    (is (= (vec (reverse (:inputs (soac/facts source)))) (:inputs prepared)))
    (is (some #(= :immutable-scalar (:storage %)) (vals (:residual-bindings owner))))
    (is (every? #(= source-equations (get-in % [:origin :source-equations]))
                (vals (:residual-bindings owner))))
    (is (= (mapv fbits primal) (mapv fbits output) (mapv fbits jvp-output)))
    (is (= (mapv fbits expected-tangent) (mapv fbits tangent)))
    (dotimes [i 3]
      (aset xs i (float 99.0))
      (aset qs i (float 88.0))
      (aset output i (float -77.0)))
    (let [slots (pullback dy)
          x-slot (.indexOf ^java.util.List (:inputs prepared) 'x)
          q-slot (.indexOf ^java.util.List (:inputs prepared) 'q)]
      (is (= (mapv fbits expected-gradient) (mapv fbits (nth slots x-slot))))
      (is (nil? (nth slots q-slot))))
    (is (= :ad-program-bundle-correspondence
           (reason #(canonical/validate!
                     (recertify
                      (update-in owner
                                 [:residual-bindings (first (keys (:residual-bindings owner)))
                                  :origin :source-equations]
                                 reverse))))))))

(deftest consumer-scalar-capture-keeps-its-outer-input-owner
  (let [checked (evidence (scalar-capture-chain))
        owner (canonical/adapt-map checked)
        scalar-bindings (filter #(= :immutable-scalar (:storage (val %)))
                                (:residual-bindings owner))
        [[coefficient binding]] scalar-bindings
        certified-scalars (:scalar-values (normalized/execution-map-scope checked))
        prepared (runtime/prepare-owner owner)
        xs0 (mapv float [4097.0 -3.0])
        xs (float-array xs0)
        args (arguments prepared {'n (long 2) 'x xs 'scale (float 0.25)})
        dx (float-array [0.5 -2.0])
        dy (float-array [-0.5 1.25])
        [output pullback] ((:vjp prepared) args)
        [_ tangent] ((:jvp prepared) args dx)
        gradient (nth (pullback dy) (.indexOf ^java.util.List (:inputs prepared) 'x))
        expected-output (mapv #(float (* (float (* % %)) (float 0.25))) xs0)
        expected-tangent
        (mapv (fn [x d]
                (float (* (float (* (float (* (float 2.0) x)) d)) (float 0.25))))
              xs0 (mapv float [0.5 -2.0]))
        expected-gradient
        (mapv (fn [x seed]
                (let [consumer-pull (float (* (float 0.25) seed))]
                  (float (* (float (* (float 2.0) x)) consumer-pull))))
              xs0 (mapv float [-0.5 1.25]))]
    (is (= 1 (count scalar-bindings)))
    (is (= 'float (:raster.type/tag (meta coefficient))))
    (is (#'canonical/exact-scalar-member? certified-scalars coefficient))
    (is (= 'scale (:id binding) (get-in binding [:producer :input])))
    (is (= [] (get-in binding [:value :shape])))
    (is (= (mapv second (soac/equations (:source owner)))
           (get-in binding [:origin :source-equations])))
    (is (= (second (first (soac/equations (:execution-source owner))))
           (get-in binding [:origin :execution-equation])))
    (is (not-any? #(= 'scale (:id %))
                  (filter #(= :owned-array (:storage %))
                          (vals (:residual-bindings owner)))))
    (is (= (mapv fbits expected-output) (mapv fbits output)))
    (is (= (mapv fbits expected-tangent) (mapv fbits tangent)))
    (is (= (mapv fbits expected-gradient) (mapv fbits gradient)))
    (let [captures (get-in checked [:map-programs :interface :scalar-captures])
          changed-key (with-meta coefficient
                        (assoc (meta coefficient) :raster.type/tag 'double))
          changed (assoc-in checked [:map-programs :interface :scalar-captures]
                            {changed-key (get captures coefficient)})
          changed (assoc changed :certificate (#'normalized/certificate changed))]
      (is (some? (reason #(normalized/validate! changed)))
          "recertification cannot alter the typed residual identity"))))

(deftest zero-extent-executes-without-inventing-residual-values
  (let [prepared (runtime/prepare (evidence (typed-chain 0)))
        args (arguments prepared {'x (float-array 0) 'q (float-array 0)})
        [output pullback] ((:vjp prepared) args)
        [jvp-output tangent] ((:jvp prepared) args (float-array 0))
        slots (pullback (float-array 0))]
    (is (= [] (vec output) (vec jvp-output) (vec tangent)))
    (is (= [] (vec (nth slots (.indexOf ^java.util.List (:inputs prepared) 'x)))))
    (is (nil? (nth slots (.indexOf ^java.util.List (:inputs prepared) 'q))))))

(deftest finite-spine-zero-extent-retains-empty-primal-jvp-and-vjp
  (let [prepared (runtime/prepare (evidence (four-map-spine 0)))
        args (arguments prepared {'x (float-array 0) 'q (float-array 0) 'scale (float 0.25)})
        [output pullback] ((:vjp prepared) args)
        [jvp-output tangent] ((:jvp prepared) args (float-array 0))
        slots (pullback (float-array 0))]
    (is (= [] (vec output) (vec jvp-output) (vec tangent)))
    (is (= [] (vec (nth slots (.indexOf ^java.util.List (:inputs prepared) 'x)))))
    (is (nil? (nth slots (.indexOf ^java.util.List (:inputs prepared) 'q))))))

(deftest canonical-reconstruction-rejects-every-region-scope-tamper
  (let [owner (canonical/adapt-map (evidence (typed-diamond 'n)))
        coefficient (first (keys (:residual-bindings owner)))
        changes [#(update % :source alter-facts
                          (fn [facts] (assoc-in facts [:values 'x :dtype] :double)))
                 #(update % :execution-source alter-facts
                          (fn [facts] (assoc-in facts [:values 'x :dtype] :double)))
                 #(update-in % [:residual-bindings coefficient :origin :source-equations] reverse)
                 #(assoc-in % [:residual-bindings coefficient :origin :execution-equation] 'forged)
                 #(assoc-in % [:interface :inputs 0 :value :dtype] :double)
                 #(update % :primal alter-facts
                          (fn [facts] (assoc-in facts [:values 'x :dtype] :double)))]]
    (doseq [change changes]
      (is (= :ad-program-bundle-correspondence
             (reason #(canonical/validate! (recertify (change owner)))))))))
