(ns raster.compiler.passes.parallel.soac-graph-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-graph :as sg]
            [raster.par :as par]))

;; ================================================================
;; Helper
;; ================================================================

(defn- make-graph
  "Build a fusion graph from let* binding pairs."
  [pairs]
  (sg/build-fusion-graph (soac/let-bindings->nodes pairs)))

;; ================================================================
;; Graph construction
;; ================================================================

(deftest build-graph-simple-chain-test
  (testing "Simple chain A→B→C"
    (let [pairs [['a '(raster.par/map! a i n double (* (aget x i) 2.0))]
                 ['b '(raster.par/map! b i n double (+ (aget a i) 1.0))]
                 ['c '(raster.par/map! c i n double (* (aget b i) 3.0))]]
          graph (make-graph pairs)]
      (is (= 3 (count (:nodes graph))))
      ;; Check edges: a→b and b→c should be :dep
      (is (some (fn [[from to typ]]
                  (and (= from 0) (= to 1) (= typ :dep)))
                (:edges graph)))
      (is (some (fn [[from to typ]]
                  (and (= from 1) (= to 2) (= typ :dep)))
                (:edges graph))))))

(deftest build-graph-diamond-test
  (testing "Diamond: A→B, A→C, B→D, C→D"
    (let [pairs [['a '(raster.par/map! a i n double (aget x i))]
                 ['b '(raster.par/map! b i n double (* (aget a i) 2.0))]
                 ['c '(raster.par/map! c i n double (* (aget a i) 3.0))]
                 ['d '(raster.par/map! d i n double (+ (aget b i) (aget c i)))]]
          graph (make-graph pairs)]
      (is (= 4 (count (:nodes graph))))
      ;; A produces for both B and C
      (let [a-consumers (filter (fn [[from _ _]] (= from 0)) (:edges graph))]
        (is (= 2 (count a-consumers)))))))

(deftest build-graph-independent-test
  (testing "Independent maps (no edges)"
    (let [pairs [['a '(raster.par/map! a i n double (aget x i))]
                 ['b '(raster.par/map! b i n double (aget y i))]]
          graph (make-graph pairs)]
      ;; No dep edges between a and b
      (is (empty? (filter (fn [[from to typ]]
                            (and (= typ :dep)
                                 (#{#{0 1}} #{from to})))
                          (:edges graph)))))))

(deftest build-graph-scalar-edge-test
  (testing "Scalar binding creates :inf-dep edge"
    (let [pairs [['s '42]
                 ['a '(raster.par/map! a i n double (* (aget x i) s))]]
          graph (make-graph pairs)]
      (is (some (fn [[from to typ]]
                  (and (= from 0) (= to 1) (= typ :inf-dep)))
                (:edges graph))))))

(deftest build-graph-reduce-edge-test
  (testing "Reduce output creates :inf-dep edge (scalar, not array)"
    (let [pairs [['s '(raster.par/reduce acc 0.0 i n (+ acc (aget x i)))]
                 ['out '(raster.par/map! out i n double (* (aget y i) s))]]
          graph (make-graph pairs)]
      ;; Reduce→Map edge must be :inf-dep (scalar dependency)
      (is (some (fn [[from to typ]]
                  (and (= from 0) (= to 1) (= typ :inf-dep)))
                (:edges graph)))
      ;; No :dep edges from reduce to map
      (is (not (some (fn [[from to typ]]
                       (and (= from 0) (= to 1) (= typ :dep)))
                     (:edges graph)))))))

;; ================================================================
;; Reachability
;; ================================================================

(deftest reachable-transitive-test
  (testing "Transitive reachability through chain"
    (let [pairs [['a '(raster.par/map! a i n double (aget x i))]
                 ['b '(raster.par/map! b i n double (aget a i))]
                 ['c '(raster.par/map! c i n double (aget b i))]]
          graph (make-graph pairs)]
      (is (sg/reachable? graph 0 2))
      (is (sg/reachable? graph 0 1))
      (is (not (sg/reachable? graph 2 0))))))

(deftest reachable-excluding-test
  (testing "Reachability excluding intermediate node"
    (let [pairs [['a '(raster.par/map! a i n double (aget x i))]
                 ['b '(raster.par/map! b i n double (aget a i))]
                 ['c '(raster.par/map! c i n double (aget b i))]]
          graph (make-graph pairs)]
      ;; 0→2 goes through 1; excluding 1 blocks it
      (is (not (sg/reachable-excluding? graph 0 2 #{1}))))))

;; ================================================================
;; Vertical fusion
;; ================================================================

(deftest can-fuse-vertically-map-map-test
  (testing "Map→Map vertical fusion is legal"
    (let [pairs [['tmp '(raster.par/map! tmp i n double (* (aget a i) 2.0))]
                 ['out '(raster.par/map! out i n double (+ (aget tmp i) 1.0))]]
          graph (make-graph pairs)]
      (is (sg/can-fuse-vertically? graph 0 1)))))

(deftest cannot-fuse-different-bounds-test
  (testing "Cannot fuse maps with different bounds"
    (let [pairs [['tmp '(raster.par/map! tmp i n double (aget a i))]
                 ['out '(raster.par/map! out i m double (aget tmp i))]]
          graph (make-graph pairs)]
      (is (not (sg/can-fuse-vertically? graph 0 1))))))

(deftest fuse-vertical-map-map-test
  (testing "Fuse map→map: body inlined"
    (let [pairs [['tmp '(raster.par/map! tmp i n double (* (aget a i) 2.0))]
                 ['out '(raster.par/map! out i n double (+ (aget tmp i) 1.0))]]
          graph (make-graph pairs)
          fused (sg/fuse-vertical graph 0 1)]
      ;; Only 1 node remains
      (is (= 1 (count (:nodes fused))))
      ;; The remaining node should have 'a' as input, not 'tmp'
      (let [node (first (vals (:nodes fused)))]
        (is (contains? (:inputs node) 'a))
        (is (not (contains? (:inputs node) 'tmp)))))))

(deftest can-fuse-vertically-map-reduce-test
  (testing "Map→Reduce vertical fusion is legal"
    (let [pairs [['tmp '(raster.par/map! tmp i n double (* (aget a i) (aget a i)))]
                 ['result '(raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))]]
          graph (make-graph pairs)]
      (is (sg/can-fuse-vertically? graph 0 1)))))

(deftest can-fuse-vertically-scan-map-test
  (testing "Scan→Map vertical fusion is legal"
    (let [pairs [['prefix '(raster.par/scan prefix acc 0.0 i n double (+ acc (aget a i)))]
                 ['out '(raster.par/map! out i n double (* (aget prefix i) 2.0))]]
          graph (make-graph pairs)]
      (is (sg/can-fuse-vertically? graph 0 1)))))

(deftest cannot-fuse-scan-reduce-test
  (testing "Scan→Reduce vertical fusion is not supported (consumer must be Map)"
    (let [pairs [['prefix '(raster.par/scan prefix acc 0.0 i n double (+ acc (aget a i)))]
                 ['result '(raster.par/reduce acc2 0.0 j n (+ acc2 (aget prefix j)))]]
          graph (make-graph pairs)]
      (is (not (sg/can-fuse-vertically? graph 0 1))))))

(deftest fuse-vertical-scan-map-test
  (testing "Fuse scan→map: map body folded into scan loop"
    (let [pairs [['prefix '(raster.par/scan prefix acc 0.0 i n double (+ acc (aget a i)))]
                 ['out '(raster.par/map! out i n double (* (aget prefix i) 2.0))]]
          graph (make-graph pairs)
          fused (sg/fuse-vertical graph 0 1)]
      ;; 2 nodes: fused SoacScan + ScalarBinding alias
      (is (= 2 (count (:nodes fused))))
      ;; The SoacScan should remain with a do-block body
      (let [scan-node (first (filter #(instance? raster.compiler.ir.soac.SoacScan %)
                                     (vals (:nodes fused))))]
        (is (some? scan-node))
        (is (= 'do (first (:lambda scan-node))))
        ;; Outputs include both prefix and out
        (is (contains? (:outputs scan-node) 'prefix))
        (is (contains? (:outputs scan-node) 'out))))))

(deftest fixpoint-scan-map-test
  (testing "Scan→Map fusion through fixpoint"
    (let [pairs [['prefix '(raster.par/scan prefix acc 0.0 i n double (+ acc (aget x i)))]
                 ['scaled '(raster.par/map! scaled i n double (* (aget prefix i) 2.0))]]
          graph (make-graph pairs)
          [fused stats] (sg/fusion-fixpoint graph)]
      ;; Map eliminated, scan with side-effect remains + alias
      (is (= 2 (count (:nodes fused))))
      (is (= 1 (:vertical stats))))))

(deftest multi-consumer-fusion-test
  (testing "Multi-consumer vertical fusion: producer stays alive"
    (let [pairs [['tmp '(raster.par/map! tmp i n double (* (aget a i) 2.0))]
                 ['out1 '(raster.par/map! out1 i n double (aget tmp i))]
                 ['out2 '(raster.par/map! out2 i n double (+ (aget tmp i) 1.0))]]
          graph (make-graph pairs)]
      ;; tmp has two consumers — both can now be fused
      (is (sg/can-fuse-vertically? graph 0 1))
      (is (sg/can-fuse-vertically? graph 0 2))
      ;; After fusing tmp→out1, producer stays alive for out2
      (let [fused (sg/fuse-vertical graph 0 1)]
        ;; All 3 nodes still present (producer kept alive)
        (is (= 3 (count (:nodes fused))))
        ;; out1's body now uses 'a' directly
        (let [out1-node (get (:nodes fused) 1)]
          (is (contains? (:inputs out1-node) 'a)))))))

;; ================================================================
;; Horizontal fusion
;; ================================================================

(deftest find-horizontal-groups-test
  (testing "Find independent same-bound map groups"
    (let [pairs [['out1 '(raster.par/map! out1 i n double (aget a i))]
                 ['out2 '(raster.par/map! out2 i n double (aget b i))]]
          graph (make-graph pairs)
          groups (sg/find-horizontal-groups graph)]
      (is (= 1 (count groups)))
      (is (= 2 (count (first groups)))))))

(deftest no-horizontal-fusion-dependent-test
  (testing "No horizontal fusion when maps are dependent"
    (let [pairs [['out1 '(raster.par/map! out1 i n double (aget a i))]
                 ['out2 '(raster.par/map! out2 i n double (aget out1 i))]]
          graph (make-graph pairs)
          groups (sg/find-horizontal-groups graph)]
      (is (empty? groups)))))

(deftest no-horizontal-fusion-late-dependency-test
  (testing "No horizontal fusion when secondary has data dependency after primary"
    ;; Simulates the real GSDM bug: secondary reads data produced between
    ;; primary and secondary positions.  Fusing would push the fused map
    ;; past consumers of the primary's output.
    (let [pairs [['out1    '(raster.par/map! out1 i n double (aget a i))]
                 ['mid     '(raster.par/map! mid  j n double (* (aget out1 j) 2.0))]
                 ['out2    '(raster.par/map! out2 i n double (+ (aget b i) (aget mid i)))]]
          graph (make-graph pairs)
          groups (sg/find-horizontal-groups graph)]
      ;; out1 and out2 have the same bound, but out2 depends on mid which
      ;; depends on out1 — so out1→out2 are transitively dependent.
      ;; And even without direct reachability through the graph edges,
      ;; out2's data dependency on 'mid' (produced at node 1, after primary 0)
      ;; should block fusion.
      (is (empty? groups)))))

(deftest fuse-horizontal-test
  (testing "Horizontal fusion merges into do block"
    (let [pairs [['out1 '(raster.par/map! out1 i n double (Math/sin (aget a i)))]
                 ['out2 '(raster.par/map! out2 i n double (Math/cos (aget a i)))]]
          graph (make-graph pairs)
          groups (sg/find-horizontal-groups graph)
          fused (sg/fuse-horizontal graph (first groups))]
      ;; 2 nodes: fused SoacMap + ScalarBinding alias for secondary sym
      (is (= 2 (count (:nodes fused))))
      ;; Lambda of the SoacMap should be a do block
      (let [soac-node (first (filter #(instance? raster.compiler.ir.soac.SoacMap %)
                                     (vals (:nodes fused))))]
        (is (= 'do (first (:lambda soac-node))))))))

(deftest multi-consumer-fixpoint-test
  (testing "Multi-consumer: ev→result fuses, ev stays alive for reduce"
    (let [pairs [['ev     '(raster.par/map! ev i n double (Math/exp (aget shifted i)))]
                 ['s      '(raster.par/reduce acc 0.0 i n (+ acc (aget ev i)))]
                 ['result '(raster.par/map! result i n double (/ (aget ev i) s))]]
          graph (make-graph pairs)
          [fused stats] (sg/fusion-fixpoint graph)]
      ;; ev→result fused: result now computes exp(shifted[i])/s directly
      (is (pos? (:vertical stats)) "Should have at least one vertical fusion")
      ;; ev stays alive (needed by reduce s)
      (is (some #(instance? raster.compiler.ir.soac.SoacMap %)
                (vals (:nodes fused)))
          "At least one SoacMap should remain"))))

;; ================================================================
;; Fixpoint fusion
;; ================================================================

(deftest fixpoint-chain-of-3-maps-test
  (testing "Chain of 3 maps fuses to single map"
    (let [pairs [['t1 '(raster.par/map! t1 i n double (* (aget a i) 2.0))]
                 ['t2 '(raster.par/map! t2 i n double (+ (aget t1 i) 1.0))]
                 ['out '(raster.par/map! out i n double (* (aget t2 i) 3.0))]]
          graph (make-graph pairs)
          [fused stats] (sg/fusion-fixpoint graph)]
      (is (= 1 (count (:nodes fused))))
      (is (= 2 (:vertical stats))))))

(deftest fixpoint-no-fusion-test
  (testing "No fusion when nothing is fusible"
    (let [pairs [['out1 '(raster.par/map! out1 i n double (aget a i))]
                 ['out2 '(raster.par/map! out2 i m double (aget b i))]]
          graph (make-graph pairs)
          [fused stats] (sg/fusion-fixpoint graph)]
      ;; Nothing fused — different bounds, can't fuse
      (is (= 2 (count (:nodes fused))))
      (is (= 0 (:vertical stats)))
      (is (= 0 (:horizontal stats))))))

(deftest fixpoint-mixed-vertical-horizontal-test
  (testing "Mixed vertical + horizontal fusion"
    (let [pairs [['t1 '(raster.par/map! t1 i n double (* (aget a i) 2.0))]
                 ['out1 '(raster.par/map! out1 i n double (aget t1 i))]
                 ['out2 '(raster.par/map! out2 i n double (aget b i))]]
          graph (make-graph pairs)
          [fused stats] (sg/fusion-fixpoint graph)]
      ;; t1→out1 vertical fusion, then out1+out2 horizontal
      (is (<= (count (:nodes fused)) 2))
      (is (pos? (:vertical stats))))))

(deftest fixpoint-map-reduce-test
  (testing "Map→Reduce fusion through fixpoint"
    (let [pairs [['tmp '(raster.par/map! tmp i n double (* (aget a i) (aget a i)))]
                 ['result '(raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))]]
          graph (make-graph pairs)
          [fused stats] (sg/fusion-fixpoint graph)]
      ;; Map eliminated, only reduce remains
      (is (= 1 (count (:nodes fused))))
      (is (= 1 (:vertical stats))))))

;; ================================================================
;; nodes->let-bindings after fusion preserves correctness
;; ================================================================

(deftest fused-graph-to-bindings-test
  (testing "Fused graph converts back to valid let* bindings"
    (let [pairs [['tmp '(raster.par/map! tmp i n double (* (aget a i) 2.0))]
                 ['out '(raster.par/map! out i n double (+ (aget tmp i) 1.0))]]
          graph (make-graph pairs)
          [fused _] (sg/fusion-fixpoint graph)
          bindings (soac/nodes->let-bindings (:nodes fused))]
      (is (= 1 (count bindings)))
      (let [[sym expr] (first bindings)]
        (is (= 'out sym))
        (is (= 'raster.par/map! (first expr)))))))

;; ================================================================
;; Scalar-bound normalization (dense+relu pattern)
;; ================================================================

(deftest scalar-bound-fusion-test
  (testing "Maps with scalar-bound allocation fuse (dense+relu pattern)"
    ;; Simulates: dense produces via (double-array (long rows)), relu reads (alength h)
    ;; The bounds are different syntactically but equivalent semantically.
    (let [pairs [['rows '(.invk raster.arrays/alength_m_doubles-impl b)]
                 ['buf  '(clojure.core/double-array (long rows))]
                 ['h    '(raster.par/map! buf i rows double
                                          (+ (aget b i) (raster.par/reduce acc 0.0 j cols
                                                                           (+ acc (* (aget W (+ (* i cols) j)) (aget x j))))))]
                 ['buf2 '(clojure.core/double-array (.invk raster.arrays/alength_m_doubles-impl h))]
                 ['a    '(raster.par/map! buf2 k (.invk raster.arrays/alength_m_doubles-impl h) double
                                          (max 0.0 (aget h k)))]]
          graph (make-graph pairs)]
      ;; h and a should be fusible (same bound after normalization)
      (is (sg/can-fuse-vertically? graph 2 4)
          "dense→relu should be fusible via scalar-bound normalization")
      ;; Fusion through fixpoint
      (let [[fused stats] (sg/fusion-fixpoint graph)]
        (is (pos? (:vertical stats)) "Should have at least one vertical fusion")
        ;; After fusion, relu's body should contain the dense body (not aget h)
        (let [a-node (some #(when (= 'a (:sym %)) %) (vals (:nodes fused)))]
          (when a-node
            (is (not (some #{'h} (flatten (seq (:lambda a-node)))))
                "Fused lambda should not read from h anymore")))))))

(deftest alias-substitution-test
  (testing "Vertical fusion substitutes through producer sym alias"
    ;; Producer: h = (par/map! out_buf ...), consumer reads (aget h ...)
    ;; The output buffer is out_buf, but consumer uses the alias h.
    (let [pairs [['out_buf '(clojure.core/double-array n)]
                 ['h       '(raster.par/map! out_buf i n double (* (aget a i) 2.0))]
                 ['result  '(raster.par/map! result j n double (+ (aget h j) 1.0))]]
          graph (make-graph pairs)
          [fused stats] (sg/fusion-fixpoint graph)]
      ;; h→result should fuse
      (is (pos? (:vertical stats)))
      ;; result's lambda should reference 'a' directly
      (let [r-node (some #(when (= 'result (:sym %)) %) (vals (:nodes fused)))]
        (when r-node
          (is (contains? (:inputs r-node) 'a)
              "Fused result should read from 'a' directly"))))))
