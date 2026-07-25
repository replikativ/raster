(ns raster.compiler.backend.gpu.staged-contract-device-test
  "STAGED contraction on device: a reduction that accumulates in more than one level, which is
   the shape every block-quantized format needs and the flat leaves cannot express.

       out[i,j] = Σ_blk  da[i,blk]·db[j,blk] · ( Σ_t  a[i,blk,t] · b[j,blk,t] )
                         ^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         float, once per block   int32, exact, 32 MACs

   This is q8_0/q4_0 (2 stages) and, with one more level, k-quants (3 stages). The scales are
   indexed [i,blk] / [j,blk] — a free axis AND a stage axis — which is the real format layout and
   exercises the declared axis-maps rather than a per-tensor scalar.

   THE ORACLE IS THE FLAT EQUIVALENT. Because a lift is linear in the inner accumulator, staging
   is a SCHEDULE: the staged contraction equals the flat contraction whose body absorbed the lift
   factors (ir/contract-stages/flat-equivalent). So each case is checked against BOTH a direct
   block reference and the flat form — and with power-of-two scales the identity is exact in
   float, so it is asserted with `=`, not a tolerance. Gated on a real GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.contract-stages :as cs]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.backend.gpu.segop-opencl :as sco]))

(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(def ^:private M 4)
(def ^:private N 6)
(def ^:private B 32)                  ; block length (q8_0's block)
(def ^:private NB 4)                  ; blocks per row
(def ^:private K (* NB B))            ; 128

;; operands are K-contiguous (the :nt layout block-quant formats use)
(defn- a-idx [] (list 'clojure.core/+ (list 'clojure.core/* 'i K)
                      (list 'clojure.core/+ (list 'clojure.core/* 'blk B) 't)))
(defn- b-idx [] (list 'clojure.core/+ (list 'clojure.core/* 'j K)
                      (list 'clojure.core/+ (list 'clojure.core/* 'blk B) 't)))
(def ^:private body
  (list 'raster.numeric/* (list 'aget 'a (a-idx)) (list 'aget 'b (b-idx))))

(def ^:private two-stage
  ;; scale maps: da[i,blk] → i*NB + blk ; db[j,blk] → j*NB + blk (declared, not inferred)
  [{:axis 'blk :extent NB :dtype :float :init 0.0
    :lift '(raster.numeric/* inner (aget da _) (aget db _))
    :operands [{:sym 'da :map {:groups [[['i M]] [['blk NB]]]} :dtype :float}
               {:sym 'db :map {:groups [[['j N]] [['blk NB]]]} :dtype :float}]}
   {:axis 't :extent B :dtype :int :init 0}])

;; ── host references ─────────────────────────────────────────────────────────────────
(defn- ref-staged
  "The block formula, evaluated exactly as the kernel schedules it: int32 inner, float outer."
  [^bytes a ^bytes b ^floats da ^floats db]
  (let [out (float-array (* M N))]
    (dotimes [i M]
      (dotimes [j N]
        (let [acc (loop [blk 0 acc 0.0]
                    (if (< blk NB)
                      (let [inner (loop [t 0 s 0]
                                    (if (< t B)
                                      (recur (inc t)
                                             (+ s (* (int (aget a (+ (* i K) (* blk B) t)))
                                                     (int (aget b (+ (* j K) (* blk B) t))))))
                                      s))]
                        (recur (inc blk)
                               (+ acc (* (double inner)
                                         (double (aget da (+ (* i NB) blk)))
                                         (double (aget db (+ (* j NB) blk)))))))
                      acc))]
          (aset out (+ (* i N) j) (float acc)))))
    out))

(defn- ref-flat
  "The FLAT equivalent: one reduction over l, lift factors pushed into the body — i.e. what
   `flat-equivalent` derives, and what a CPU/interpreter path would run."
  [^bytes a ^bytes b ^floats da ^floats db]
  (let [out (float-array (* M N))]
    (dotimes [i M]
      (dotimes [j N]
        (let [acc (loop [l 0 acc 0.0]
                    (if (< l K)
                      (let [blk (quot l B)]
                        (recur (inc l)
                               (+ acc (* (double (int (aget a (+ (* i K) l))))
                                         (double (int (aget b (+ (* j K) l))))
                                         (double (aget da (+ (* i NB) blk)))
                                         (double (aget db (+ (* j NB) blk)))))))
                      acc))]
          (aset out (+ (* i N) j) (float acc)))))
    out))

;; ── device run ──────────────────────────────────────────────────────────────────────
(defn- run-staged
  "Emit, register and launch a staged contraction; return the device output vector. `lift-arrays`
   maps each lift operand symbol → its float array, bound in the emitter's declared order."
  [stages body-expr ^bytes a ^bytes b lift-arrays & [extra]]
  (let [ze (find-ns 'raster.gpu.ze-runtime)
        register! (ns-resolve ze 'register-kernel!)
        ensure-loaded! (ns-resolve ze 'ensure-kernel-loaded!)
        make-buffer (ns-resolve ze 'make-buffer)
        arr->buf! (ns-resolve ze 'array->buffer!)
        buf->floats (ns-resolve ze 'buffer->float-array)
        launch-2d! (ns-resolve ze 'launch-2d!)
        {:keys [kernel-name source array-params lift-operands out-elems]}
        (sco/generate-staged-contraction-kernel
         (merge {:free-axes [['i M] ['j N]] :stages stages :body body-expr
                 :inputs '[a b] :dtype :byte :out-dtype :float}
                extra)
         'out)
        _ (assert (= '[a b] array-params))
        _ (register! kernel-name {:source source :dtype :byte})
        {:keys [kernel-handle]} (ensure-loaded! kernel-name)
        abuf (arr->buf! (make-buffer (alength a) :byte) a)
        bbuf (arr->buf! (make-buffer (alength b) :byte) b)
        lbufs (mapv (fn [sym] (let [^floats arr (get lift-arrays sym)]
                                (arr->buf! (make-buffer (alength arr) :float) arr)))
                    lift-operands)
        obuf (make-buffer out-elems :float)
        nseg (long out-elems)
        gx (long (Math/ceil (/ (double nseg) 256.0)))
        args (concat [(:segment abuf) (:segment bbuf)]
                     (map :segment lbufs)
                     [(:segment obuf) {:type :int :value (int nseg)}])]
    (launch-2d! kernel-handle [256 1] [gx 1] (vec args))
    (vec (buf->floats obuf))))

(defn- pow2-scales [n seed]
  ;; powers of two → every float multiply is exact, so staged and flat agree BIT-for-bit
  (float-array (map (fn [i] (Math/scalb 1.0 (int (- (int (mod (+ i seed) 5)) 2)))) (range n))))

(def ^:private a-data (byte-array (map #(byte (- (mod (* 7 %) 15) 7)) (range (* M K)))))
(def ^:private b-data (byte-array (map #(byte (- (mod (* 5 %) 15) 7)) (range (* N K)))))

(deftest two-stage-block-quant-on-device
  (if-not @gpu?
    (println "  [skip] no GPU — staged contraction device test")
    (let [da (pow2-scales (* M NB) 0)
          db (pow2-scales (* N NB) 3)
          gpu (run-staged two-stage body a-data b-data {'da da 'db db})
          staged (vec (ref-staged a-data b-data da db))
          flat (vec (ref-flat a-data b-data da db))]
      (testing "flat-equivalent derives exactly the flat body ref-flat evaluates by hand"
        (is (= (list 'raster.numeric/* body
                     '(aget da (clojure.core/+ (clojure.core/* i 4) blk))
                     '(aget db (clojure.core/+ (clojure.core/* j 4) blk)))
               (cs/flat-equivalent two-stage body))
            "and the placeholder index never escapes — it is replaced by the declared map"))
      (testing "the staged/flat identity holds EXACTLY on real data (power-of-two scales)"
        (is (= staged flat)
            "staging is a schedule: it must not change the value, only the accumulator"))
      (testing "device matches the reference"
        (is (= staged gpu)))
      (testing "the result is actually non-trivial (guards against an all-zero pass)"
        (is (some #(not (zero? %)) gpu))))))

(deftest three-stage-super-block-on-device
  (if-not @gpu?
    (println "  [skip] no GPU — 3-stage staged contraction device test")
    ;; k-quant nesting: super-block scale × sub-block scale × int MAC. NB is split 4 = 2 × 2.
    (let [nsb 2 nsub 2
          stages [{:axis 'sb :extent nsb :dtype :float :init 0.0
                   :lift '(raster.numeric/* inner (aget dsuper _))
                   :operands [{:sym 'dsuper :map {:groups [[['i M]] [['sb nsb]]]} :dtype :float}]}
                  {:axis 'blk2 :extent nsub :dtype :float :init 0.0
                   :lift '(raster.numeric/* inner (aget dsub _))
                   :operands [{:sym 'dsub :map {:groups [[['j N]] [['blk2 nsub]]]} :dtype :float}]}
                  {:axis 't :extent B :dtype :int :init 0}]
          ;; l = ((sb*nsub) + blk2)*B + t
          l-expr (list 'clojure.core/+
                       (list 'clojure.core/* (list 'clojure.core/+
                                                   (list 'clojure.core/* 'sb nsub) 'blk2) B)
                       't)
          body3 (list 'raster.numeric/*
                      (list 'aget 'a (list 'clojure.core/+ (list 'clojure.core/* 'i K) l-expr))
                      (list 'aget 'b (list 'clojure.core/+ (list 'clojure.core/* 'j K) l-expr)))
          dsuper (pow2-scales (* M nsb) 1)
          dsub (pow2-scales (* N nsub) 2)
          gpu (run-staged stages body3 a-data b-data {'dsuper dsuper 'dsub dsub})
          ;; host reference for the 3-level nest
          ref (let [out (float-array (* M N))]
                (dotimes [i M]
                  (dotimes [j N]
                    (let [acc (loop [sb 0 acc 0.0]
                                (if (< sb nsb)
                                  (let [mid (loop [blk2 0 m 0.0]
                                              (if (< blk2 nsub)
                                                (let [inner (loop [t 0 s 0]
                                                              (if (< t B)
                                                                (let [l (+ (* (+ (* sb nsub) blk2) B) t)]
                                                                  (recur (inc t)
                                                                         (+ s (* (int (aget a-data (+ (* i K) l)))
                                                                                 (int (aget b-data (+ (* j K) l)))))))
                                                                s))]
                                                  (recur (inc blk2)
                                                         (+ m (* (double inner)
                                                                 (double (aget dsub (+ (* j nsub) blk2)))))))
                                                m))]
                                    (recur (inc sb)
                                           (+ acc (* mid (double (aget dsuper (+ (* i nsb) sb)))))))
                                  acc))]
                      (aset out (+ (* i N) j) (float acc)))))
                (vec out))]
      (testing "3 stages (k-quant nesting) is the same mechanism, one level deeper"
        (is (= ref gpu)))
      (testing "non-trivial result"
        (is (some #(not (zero? %)) gpu))))))

(deftest one-stage-is-the-flat-contraction
  (testing "a single stage needs no lift and emits a plain accumulate loop (no nesting)"
    (let [{:keys [source lift-operands]}
          (sco/generate-staged-contraction-kernel
           {:free-axes [['i M] ['j N]]
            :stages [{:axis 'l :extent K :dtype :float :init 0.0}]
            :body (list 'raster.numeric/*
                        (list 'aget 'a (list 'clojure.core/+ (list 'clojure.core/* 'i K) 'l))
                        (list 'aget 'b (list 'clojure.core/+ (list 'clojure.core/* 'j K) 'l)))
            :inputs '[a b] :dtype :float :out-dtype :float}
           'out)]
      (is (= [] lift-operands) "no scale arrays, so no extra kernel params")
      (is (= 1 (count (re-seq #"for \(int" source))) "exactly one loop level")
      (is (not (clojure.string/includes? source "acc_1"))))))

(deftest illegal-stages-are-refused-not-emitted
  (testing "the emitter refuses an illegal stage list rather than emitting a wrong kernel"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"lift-discards-inner-accumulator"
         (sco/generate-staged-contraction-kernel
          {:free-axes [['i M] ['j N]]
           :stages [{:axis 'blk :extent NB :dtype :float :init 0.0
                     :lift '(raster.numeric/* (aget da _) 2.0)
                     :operands [{:sym 'da :map {:groups [[['blk NB]]]}}]}
                    {:axis 't :extent B :dtype :int :init 0}]
           :body body :inputs '[a b] :dtype :byte :out-dtype :float}
          'out)))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"lift-not-linear-in-inner"
         (sco/generate-staged-contraction-kernel
          {:free-axes [['i M] ['j N]]
           :stages [{:axis 'blk :extent NB :dtype :float :init 0.0
                     :lift '(raster.numeric/sqrt inner)}
                    {:axis 't :extent B :dtype :int :init 0}]
           :body body :inputs '[a b] :dtype :byte :out-dtype :float}
          'out)))))

;; ── the int8 PEAK leaf: tensorize the inner stage with dp4a ──────────────────────────
(def ^:private body-maps
  ;; a[i,(blk t)] and b[j,(blk t)] — both K-CONTIGUOUS in the inner stage axis, which is what dp4a
  ;; requires (the :nt layout). Declared, then VERIFIED against the body by the gate.
  {'a (am/of-groups [[['i M]] [['blk NB] ['t B]]])
   'b (am/of-groups [[['j N]] [['blk NB] ['t B]]])})

(deftest dp4a-tensorized-inner-stage-matches-the-scalar-nest
  (if-not @gpu?
    (println "  [skip] no GPU — dp4a staged inner-stage device test")
    (let [da (pow2-scales (* M NB) 0)
          db (pow2-scales (* N NB) 3)
          ;; body built FROM the declared maps, so declaration and use cannot drift
          body' (list 'raster.numeric/*
                      (list 'aget 'a (am/index-expr (get body-maps 'a)))
                      (list 'aget 'b (am/index-expr (get body-maps 'b))))
          operands [{:sym 'a :map (get body-maps 'a)} {:sym 'b :map (get body-maps 'b)}]
          scalar (run-staged two-stage body' a-data b-data {'da da 'db db})
          packed (run-staged two-stage body' a-data b-data {'da da 'db db}
                             {:operands operands :tensorize-inner? true})
          ref (vec (ref-staged a-data b-data da db))]
      (testing "dp4a and the scalar nest agree EXACTLY — int32 accumulation is exact, so any
                difference would be a real defect, not rounding"
        (is (= scalar packed)))
      (testing "…and both match the host reference"
        (is (= ref packed)))
      (testing "non-trivial result"
        (is (some #(not (zero? %)) packed))))))

(deftest dp4a-gate-refuses-rather-than-miscompiling
  (let [good-a (get body-maps 'a) good-b (get body-maps 'b)
        body' (list 'raster.numeric/*
                    (list 'aget 'a (am/index-expr good-a))
                    (list 'aget 'b (am/index-expr good-b)))
        spec {:free-axes [['i M] ['j N]] :stages two-stage :body body'
              :inputs '[a b] :dtype :byte :out-dtype :float
              :operands [{:sym 'a :map good-a} {:sym 'b :map good-b}]}]
    (testing "the honest case is legal"
      (is (:ok (sco/staged-inner-dp4a-legal? spec))))
    (testing "a MISDECLARED operand map is caught, not trusted — assuming a layout while having
              checked only the axis symbols is how a transpose rewrite silently miscompiled before"
      (is (= :declared-map-does-not-match-the-body-index
             (:reason (sco/staged-inner-dp4a-legal?
                       (assoc-in spec [:operands 0 :map]
                                 (am/of-groups [[['i M]] [['t B] ['blk NB]]])))))))
    (testing "an operand that is NOT contiguous in the inner stage axis cannot be packed"
      (let [nn-b (am/of-groups [[['blk NB] ['t B]] [['j N]]])]   ; b[(blk t), j] — j innermost
        (is (= :inner-stage-axis-is-not-contiguous
               (:reason (sco/staged-inner-dp4a-legal?
                         (-> spec
                             (assoc-in [:operands 1 :map] nn-b)
                             (assoc :body (list 'raster.numeric/*
                                                (list 'aget 'a (am/index-expr good-a))
                                                (list 'aget 'b (am/index-expr nn-b)))))))))))
    (testing "a float inner accumulator is not a dp4a shape"
      (is (= :inner-stage-accumulator-not-integral
             (:reason (sco/staged-inner-dp4a-legal?
                       (assoc-in spec [:stages 1 :dtype] :float))))))
    (testing "requesting tensorize when illegal THROWS — it never silently degrades to scalar,
              because a silent degradation is an invisible perf cliff"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"cannot tensorize inner stage"
           (sco/generate-staged-contraction-kernel
            (assoc spec :tensorize-inner? true
                   :operands [{:sym 'a :map good-a}]) 'out))))))
