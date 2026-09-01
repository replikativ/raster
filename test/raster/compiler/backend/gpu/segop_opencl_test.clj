(ns raster.compiler.backend.gpu.segop-opencl-test
  "SegRed OpenCL kernel generation — pins #55: aget lowering must NOT be
   ns-sensitive. A parametric-array kernel's reads arrive DEVIRTUALIZED
   ((.invk aget-impl arr i) with :raster.op/original), a typed-array kernel's
   as bare clojure.core/aget — both must lower to the subscript `arr[i]`.
   Before the fix SegRed skipped normalize-array-prims (SegMap didn't) and its
   let*-rewrap dropped the metadata, so the devirtualized shape fell through
   to a broken gpufn_aget helper call (qlinear-k) while the typed shape
   emitted x[i] (decoder-gpu)."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.backend.gpu.segop-opencl :as sg]))

(deftest two-phase-reduction-graph-emits-its-explicit-scheduled-dataflow
  (let [source '(let* [result (raster.par/reduce acc 0.0 i n
                                                 (+ acc (clojure.core/aget a i)))]
                      result)
        options {:dtype :double :array-types {'a :double}
                 :scalar-types {'n :long}}
        typed (:program (typed-route/attempt source :double {'a :double} options))
        algorithm (get-in typed [:equations 0 :algorithm])
        scheduled (:form (segop-lower/segop-lower-pass typed options))
        graph (equation-graph/make algorithm scheduled)
        emitted (sg/generate-kernel-graph
                 graph :array-types {'a :double} :scalar-types {'n :long})
        [phase-one phase-two] (mapv :operation (:nodes emitted))
        partials (first (:inputs (second (mapv :operation (:nodes graph)))))
        temporary-specs (graph-call/temporary-specs
                         emitted {'n {:type :long :value 1025}})]
    (is (= 2 (count (:nodes emitted))))
    (is (every? kart/kernel-artifact? [phase-one phase-two]))
    (is (str/includes? (:source phase-two) (str (name partials) "[")))
    (is (not (re-find #"\ba\[" (:source phase-two)))
        "the cross-block target kernel must not resurrect the original reduction body")
    (is (= 2 (second (get temporary-specs partials)))
        "dynamic two-phase scratch resolves through KernelLaunch IR at call time")))

(deftest typed-stencil-emits-a-guarded-typed-artifact
  (let [source '(let* [result
                       (raster.par/stencil!
                        du [u] 1 :dirichlet double i n
                        (* alpha inv-dx2
                           (+ (clojure.core/aget u (clojure.core/- i 1))
                              (* -2.0 (clojure.core/aget u i))
                              (clojure.core/aget u (clojure.core/+ i 1)))))]
                      result)
        typed (-> (typed-route/attempt
                   source :double {'du :double 'u :double}
                   {:scalar-types {'n :long 'alpha :double 'inv-dx2 :double}})
                  :program :equations first :algorithm)
        scheduled (first (lower/lower-typed-stencil typed :cpu:0 :dtype :double))
        artifact (sg/generate-segstencil-kernel
                  scheduled
                  :array-types {'du :double 'u :double}
                  :scalar-types {'n :long 'alpha :double 'inv-dx2 :double})
        source (:source artifact)
        guard-position (str/index-of source "if (idx < 1 || idx >= _n_bound - 1)")
        first-load-position (str/index-of source "u[")]
    (is (kart/kernel-artifact? artifact))
    (is (= '[u du alpha inv-dx2 _n_bound] (mapv :name (:abi artifact))))
    (is (= [:input :output :scalar :scalar :scalar] (mapv :kind (:abi artifact))))
    (is (= [:double :double :double :double :int] (mapv :dtype (:abi artifact))))
    (is (= :no-write-alias (get-in artifact [:abi 0 :aliasing])))
    (is (= {:kind :stencil :boundary :dirichlet :radius 1} (:effects artifact)))
    (is (and guard-position first-load-position (< guard-position first-load-position))
        "the boundary test must dominate every neighborhood read")))

(defn- segred-source [body-expr]
  (let [form (list 'raster.par/reduce 'acc 0.0 'j 'n body-expr)
        s (soac/par-form->soac 'result form 0)
        segops (lower/lower-reduce s nil)]
    (:source (sg/generate-segred-kernel (first segops) 'out :dtype :float))))

(defn- emitted-scan-graph
  ([form] (emitted-scan-graph form {}))
  ([form opts]
   (let [node (soac/par-form->soac 'scan-result form 71)
         operations (lower/lower-scan node nil)
         graph (lower/scan-kernel-graph node operations opts)]
     (sg/generate-scan-kernel-graph graph
                                    :scalar-types (:scalar-types opts)))))

(deftest segscan-graph-emits-one-verified-artifact-per-scheduled-node
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 0.0 i n double
                                   (+ acc (* scale (clojure.core/aget values i))))
                 {:array-types {'values :double 'out :double}
                  :scalar-types {'scale :double}})
        [intra block carry] (mapv :operation (:nodes emitted))]
    (is (kgraph/kernel-graph? emitted))
    (is (every? kart/kernel-artifact? [intra block carry]))
    (is (= [:intra-block :block-scan :carry-in]
           (mapv #(get-in % [:attributes :phase]) [intra block carry])))
    (testing "stage 1 owns both the result and graph temporary in one exact ABI"
      (is (= [:temporary :result :operand :parameter :bound]
             (mapv :role (:abi intra))))
      (is (= '[scale n] (take-last 2 (:arguments intra))))
      (is (re-find #"value = \(double\).*scale.*values\[idx\]" (:source intra))))
    (testing "block totals are scanned in one chunked workgroup for unbounded symbolic sizes"
      (is (= [1] (get-in block [:launch :group-count])))
      (is (re-find #"for \(int base = 0; base < _n_bound; base \+= 256\)" (:source block))))
    (testing "carry consumes the scanned total of the preceding block"
      (is (re-find #"block_totals_.*\[block - 1\].*out_\[idx\]" (:source carry))))
    (testing "entry-point names are valid C identifiers"
      (is (every? #(re-matches #"[A-Za-z_][A-Za-z0-9_]*" (:kernel-name %))
                  [intra block carry])))
    (let [[intra-node block-node carry-node] (:nodes emitted)]
      (is (= [(:id intra-node)] (:dependencies block-node)))
      (is (= (mapv :id [intra-node block-node]) (:dependencies carry-node))
          "target lowering preserves the verified schedule"))))

(deftest typed-scan-schedule-target-lowers-without-a-legacy-soac-node
  (let [source '(let* [result (raster.par/scan out acc 0.0 i n float
                                               (+ acc (clojure.core/aget values i)))]
                      result)
        typed (:program (typed-route/attempt source :float
                                             {'values :float 'out :float}))
        scheduled (:form (segop-lower/segop-lower-pass
                          typed {:dtype :float :target-device :ocl:0
                                 :array-types {'values :float 'out :float}}))
        graph (get-in scheduled [:equations 0 :attributes :kernel-graph])
        emitted (sg/generate-kernel-graph graph)]
    (is (= :typed-soac (get-in scheduled [:provenance :source-dialect])))
    (is (= 3 (count (:nodes emitted))))
    (is (every? kart/kernel-artifact? (map :operation (:nodes emitted))))
    (is (= [:intra-block :block-scan :carry-in]
           (mapv #(get-in % [:operation :attributes :phase]) (:nodes emitted))))))

(deftest segscan-artifact-abi-preserves-mixed-input-storage
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 0.0 i n double
                                   (+ acc (double (clojure.core/aget values i))))
                 {:array-types {'values :int 'out :double}})
        intra (get-in emitted [:nodes 0 :operation])
        values-slot (first (filter #(= 'values (:name %)) (:abi intra)))]
    (is (= :int (:dtype values-slot)))
    (is (= :int (:kernel-dtype values-slot)))
    (is (re-find #"__global const int\* restrict values" (:source intra)))))

(deftest small-segscan-emits-the-degenerate-one-artifact-graph
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 0.0 i 64 double (+ acc (aget values i))))
        artifact (get-in emitted [:nodes 0 :operation])]
    (is (= 1 (count (:nodes emitted))))
    (is (= :single (get-in artifact [:attributes :phase])))
    (is (= [1] (get-in artifact [:launch :group-count])))
    (is (not (re-find #"block_totals" (:source artifact))))))

(deftest segscan-carry-emits-the-certified-combine
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 1.0 i n double (* acc (aget values i))))
        carry (get-in emitted [:nodes 2 :operation])]
    (is (re-find #"block_totals_.*\[block - 1\] \* out_\[idx\]" (:source carry))
        "carry propagation must use the certified monoid, not a hard-coded addition")))

;; ================================================================
;; Silently-ignored-information family: a SegRed combine that the
;; extractor only partially models must FAIL LOUD, not miscompile.
;; ================================================================

(deftest segred-nary-combine-rejected
  (testing "a variadic (+ acc x y) combine is rejected, not silently summing only x"
    ;; Before the fix, op-args extraction took a0=acc a1=x and DROPPED y — emitting a
    ;; kernel that reduced with just x. A segmented reduce combine is binary (op acc elem).
    (is (thrown-with-msg?
         Exception #"certified associative reduction"
         (segred-source '(+ acc (clojure.core/aget a j) (clojure.core/aget b j))))))
  (testing "the ordinary binary combine still lowers unchanged"
    (let [source (segred-source '(+ acc (clojure.core/aget a j)))]
      (is (re-find #"tree_combined_.* = .* \+ " source))
      (is (re-find #"output\[" source)))))

(deftest segred-multistatement-lambda-rejected
  (testing "a reduce lambda with a multi-statement let body is rejected, not (last bdy)-dropped"
    ;; The single-aset-void store-drop shape on the reduce side: (last bdy) silently dropped
    ;; the earlier body forms.
    (is (thrown-with-msg?
         Exception #"certified associative reduction"
         (segred-source '(let* [t (clojure.core/aget a j)] (+ acc t) (+ acc t)))))))

(deftest segred-devirtualized-aget-lowers-to-subscript
  (testing "the parametric (.invk aget-impl …) shape — the qlinear-k side of #55"
    (let [aget-invk (with-meta (list '.invk 'raster.arrays/aget_m_floats_long-impl 'a 'j)
                      {:raster.op/original 'raster.arrays/aget
                       :raster.type/tag 'float})
          plus-invk (with-meta (list '.invk 'raster.numeric/_plus__m_double_double-impl
                                     'acc (list 'double aget-invk))
                      {:raster.op/original 'raster.numeric/+
                       :raster.type/tag 'double})
          src (segred-source plus-invk)]
      (is (not (re-find #"gpufn_aget" src))
          "devirtualized aget must normalize to a subscript, not a helper call")
      (is (re-find #"a\[" src)))))

(deftest segred-bare-aget-lowers-to-subscript
  (testing "the typed clojure.core/aget shape — the decoder-gpu side of #55"
    (let [src (segred-source '(+ acc (clojure.core/aget a j)))]
      (is (not (re-find #"gpufn_aget" src)))
      (is (re-find #"a\[" src)))))

(deftest segred-emits-complete-ordered-typed-abi
  (let [form (with-meta '(raster.par/reduce acc 0.0 i n
                                            (+ (float acc) (* scale (clojure.core/aget a i))))
               {:raster.type/elem-type :float})
        s (soac/par-form->soac 'result form 0)
        segred (first (lower/lower-reduce s nil))
        k (sg/generate-segred-kernel segred 'result-buffer :dtype :float
                                     :scalar-types {'scale :float 'n :int})]
    (is (kart/kernel-artifact? k))
    (testing "signature, ABI and compiler values have one identical order"
      (is (= '[a result-buffer scale _n_bound] (mapv :name (:abi k))))
      (is (= [:input :output :scalar :scalar] (mapv :kind (:abi k))))
      (is (= [:float :float :float :int] (mapv :kernel-dtype (:abi k))))
      (is (= [:operand :result :parameter :bound] (mapv :role (:abi k))))
      (is (= '[a result-buffer scale n] (:arguments k)))
      (is (= (kabi/signature-shape (:abi k))
             (kabi/source-signature-shape (:kernel-name k) (:source k))))
      (is (= :no-write-alias (get-in k [:abi 0 :aliasing]))))
    (testing "launch and semantic origin are part of the value"
      (is (klaunch/launch-spec? (:launch k)))
      (is (= [256] (get-in k [:launch :workgroup-size])))
      (is (= 1024 (get-in k [:launch :shared-memory-bytes])))
      (is (= {:dialect :segred :segop-id (:id segred)} (:provenance k)))
      (is (= :float (get-in k [:attributes :dtype])))
      (is (= :kernel-body (get-in k [:attributes :emission-route])))
      (is (some? (get-in k [:attributes :kernel-body])))))
  (testing "host-scalar staging retains the result position as an explicit nil placeholder"
    (let [form '(raster.par/reduce acc 0.0 i n (+ acc (clojure.core/aget a i)))
          s (soac/par-form->soac 'result form 0)
          k (sg/generate-segred-kernel (first (lower/lower-reduce s nil)) nil :dtype :float)]
      (is (= '[a output _n_bound] (mapv :name (:abi k))))
      (is (= '[a nil n] (:arguments k))))))

(deftest segred-kernel-body-decline-is-retained-in-the-fallback-artifact
  (let [form '(raster.par/reduce acc 0.0 i n
                                 (+ acc (clojure.core/aget a (+ i offset))))
        node (soac/par-form->soac 'result form 22 :dtype :float)
        operation (first (lower/lower-reduce node nil :dtype :float))
        artifact (sg/generate-segred-kernel
                  operation nil :dtype :float :scalar-types {'offset :int})]
    (is (= :verified-segred-opencl (get-in artifact [:attributes :emission-route])))
    (is (= :segred-kernel-body-declined
           (get-in artifact [:attributes :kernel-body-decline :reason])))
    (is (= :uniform-scalar-storage
           (get-in artifact [:attributes :kernel-body-decline :missing-rule])))
    (is (= :verified-segred-opencl
           (get-in artifact [:attributes :kernel-body-decline :fallback])))))

;; ================================================================
;; Horizontally-fused multi-output SegMap: the SECONDARY output (written
;; only via a side-effect aset in the fused lambda) must be a NON-const
;; __global array param — never a scalar. Before the fix it was declared
;; `float hfuse_out__N` (a scalar) while the body indexed it as an array:
;; a broken kernel, and the extraction layer then eval'd the bare buffer
;; symbol on the host (`Unable to resolve symbol: hfuse_out__N`).
;; ================================================================

(deftest segmap-fused-secondary-output-is-array-param
  (let [form '(raster.par/map! hout1 i n float
                               (do (raster.arrays/aset hout2 i (float (* (clojure.core/aget d i)
                                                                         (clojure.core/aget a i))))
                                   (* (clojure.core/aget d i) (clojure.core/aget b i))))
        s (soac/par-form->soac 'da form 0)
        segmap (first (lower/lower-map s nil :dtype :float))
        k (sg/generate-segmap-kernel segmap 'hout1 :dtype :float)]
    (testing "secondary output is an array param, not a scalar param"
      (is (kart/kernel-artifact? k))
      (is (some #{'hout2} (kart/attribute k :array-params)))
      (is (not (some #{'hout2} (kart/attribute k :scalar-params))))
      (is (some #{'hout2} (kart/attribute k :written-arrays))))
    (testing "declared __global and NON-const in the C signature"
      (is (re-find #"__global float\* restrict hout2" (:source k)))
      (is (not (re-find #"const float\* restrict hout2" (:source k)))))
    (testing "written via subscript in the body"
      (is (re-find #"hout2\[" (:source k))))
    (testing "the ABI preserves secondary-output, primary-output, scalar and bound order"
      (is (= '[a b d hout2 hout1 _n_bound] (mapv :name (:abi k))))
      (is (= [:input :input :input :output :output :scalar]
             (mapv :kind (:abi k))))
      (is (= '[a b d hout2 hout1 n] (:arguments k))))))

(deftest segmap-abi-preserves-integer-scalar-type
  (let [form '(raster.par/map! out i n float
                               (clojure.core/aget a (clojure.core/+ i offset)))
        s (soac/par-form->soac 'out form 1)
        segmap (first (lower/lower-map s nil :dtype :float))
        k (sg/generate-segmap-kernel segmap 'out :dtype :float
                                     :scalar-types {'offset :int})]
    (is (= '[a out offset _n_bound] (mapv :name (:abi k))))
    (is (= [:float :float :int :int] (mapv :dtype (:abi k))))
    (is (re-find #"int offset" (:source k)))
    (is (= '[a out offset n] (:arguments k)))))
