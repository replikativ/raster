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
            [raster.compiler.ir.kernel-body :as kernel-body]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.segmap-body :as segmap-body]
            [raster.compiler.passes.parallel.segred-body :as segred-body]
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

(deftest scheduled-reduction-graph-has-no-source-shaped-target-fallback
  (let [source '(let* [result (raster.par/reduce acc 0.0 i n
                                                 (+ acc (clojure.core/aget a i)))]
                      result)
        options {:dtype :double :array-types {'a :double}}
        typed (:program (typed-route/attempt source :double {'a :double} options))
        algorithm (get-in typed [:equations 0 :algorithm])
        scheduled (:form (segop-lower/segop-lower-pass typed options))
        graph (equation-graph/make algorithm scheduled)
        decline (fn [& _]
                  (throw (ex-info "simulated portable coverage gap"
                                  {:reason :segred-kernel-body-declined
                                   :missing-rule :simulated
                                   :fallback :none})))]
    (doseq [target-dialect [:opencl-intel :cuda]]
      (testing (name target-dialect)
        (let [exception
              (try
                (with-redefs [segred-body/lower decline]
                  (sg/generate-kernel-graph graph :array-types {'a :double}
                                            :target-dialect target-dialect))
                nil
                (catch clojure.lang.ExceptionInfo exception exception))]
          (is (= :kernel-graph-target-lowering-missing (:reason (ex-data exception))))
          (is (= :simulated (get-in (ex-data exception)
                                    [:kernel-body-decline :missing-rule])))
          (is (= :none (:fallback (ex-data exception)))))))))

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
        artifact (sg/generate-segstencil-kernel-body
                  scheduled
                  :array-types {'du :double 'u :double}
                  :scalar-types {'n :long 'alpha :double 'inv-dx2 :double})
        kernel-body (get-in artifact [:attributes :kernel-body])
        if-region (some #(when (= "IfRegion" (some-> % class .getSimpleName)) %)
                        (:operations kernel-body))
        interior-region (some #(when (= "IfRegion" (some-> % class .getSimpleName)) %)
                              (:then-operations if-region))
        loads (filterv #(= "ScalarLoad" (some-> % class .getSimpleName))
                       (:then-operations interior-region))]
    (is (kart/kernel-artifact? artifact))
    (is (= '[u du alpha inv-dx2 _n_bound] (mapv :name (:abi artifact))))
    (is (= [:input :output :scalar :scalar :scalar] (mapv :kind (:abi artifact))))
    (is (= [:double :double :double :double :int] (mapv :dtype (:abi artifact))))
    (is (= :no-write-alias (get-in artifact [:abi 0 :aliasing])))
    (is (= {:kind :stencil :boundary :dirichlet :radius 1} (:effects artifact)))
    (is (= :portable-segstencil (get-in kernel-body [:attributes :kind])))
    (is (= 3 (count loads))
        "the interior control region must dominate every neighborhood read")))

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

(defn- kernel-body-operations
  [artifact]
  (letfn [(walk [operations]
            (mapcat
             (fn [operation]
               (cons operation
                     (concat (walk (or (:operations operation) []))
                             (walk (or (:then-operations operation) []))
                             (walk (or (:else-operations operation) [])))))
             operations))]
    (walk (get-in artifact [:attributes :kernel-body :operations]))))

(defn- operation-kind [operation]
  (some-> operation class .getSimpleName))

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
    (is (every? #(= :portable-segscan
                    (get-in % [:attributes :kernel-body :attributes :kind]))
                [intra block carry]))
    (testing "stage 1 owns both the result and graph temporary in one exact ABI"
      (is (= [:temporary :result :operand :parameter :bound]
             (mapv :role (:abi intra))))
      (is (= '[scale n] (take-last 2 (:arguments intra))))
      (let [operations (kernel-body-operations intra)]
        (is (some #(and (= "ScalarLoad" (operation-kind %))
                        (= 'values (:buffer %)))
                  operations))
        (is (some #(and (= "ScalarCompute" (operation-kind %))
                        (= :* (get-in % [:expression :op])))
                  operations))))
    (testing "block totals are scanned in one chunked workgroup for unbounded symbolic sizes"
      (is (= [1] (get-in block [:launch :group-count])))
      (let [loop (some #(when (= "ForLoop" (operation-kind %)) %) (kernel-body-operations block))]
        (is (= [0 '_n_bound 256 :ordered]
               [(:lower loop) (:upper loop) (:step loop)
                (get-in loop [:attributes :association])]))))
    (testing "carry consumes the scanned total of the preceding block"
      (let [loads (filter #(= "ScalarLoad" (operation-kind %))
                          (kernel-body-operations carry))]
        (is (= #{(first (map :id (:temporaries emitted))) 'out}
               (set (map :buffer loads))))))
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
        carry (get-in emitted [:nodes 2 :operation])
        combines (for [operation (kernel-body-operations carry)
                       :when (= "ScalarCompute" (operation-kind operation))]
                   (get-in operation [:expression :op]))]
    (is (= [:*] (vec combines))
        "carry propagation must use the certified monoid, not a hard-coded addition")))

(deftest segscan-carry-separates-its-launch-width-from-the-scanned-block-width
  (let [node (soac/par-form->soac
              'scan-result
              '(raster.par/scan out acc 0.0 i n float (+ acc (aget values i)))
              73)
        operations (lower/lower-scan node nil :dtype :float)
        scan-width (get-in operations [0 :grid :block-size])
        carry-width (quot scan-width 2)
        operations (assoc-in operations [2 :grid]
                             (segop/->KernelGrid
                              (klaunch/ceil-div 'n carry-width)
                              carry-width 0))
        graph (lower/scan-kernel-graph
               node operations {:array-types {'values :float 'out :float}})
        carry (get-in (sg/generate-scan-kernel-graph graph) [:nodes 2 :operation])
        scan-block (some #(when (= 'scan-block (:id %)) %)
                         (get-in carry [:attributes :kernel-body :indices]))]
    (is (= [carry-width] (get-in carry [:launch :workgroup-size])))
    (is (= scan-width
           (get-in carry [:attributes :kernel-body :schedule :scan-block-size])))
    (is (= (kernel-body/expression :floor-div 'scan-index scan-width)
           (:expression scan-block)))))

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
  (let [product (with-meta '(* scale (clojure.core/aget a i))
                  {:raster.type/tag 'float})
        form (with-meta (list 'raster.par/reduce 'acc 0.0 'i 'n
                              (list '+ (list 'float 'acc) product))
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

(deftest mixed-index-scalars-stay-on-the-portable-reduction-route
  (let [form '(raster.par/reduce acc 0.0 i n
                                 (+ acc (clojure.core/aget a (+ i offset))))
        node (soac/par-form->soac 'result form 22 :dtype :float)
        operation (first (lower/lower-reduce node nil :dtype :float))
        artifact (sg/generate-segred-kernel
                  operation nil :dtype :float :scalar-types {'offset :int})]
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
    (is (= :int (some #(when (= 'offset (:name %)) (:dtype %)) (:abi artifact))))
    (is (re-find #"element_index.* \+ offset" (:source artifact)))
    (is (nil? (get-in artifact [:attributes :kernel-body-decline])))))

(deftest mixed-floating-reduction-arithmetic-is-explicit-typed-ssa
  (let [product (with-meta
                  '(* (double scale) (clojure.core/aget a i))
                  {:raster.type/tag 'double})
        form (with-meta
               (list 'raster.par/reduce 'acc 0.0 'i 'n
                     (list '+ 'acc (list 'float product)))
               {:raster.type/elem-type :float})
        node (soac/par-form->soac 'result form 88 :dtype :float)
        operation (first (lower/lower-reduce node :ze:0 :dtype :float))
        artifact (sg/generate-segred-kernel
                  operation nil :dtype :float
                  :scalar-types {'scale :double 'n :long})
        loop-body (:operations (first (:operations
                                       (get-in artifact [:attributes :kernel-body]))))
        casts (keep (fn [operation]
                      (let [expression (:expression operation)]
                        (when (= :cast (:op expression))
                          [(:result-type expression) (:options expression)])))
                    loop-body)]
    (is (= [[:double {:rounding :exact :overflow :exact}]
            [:float {:rounding :nearest-even :overflow :ieee}]]
           casts))
    (is (= :double (some #(when (= 'scale (:name %)) (:kernel-dtype %))
                         (:abi artifact))))
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))))

(deftest reduction-scalar-typing-declines-uncertified-language-semantics
  (let [options {:index 'i :coordinate 'element-index :dtype :float
                 :arrays #{'a} :scalars #{'scale 'counter}
                 :scalar-types {'scale :double 'counter :long}}
        decline (fn [expression]
                  (try
                    (segred-body/lower-element-operations expression options)
                    nil
                    (catch clojure.lang.ExceptionInfo exception
                      (ex-data exception))))]
    (testing "mixed arithmetic needs its retained walker/TypedClojure result fact"
      (is (= :scalar-result-dtype
             (:missing-rule
              (decline '(* scale (clojure.core/aget a i)))))))
    (testing "even uniform Float operands do not identify clojure.core versus typed arithmetic"
      (is (= :scalar-result-dtype
             (:missing-rule
              (decline '(* (clojure.core/aget a i)
                           (clojure.core/aget a i)))))))
    (testing "checked integral casts are not reinterpreted as wrap or saturation"
      (is (= :checked-scalar-cast (:missing-rule (decline '(int scale)))))
      (is (= :checked-scalar-cast (:missing-rule (decline '(byte 128))))))
    (testing "integral value arithmetic waits for an explicit overflow contract"
      (is (= :integral-scalar-arithmetic
             (:missing-rule
              (decline (with-meta '(+ counter 1) {:raster.type/tag 'long}))))))
    (testing "literal and expression results cannot be silently narrowed to the accumulator"
      (is (= {:missing-rule :element-result-dtype :element-dtype :double}
             (select-keys (decline 0.1) [:missing-rule :element-dtype])))
      (is (= :element-result-dtype
             (:missing-rule
              (decline (with-meta '(* scale (clojure.core/aget a i))
                         {:raster.type/tag 'double}))))))
    (testing "beta reduction cannot erase a typed local conversion"
      (let [local (with-meta 'local {:raster.type/tag 'float})]
        (is (= :typed-scalar-binding-conversion
               (:missing-rule
                (decline (list 'let* [local 0.1] (list 'double local))))))))))

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

(deftest scheduled-segmap-uses-one-observable-compatibility-boundary
  (let [form '(raster.par/map! out i n float
                               (clojure.core/aget a i))
        operation (-> (soac/par-form->soac 'out form 2)
                      (lower/lower-map nil :dtype :float)
                      first)
        artifact
        (with-redefs [segmap-body/lower
                      (fn [& _]
                        (throw (ex-info "simulated portable coverage gap"
                                        {:reason :segmap-kernel-body-declined
                                         :missing-rule :simulated
                                         :fallback :none})))]
          (sg/generate-scheduled-segmap-kernel operation :dtype :float))]
    (is (= :verified-segmap-opencl
           (get-in artifact [:attributes :emission-route])))
    (is (= :simulated
           (get-in artifact [:attributes :kernel-body-decline :missing-rule])))
    (is (= :verified-segmap-opencl
           (get-in artifact [:attributes :kernel-body-decline :fallback])))))
