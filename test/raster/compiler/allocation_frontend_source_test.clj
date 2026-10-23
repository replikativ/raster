(ns raster.compiler.allocation-frontend-source-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.arrays :as arrays]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]))

(defmacro allocate-body [n] `(float-array ~n))
(core/deftm allocate-helper [ref :- (Array float) n :- Long] :- (Array float)
  (allocate-body n))
(core/deftm scalar-helper [x :- Double] :- Double (+ x 1.0))
(core/deftm storage-helper [ref :- (Array float) n :- Long] :- (Array float)
  (float-array n))
(core/deftm forwarding-helper [ref :- (Array float) n :- Long] :- (Array float)
  (storage-helper ref n))
(core/deftm two-hop-helper [ref :- (Array float) n :- Long] :- (Array float)
  (forwarding-helper ref n))
(core/deftm computed-forward [ref :- (Array float) n :- Long] :- (Array float)
  (storage-helper ref (inc n)))
(core/deftm constant-forward [ref :- (Array float) n :- Long] :- (Array float)
  (storage-helper ref 1))
(core/deftm dropped-forward [ref :- (Array float) n :- Long unused :- Long] :- (Array float)
  (storage-helper ref n))
(core/deftm two-array-helper
  [left :- (Array float) right :- (Array float) n :- Long] :- (Array float)
  (float-array n))
(core/deftm duplicate-forward
  [left :- (Array float) right :- (Array float) n :- Long] :- (Array float)
  (two-array-helper left left n))
(core/deftm permuted-forward
  [left :- (Array float) right :- (Array float) n :- Long] :- (Array float)
  (two-array-helper right left n))

(defn- compiled-helper [generic arguments]
  (let [source (core/resolve-deftm-var generic {:dtype :float})
        implementation (symbol (str (:ns (meta source))) (str (:name (meta source)) "-impl"))
        receiver (ns-resolve (:ns (meta source)) (symbol (name implementation)))]
    (core/ensure-bytecode-implementation! receiver)
    {:source source :receiver receiver
     :call (with-meta (list* '.invk implementation arguments)
             {:raster.op/original (symbol (str (:ns (meta generic))) (str (:name (meta generic))))})}))

(deftest composed-allocation-helpers-use-their-actual-emitted-forwarding-chain
  (let [leaf (compiled-helper #'storage-helper '[x n])
        middle (compiled-helper #'forwarding-helper '[x n])
        outer (compiled-helper #'two-hop-helper '[x n])
        real (compiled-helper #'arrays/alloc-like '[x n])
        facts {'x {:carrier (class (float-array 0))} 'n {:carrier Long/TYPE}}]
    (doseq [helper [middle outer real]]
      (entry/call-with-definition-inputs
       (fn []
         (let [projection (descriptor/direct-allocation-projection (:call helper) {} facts)
               definition (get-in projection [:helper :definition])
               chain (take-while some? (iterate #(get-in % [:delegation :definition]) definition))
               projected (eval (list 'fn '[n] (:constructor projection)))
               original (:source helper)
               observe (fn [invoke]
                         (try (let [array (invoke)] [(class array) (vec array)])
                              (catch Throwable error {:error (class error)})))]
           (is (some? (:helper projection)))
           (is (= '(clojure.core/float-array n) (:constructor projection)))
           (is (= (if (identical? helper outer) 3 2) (count chain)))
           (is (every? #(= '[floats long] (:parameter-tags %)) chain))
           (is (not-any? #(or (fn? %) (instance? java.lang.reflect.Method %)
                             (instance? clojure.lang.Namespace %))
                         (tree-seq coll? seq definition)))
           (doseq [ref [nil (float-array 0)] n [-1 0 3]]
             (is (= (observe #(original ref n)) (observe #(projected n)))))))))
    (let [[projection observations]
          (entry/call-with-definition-inputs
           (fn [] [(descriptor/direct-allocation-projection (:call outer) {} facts)
                   @entry/*emission-inputs*]))]
      (is (some? (:helper projection)))
      (doseq [helper [leaf middle outer]]
        (is (contains? (:definition-proofs observations) (:source helper)))
        (is (contains? (:definition-proofs observations) (:receiver helper)))
        (with-redefs-fn {(:receiver helper) (fn [& _] (float-array 0))}
          #(is (thrown? clojure.lang.ExceptionInfo
                        (entry/assert-definition-inputs-current! observations))))))))

(deftest forwarding-composition-refuses-unproven-or-nonidentity-edges
  (let [helper (compiled-helper #'forwarding-helper '[x n])
        facts {'x {:carrier (class (float-array 0))} 'n {:carrier Long/TYPE}}
        context entry/frontend-completed-invocation-context
        bridge entry/frontend-reference-return-bridge
        body-entry entry/frontend-body-entry-carriers
        wrong-method (first (.getDeclaredMethods ^Class (:iface-class (types/ensure-fn-interface! '[doubles long]))))]
    (is (some? (:helper (descriptor/direct-allocation-projection (:call helper) {} facts))))
    (doseq [edit [(constantly nil)
                  #(assoc % :invocation-mode :unproved)
                  #(assoc % :receiver 'foreign/implementation)
                  #(assoc % :method nil)
                  #(assoc % :method wrong-method)
                  #(assoc-in % [:receiver-input :ports] {'foreign {:type :ref}})
                  #(update-in % [:inputs 0 :ports 'ref :slot] inc)
                  #(assoc-in % [:input-completions 1 :adaptation] :coerce)
                  #(assoc-in % [:input-completions 0 :parameter-carrier] Object)]]
      (with-redefs [entry/frontend-completed-invocation-context
                    (fn [& args] (some-> (apply context args) edit))]
        (is (nil? (descriptor/direct-allocation-projection (:call helper) {} facts)))))
    (with-redefs [entry/frontend-reference-return-bridge
                  (fn [& args] (some-> (apply bridge args)
                                      (assoc :body-return-carrier (class (double-array 0)))))]
      (is (nil? (descriptor/direct-allocation-projection (:call helper) {} facts))))
    (with-redefs [entry/frontend-body-entry-carriers
                  (fn [& args] (some-> (apply body-entry args)
                                      (assoc-in [:parameter-facts 'n :carrier] Double/TYPE)))]
      (is (nil? (descriptor/direct-allocation-projection (:call helper) {} facts))))
    (entry/call-with-definition-inputs
     (fn []
       (let [selected (entry/selected-definition
                       'raster.compiler.allocation-frontend-source-test/forwarding-helper
                       (second (:call helper)) 2)
             visited [[(:source-var selected) (get-in selected [:entry :typed-impl])]]]
         (is (nil? (#'descriptor/direct-allocation-projection-in-context (:call helper) {} facts visited)))
         (is (some? (#'descriptor/direct-allocation-projection-in-context
                     (:call helper) {} facts [[(:source-var selected) (Object.)]]))))))
    (doseq [[generic arguments] [[#'computed-forward '[x n]]
                                 [#'constant-forward '[x n]]
                                 [#'dropped-forward '[x n extra]]
                                 [#'duplicate-forward '[x other n]]
                                 [#'permuted-forward '[x other n]]]]
      (let [bad (compiled-helper generic arguments)]
        (is (nil? (descriptor/direct-allocation-projection
                   (:call bad) {} (assoc facts 'other (get facts 'x) 'extra {:carrier Long/TYPE}))))))))

(deftest selected-helper-carriers-do-not-invent-exemplar-capacity
  (let [v (core/resolve-deftm-var #'storage-helper {:dtype :float})
        implementation (symbol (str (:ns (meta v))) (str (:name (meta v)) "-impl"))
        _ (core/ensure-bytecode-implementation!
           (ns-resolve (:ns (meta v)) (symbol (name implementation))))
        call (with-meta (list '.invk implementation 'x 'size)
               {:raster.op/original 'raster.compiler.allocation-frontend-source-test/storage-helper})
        facts {'x {:carrier (class (float-array 0))} 'size {:carrier Long/TYPE}}
        projected (descriptor/direct-allocation-projection call {} facts)
        x (av/tensor {:dtype :float :shape [3 5] :ownership :borrowed})
        values {'x x 'size (av/tensor {:dtype :long :shape []})}]
    (is (nil? (descriptor/direct-allocation-projection call {})))
    (is (= '(clojure.core/float-array size) (:constructor projected)))
    (is (= #{:constructor :helper} (set (keys projected))))
    (is (= #{:operation :implementation :definition :arguments} (set (keys (:helper projected)))))
    (is (= ['x 'size] (get-in projected [:helper :arguments])))
    (is (= projected (descriptor/direct-allocation-projection call values facts)))
    (is (= projected (descriptor/direct-allocation-projection
                      call {'x (assoc x :dtype :f32) 'size (av/tensor {:dtype :i64 :shape []})})))
    (is (identical? x (get values 'x)))
    (is (= [3 5] (:shape x)))
    (doseq [bad [(dissoc facts 'x) (dissoc facts 'size)
                 (assoc facts 'x nil) (assoc facts 'size {})
                 (assoc-in facts ['x :carrier] Object)
                 (assoc-in facts ['x :carrier] (class (object-array 0)))
                 (assoc-in facts ['x :carrier] (class (double-array 0)))
                 (assoc-in facts ['size :carrier] Long)
                 (assoc-in facts ['size :carrier] Number)
                 (assoc-in facts ['size :carrier] Integer/TYPE)
                 (assoc-in facts ['size :carrier] Short/TYPE)
                 (assoc-in facts ['size :carrier] Boolean/TYPE)
                 (assoc-in facts ['size :carrier] Character/TYPE)]]
      (is (nil? (descriptor/direct-allocation-projection call {} bad))))
    (doseq [bad [(assoc facts 'x nil) (assoc facts 'size {})
                 (assoc-in facts ['x :carrier] nil)
                 (assoc-in facts ['x :carrier] Object)
                 (assoc-in facts ['x :carrier] Float/TYPE)
                 (assoc-in facts ['x :carrier] (class (double-array 0)))
                 (assoc-in facts ['size :carrier] Long)
                 (assoc-in facts ['size :carrier] Double/TYPE)]]
      (is (nil? (descriptor/direct-allocation-projection call values bad))
          "an existing logical value cannot conceal an explicit incompatible carrier"))
    (is (nil? (descriptor/direct-allocation-projection
               (with-meta (list '.invk implementation 'x '(clojure.core/inc size)) (meta call))
               {} facts)))
    (is (= {:constructor '(clojure.core/float-array n)}
           (descriptor/direct-allocation-projection '(clojure.core/float-array n) {} {'n {:carrier Object}})))))

(deftest ordered-storage-facts-reach-the-actual-selected-allocation-helper
  (let [v (core/resolve-deftm-var #'storage-helper {:dtype :float})
        implementation (symbol (str (:ns (meta v))) (str (:name (meta v)) "-impl"))
        _ (core/ensure-bytecode-implementation!
           (ns-resolve (:ns (meta v)) (symbol (name implementation))))
        call (with-meta (list '.invk implementation 'alias 'size)
               {:raster.op/original 'raster.compiler.allocation-frontend-source-test/storage-helper})
        descriptions [{:kind :scalar :id 0 :sym 'alias :expr 'x}
                      {:kind :scalar :id 1 :sym 'size :expr '(clojure.core/* m n)}
                      {:kind :scalar :id 2 :sym 'out :expr call}]
        x (av/tensor {:dtype :float :shape ['m 'n]})
        analyze (fn [descriptions values]
                  (#'frontend/source-storage-values descriptions
                   {'x :float 'alias :float 'out :float}
                   {'m :long 'n :long}
                   {'m :long 'n :long 'size :long} values {}))
        result (analyze descriptions {'x x})
        allocation (first (:allocations result))]
    (is (= 1 (count (:allocations result))))
    (is (= 'out (:destination allocation)))
    (is (= implementation (get-in allocation [:source-projection :helper :implementation])))
    (is (= ['alias 'size] (get-in allocation [:source-projection :helper :arguments])))
    (is (identical? call (:source-expression allocation)))
    (is (identical? x (get-in result [:values 'alias])))
    (is (= :float (get-in result [:values 'out :dtype])))
    (is (empty? (:allocations (analyze descriptions {})))
        "array element declarations do not invent exemplar capacity")
    (is (empty? (:allocations (analyze (subvec (vec descriptions) 2) {'x x 'alias x})))
        "a future or missing scalar binding cannot be supplied by helper shape inference")))

(deftest constructor-result-types-and-alias-values-have-producer-evidence
  (let [n (av/tensor {:dtype :long :shape []})
        core-description {:kind :scalar :id 0 :sym 'out :expr '(clojure.core/float-array n)}
        helper (compiled-helper #'storage-helper '[x n])
        helper-description (assoc core-description :expr (:call helper))
        facts {'x {:carrier (class (float-array 0))} 'n {:carrier Long/TYPE}}
        contracts (fn [description arrays carriers]
                    (#'frontend/allocation-contracts [description] arrays {'n :long} {'n n} {} carriers))]
    (doseq [[description carriers] [[core-description {}] [helper-description facts]]]
      (let [allocation (first (contracts description {} carriers))]
        (is (= :float (:dtype allocation)))
        (is (= 'n (:extent allocation))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"constructor and retained element type disagree"
                            (contracts description {'out :double} carriers))))
    (is (empty? (contracts (assoc core-description :expr '(unregistered/float-array n)) {} {}))))
  (let [x (av/tensor {:dtype :float :shape [2 3] :ownership :borrowed :logical-layout {:kind :dense}})
        result (#'frontend/source-storage-values [{:kind :scalar :id 0 :sym 'alias :expr 'x}]
                                                {} {} {} {'x x} {})]
    (is (identical? x (get-in result [:values 'alias])))
    (is (empty? (:carrier-facts result)))
    (is (empty? (:pending-inputs result)))))

(deftest refused-helper-analysis-does-not-become-a-source-import
  (let [v (core/resolve-deftm-var #'scalar-helper {:dtype :double})
        _ (core/prepare-deftm-frontend! v)
        implementation (symbol (str (:ns (meta v))) (str (:name (meta v)) "-impl"))
        call (with-meta (list '.invk implementation 1.0)
               {:raster.op/original 'raster.compiler.allocation-frontend-source-test/scalar-helper})]
    (entry/call-with-definition-inputs
     (fn []
       (let [before @entry/*emission-inputs*]
         (is (nil? (descriptor/direct-allocation-projection call {})))
         (is (= before @entry/*emission-inputs*)))))))

(deftest direct-allocation-uses-the-original-frontend-not-an-earlier-walk-cache
  (let [v (core/resolve-deftm-var #'allocate-helper {:dtype :float})
        impl (ns-resolve (:ns (meta v)) (symbol (str (:name (meta v)) "-impl")))
        cache (core/ensure-walked-body! v)
        original @#'allocate-body
        operation 'raster.compiler.allocation-frontend-source-test/allocate-helper
        implementation (symbol (str (:ns (meta impl))) (str (:name (meta impl))))
        call (fn [argument]
               (with-meta (list '.invk implementation 'x argument)
                 {:raster.op/original operation}))]
    (try
      (alter-var-root #'allocate-body
                      (constantly (fn [_ _ n] (list 'clojure.core/float-array
                                                   (list 'clojure.core/inc n)))))
      (core/ensure-bytecode-implementation! impl)
      (is (= 2 (alength ^floats (allocate-helper (float-array 0) 1))))
      (is (= cache (:raster.core/deftm-walked-body (meta v))))
      (let [projected (descriptor/direct-allocation-projection
                       (call 1) {'x {:dtype :float :shape [0]}})]
        (is (= '(clojure.core/float-array (clojure.core/inc 1)) (:constructor projected)))
        (is (= ['x 1] (get-in projected [:helper :arguments]))))
      (doseq [argument [1.0 1N '(clojure.core/inc 0) 'unknown]]
        (is (nil? (descriptor/direct-allocation-projection
                   (call argument) {'x {:dtype :float :shape [0]}}))))
      (finally (alter-var-root #'allocate-body (constantly original))))))
