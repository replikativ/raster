(ns raster.ad.normalized-indexed-map-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.set :as set]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]))

(defn- indexed-map [dt it extent data-extent]
  (let [dims (filter symbol? [extent data-extent])
        values (merge {'indices (av/tensor {:dtype it :shape [extent]})
                       'data (av/tensor {:dtype dt :shape [data-extent]})
                       'out (av/tensor {:dtype dt :shape [extent] :ownership :owned})}
                      (zipmap dims (repeat (av/tensor {:dtype :long :shape []}))))]
    (soac/make
     (soac/default-program-facts
      {:values values :inputs (vec (distinct (concat '[indices data] (when (symbol? extent) [extent]))))
       :equations {'g (soac/default-equation-facts)}
       :attributes {:source-arithmetic numerics/retained-source-arithmetic}})
     [(list '= 'g '[out]
            (list 'map {:index 'i :extent extent :attributes {:stable-array-captures '[data]}}
                  '[indices] '[data]
                  (soac/lambda-form '[index data-array] '[(clojure.core/aget data-array index)])))] '[out])))

(defn- build [source]
  (normalized/linearize-indexed-map-program source (the-ns 'raster.ad.normalized-indexed-map-test)))
(defn- reason [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- alter-facts [p f] (apply list (assoc (vec p) 1 (f (soac/facts p)))))
(defn- recertify [b] (assoc b :certificate (#'normalized/certificate b)))

(deftest indexed-construction-retains-distinct-spaces-and-explicit-owned-zero-contract
  (doseq [dt [:float :double] it [:int :long] extent [0 'n]]
    (let [source (indexed-map dt it extent 'm)
          bundle (build source)
          {:keys [primal linear transpose interface required-zero-gradient]} (:gather-programs bundle)
          snapshot (:index-snapshot interface)
          result (:gradient-output interface)
          [effect] (soac/equations transpose)
          {:keys [attributes lambda]} (soac/operation-parts effect)]
      (is (identical? source (get-in bundle [:gather-programs :source])))
      (is (identical? bundle (normalized/validate! bundle)))
      (doseq [program [primal linear transpose]] (is (identical? program (soac/validate! program))))
      (doseq [program [primal linear transpose]]
        (is (not (some #{'m} (:inputs (soac/facts program)))))
        (is (= :long (get-in (soac/facts program) [:values 'm :dtype]))))
      (is (= it (get-in (soac/facts primal) [:values snapshot :dtype])))
      (is (= :owned (get-in (soac/facts primal) [:values snapshot :ownership])))
      (is (= [extent] (get-in (soac/facts linear) [:values (:tangent-output interface) :shape])))
      (is (= '[m] (get-in (soac/facts transpose) [:values result :shape])))
      (is (= {:value (:zero-gradient interface) :dtype dt :shape '[m]
              :ownership :owned :initialization :typed-zero :exclusive true} required-zero-gradient))
      (is (= :sequential (:iteration-order attributes)))
      (is (= :ordered (:conflict (soac/effect-parts (first (:body-results (soac/lambda-parts lambda)))))))
      (is (empty? (:captures (soac/operation-parts effect))))
      (is (not (some #{(:zero-gradient interface)} (:inputs (soac/facts transpose)))))
      (let [parts (soac/effect-parts (first (:body-results (soac/lambda-parts lambda))))]
        (is (= (:zero-gradient interface) (second (second (:value parts)))))
        (is (= (:zero-gradient interface) (:destination parts))))
      (is (= (:zero-gradient interface)
             (get-in (soac/facts transpose) [:equations (second effect) :aliases result]))))))

(deftest functional-frontend-indexed-map-uses-existing-layout-without-destination-erasure
  (doseq [dt [:float :double] it [:int :long]]
    (let [options {:dtype dt :array-types {'data dt 'indices it}
                   :scalar-types {'n :long 'm :long}
                   :values {'data (av/tensor {:dtype dt :shape '[m]})
                            'm (av/tensor {:dtype :long :shape []})}}
          source '(let* [out (raster.par/pmap i n nil
                                            (clojure.core/aget data (clojure.core/aget indices i)))] out)
          program (frontend/form->program (frontend/normalize-source source options) options)
          bundle (build program)]
      (is (some? program))
      (is (= '[indices] (:arrays (soac/operation-parts (first (soac/equations program))))))
      (is (= '[data] (:captures (soac/operation-parts (first (soac/equations program))))))
      (is (empty? (soac/result-storage program (second (first (soac/equations program))))))
      (is (identical? program (get-in bundle [:gather-programs :source]))))))

(deftest indexed-source-refuses-transformed-indices-aliases-effects-and-precision-drift
  (let [source (indexed-map :double :long 'n 'm)
        equation (first (soac/equations source))]
    (doseq [change [#(assoc-in % [:values 'indices :dtype] :float)
                    #(assoc-in % [:values 'out :dtype] :float)
                    #(assoc-in % [:values 'out :shape] '[m])
                    #(assoc-in % [:values 'data :shape] '[unknown])
                    #(assoc-in % [:values 'data :placement] {:device :other})
                    #(assoc-in % [:values 'out :ownership] :borrowed)
                    #(assoc % :effects #{:memory/write})
                    #(assoc-in % [:equations 'g :aliases] {'out 'data})
                    #(assoc-in % [:equations 'g :attributes :result-storage]
                               [{:destination 'data :access :write :host-return :buffer}])]]
      (is (some? (reason #(build (alter-facts source change))))))
    (doseq [body ['(clojure.core/aget data-array (clojure.core/inc index))
                  '(clojure.core/double (clojure.core/aget data-array index))
                  '(foreign.owner/aget data-array index)
                  (list 'clojure.core/aget 'data-array (with-meta 'index {:raster.type/tag 'float}))]]
      (let [changed (list '= 'g '[out]
                          (assoc (vec (nth equation 3)) 4 (soac/lambda-form '[index data-array] [body])))
            changed (apply list (assoc (vec changed) 3 (apply list (nth changed 3))))
            program (soac/make (soac/facts source) [changed] '[out])]
        (is (= :ad-normalized-indexed-map-boundary (reason #(build program))))))))

(deftest recertified-correspondence-tampering-cannot-authorize-unowned-or-unordered-transpose
  (let [bundle (build (indexed-map :double :long 'n 'm))
        interface (get-in bundle [:gather-programs :interface])]
    (doseq [changed [(assoc-in bundle [:gather-programs :required-zero-gradient :initialization] :unknown)
                     (assoc-in bundle [:gather-programs :required-zero-gradient :ownership] :borrowed)
                     (assoc-in bundle [:gather-programs :interface :index-snapshot] 'data)
                     (update-in bundle [:gather-programs :primal]
                                #(alter-facts % (fn [facts]
                                                  (assoc-in facts [:values (:index-snapshot interface) :dtype] :int))))]]
      (is (some? (reason #(normalized/validate! (recertify changed))))))))

(deftest physical-gradient-binding-remains-checked-without-overlapping-operand-roles
  (let [bundle (build (indexed-map :double :long 'n 'm))
        transpose (get-in bundle [:gather-programs :transpose])
        interface (get-in bundle [:gather-programs :interface])
        zero (:zero-gradient interface) result (:gradient-output interface)
        alternative 'alternative-zero-owner
        equation (first (soac/equations transpose))
        equation-id (second equation)]
    (let [operation (apply list (assoc (vec (nth equation 3)) 3 [zero]))
          invalid-equation (list '= equation-id (nth equation 2) operation)]
      (is (= :typed-soac-operand-role
             (reason #(soac/make (soac/facts transpose) [invalid-equation] (soac/outputs transpose))))))
    (let [operation (apply list (assoc (vec (nth equation 3)) 4 [alternative]))
            changed-equation (list '= equation-id (nth equation 2) operation)
            facts (assoc-in (soac/facts transpose) [:values alternative]
                            (get-in (soac/facts transpose) [:values zero]))
            facts (-> facts
                        (assoc-in [:equations equation-id :aliases result] alternative)
                        (assoc-in [:equations equation-id :attributes :result-storage 0 :destination]
                                  alternative))
            changed-program (soac/make facts [changed-equation] (soac/outputs transpose))
            changed (assoc-in bundle [:gather-programs :transpose] changed-program)]
        (is (identical? changed-program (soac/validate! changed-program))
            "the alternative owner retains the original dtype, shape and physical facts")
        (is (= :ad-normalized-gather-programs
               (reason #(normalized/validate! (recertify changed))))))
    (doseq [[field value] [[:dtype :float] [:shape '[n]] [:ownership :borrowed]
                          [:memory-space :device]]]
      (let [changed (update-in bundle [:gather-programs :transpose]
                               #(alter-facts % (fn [facts] (assoc-in facts [:values zero field] value))))]
        (is (some? (reason #(normalized/validate! (recertify changed)))))))))

(deftest generated-parameters-do-not-capture-core-or-conventional-source-ids
  (let [renames {'indices 'ad_gather_index 'data 'ad_gather_seed
                 'n 'ad_gather_destination 'm 'count}
        source (soac/remap-values (indexed-map :float :long 'n 'm) renames)
        bundle (build source)
        {:keys [interface transpose required-zero-gradient]} (:gather-programs bundle)
        parameters (:parameters (soac/lambda-parts
                                 (:lambda (soac/operation-parts (first (soac/equations transpose))))))]
    (is (identical? bundle (normalized/validate! bundle)))
    (is (= '[count] (:shape required-zero-gradient)))
    (is (= [(:cotangent-input interface) (:index-snapshot interface) (:zero-gradient interface)]
           parameters))
    (is (empty? (set/intersection (set parameters) (set (keys renames)))))
    (is (empty? (set/intersection (set parameters) (set (vals renames)))))))

(deftest semantic-read-metadata-never-substitutes-for-executable-callee-identity
  (let [source (indexed-map :double :long 'n 'm)
        replace-read (fn [read]
                       (soac/make (soac/facts source)
                                  [(list '= 'g '[out]
                                         (list 'map {:index 'i :extent 'n
                                                     :attributes {:stable-array-captures '[data]}}
                                               '[indices] '[data]
                                               (soac/lambda-form '[index data-array] [read])))] '[out]))]
    (is (some? (build source)))
    (is (= :typed-soac-unbound-scalar
           (reason #(build (replace-read
                           (with-meta '(.invk foreign.receiver data-array index)
                                      {:raster.op/original 'clojure.core/aget}))))))
    (is (= :ad-normalized-indexed-map-boundary
           (reason #(build (replace-read
                           (with-meta '(clojure.core/aget data-array index)
                                      {:raster.op/lexical-callee true}))))))
    (let [source-ns (create-ns (gensym "indexed_read_shadow_"))]
      (try
        ;; This actual namespace binding is deliberately not core's read Var.
        (intern source-ns 'aget (fn [& _] 99.0))
        (is (= :ad-normalized-indexed-map-boundary
               (reason #(normalized/linearize-indexed-map-program
                         (replace-read '(aget data-array index)) source-ns))))
        (is (some? (normalized/linearize-indexed-map-program source source-ns)))
        (finally (remove-ns (ns-name source-ns)))))))
