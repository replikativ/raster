(ns raster.ode.multilevel-compiler
  "Implementation-local AMR providers. No compiler-wide method/function registry.
   Providers compile ordinary numerical programs and attest their stated, conditional numerics."
  (:require [raster.compiler.equation-first :as equation]
            [raster.compiler.ir.amr-execution :as execution]
            [raster.compiler.ir.validate :refer [fail!]]
            [raster.ode.multilevel :as multilevel]))

(defn- role-id [plan array]
  (let [ids (for [[id value] (:values plan)
                  :when (and (= 1 (count (:leaves value)))
                             (identical? array (get-in plan [:nodes (get-in value [:leaves 0 :node]) :source])))]
              id)]
    (when-not (= 1 (count ids))
      (fail! "numerical provider needs one semantic LinkValue for each array role"
             :multilevel-provider-role {:values (vec ids)}))
    (first ids)))

(defn- compile-transfer [workload operation-id source destination options kind method operator]
  (let [contract (execution/operation-contract workload operation-id)
        source-shape (get-in contract [:source :shape])
        target-shape (get-in contract [:target :shape])
        coarse-shape (if (= kind :prolongation) source-shape target-shape)]
    (when-not (and (= kind (get-in contract [:operation :kind]))
                   (= method (get-in contract [:operation :operator :method]))
                   (= :cell (:centering contract)) (= [2 2] (:ratio contract))
                   (= :double (get-in contract [:source-value :dtype]))
                   (= :double (get-in contract [:target-value :dtype]))
                   (= (get-in contract [:source :device]) (get-in contract [:target :device]))
                   (instance? (Class/forName "[D") source)
                   (instance? (Class/forName "[D") destination)
                   (not (identical? source destination))
                   (= (reduce *' source-shape) (alength ^doubles source))
                   (= (reduce *' target-shape) (alength ^doubles destination)))
      (fail! "provider requires its declared dyadic FP64 whole-patch operator and disjoint exact-size arrays"
             :multilevel-provider-contract {:operation operation-id :kind kind :method method}))
    (let [target (get options :target (get-in contract [:target :device]))
          compilation (equation/compile operator {:target target :dtype :double})
          plan (equation/lower compilation [destination source (first coarse-shape) (second coarse-shape)])
          source-id (role-id plan source) target-id (role-id plan destination)
          ;; Field-qualified physical identities let successive operator entries share their
          ;; owner's pool. This changes storage identities only, not instructions or ABIs.
          plan (update plan :nodes
                       #(update-vals %
                                     (fn [node]
                                       (let [field (cond
                                                     (identical? source (:source node)) (get-in contract [:source :field])
                                                     (identical? destination (:source node)) (get-in contract [:target :field]))]
                                         (when-not field
                                           (fail! "unexpected private allocation in numerical transfer"
                                                  :multilevel-provider-storage {:operation operation-id}))
                                         (assoc-in node [:view :allocation :id] [:amr-field field])))))
          numerical (if (= kind :prolongation)
                      {:mode :exact :policy :copy-cell-average
                       :error-model {:kind :exact-cell-copy :absolute-bound 0
                                     :assumptions #{:finite-inputs}}}
                      {:mode :bounded-error :policy :fp64-cell-average
                       :rounding :nearest-even :accumulator-dtype :double
                       :error-model {:kind :four-cell-average-forward-error
                                     ;; gamma_3 = 3u/(1-3u), u = 2^-53. The sum is over the
                                     ;; four fine cells contributing to this one coarse cell.
                                     :scope :output-cell
                                     :absolute-bound [:* [:gamma 3 :double] [:sum-abs-inputs] 0.25]
                                     :assumptions #{:finite-inputs :no-intermediate-overflow
                                                    :no-underflow :ieee-nearest-even
                                                    :preserved-expression-order}}})]
      (execution/attest-implementation
       workload operation-id plan
       {:source source-id :target target-id
        :producer (if (= kind :prolongation)
                    'raster.ode.multilevel-compiler/compile-prolongation
                    'raster.ode.multilevel-compiler/compile-restriction)
        :invariants #{:constant-preserving :conservative} :numerical numerical}))))

(defn compile-prolongation
  "Compile and attest constant dyadic prolongation for an analytical workload operation.
   Returns an implementation whose :link-plan and semantic roles can be installed into the
   workload's existing device-plans. Source stability/lifetime follow ordinary LinkPlan rules."
  [workload operation-id source destination options]
  (compile-transfer workload operation-id source destination options
                    :prolongation :piecewise-constant #'multilevel/prolong-constant-2d!))

(defn compile-restriction
  "Compile and attest cell averaging. Invariants are conditional on the retained FP error model;
   the caller must establish its input-domain assumptions at execution, including after producers
   update resident inputs. Certification does not dynamically inspect those device values."
  [workload operation-id source destination options]
  (compile-transfer workload operation-id source destination options
                    :restriction :cell-average #'multilevel/restrict-average-2d!))
