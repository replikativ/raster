(ns raster.compiler.passes.parallel.collective-combine
  "Project certified collective arithmetic to ordinary functional TypedSOAC.

   This is a local binary combine, not an all-reduce implementation or transport. Distributed
   refinement still owns contribution ancestry, placement, copies and numerical reassociation.
   Scheduling, target emission and executable validation use the existing compiler boundaries."
  (:require [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.backend.gpu.parallel-program-c-family :as c-family]
            [raster.compiler.backend.gpu.target :as target]))

(defn algorithm
  "Derive one dense elementwise combine with explicit scalar types and immutable result.
   The retained certificate supplies the operator; there is no collective operator registry.
   The certificate's original element expression is not copied: operands are the two arrays."
  [algebra elements]
  (scan/validate! algebra)
  (when-not (and (integer? elements) (pos? elements) (<= elements Long/MAX_VALUE))
    (throw (ex-info "collective combine requires a positive representable element count"
                    {:reason :collective-combine-extent :elements elements})))
  (let [value (av/tensor {:dtype (:dtype algebra) :shape [elements]})
        equation (list '= 'combine '[result]
                       (list 'map {:index 'i :extent elements} '[left right] []
                             (soac/lambda-form
                              '[lhs rhs]
                              [(soac/local-value 'combined (:dtype algebra)
                                                 (list (:combine algebra) 'lhs 'rhs))]
                              '[combined])))]
    (soac/make
     (soac/default-program-facts
      {:values {'left value 'right value 'result value}
       :inputs '[left right]
       :equations {'combine (soac/default-equation-facts)}})
     [equation] '[result])))

(defn- target-options! [{:keys [target-device target-descriptor target-dialect]}]
  (let [dialect (target/kernel-body-c-dialect target-descriptor)]
    (when-not (and (keyword? target-device)
                   (= target-device (:device-id target-descriptor))
                   dialect (or (nil? target-dialect) (= dialect target-dialect)))
      (throw (ex-info "collective combine needs one matching frozen GPU target snapshot"
                      {:reason :collective-combine-target :target-device target-device
                       :descriptor-device (:device-id target-descriptor)
                       :declared-dialect target-dialect :derived-dialect dialect})))
    {:target-device target-device :target-descriptor target-descriptor
     :target-dialect dialect}))

(defn- schedule [algorithm options]
  (structured-route/schedule-program
   (assoc (typed-route/program-envelope algorithm) :dialect :typed-parallel) options))

(defn emit
  "Emit a local combine through the ordinary TypedSOAC/SegMap/KernelBody vertical.
   Returns the existing checked EmittedParallelEquation, not a new executable convention.
   Target descriptor/device/dialect are supplied by the enclosing compiler's resolved options."
  [algebra elements options]
  (let [options (assoc (target-options! options) :dtype (:dtype algebra))
        scheduled (schedule (algorithm algebra elements) options)
        emitted (get-in (c-family/emit-program scheduled options)
                        [:program :equations 0 :operations 0])]
    (assoc-in emitted [:attributes :collective-target] (dissoc options :dtype))))

(defn validate!
  "Bind an emitted local combine to its exact declared algebra and extent.
   Generic emitted-equation validation alone proves its own algorithm, not the caller's monoid."
  [algebra elements emitted]
  (let [expected (algorithm algebra elements)]
    (emitted-equation/validate! emitted)
    (when-not (= expected (:algorithm emitted))
      (throw (ex-info "emitted combine differs from its declared typed monoid"
                      {:reason :collective-combine-algorithm
                       :expected expected :actual (:algorithm emitted)})))
    ;; Generic emission validates the retained schedule and artifact against each other. Bind
    ;; that schedule back to the caller's algebra too: a valid product schedule is not a sum
    ;; merely because both have the same memory interface. Reuse the exact frozen target facts.
    (let [options (assoc (target-options! (get-in emitted [:attributes :collective-target]))
                         :dtype (:dtype algebra))
          scheduled (schedule expected options)
          body (:body (equation-graph/make-for-equation scheduled (first (:equations scheduled))))
          expected-operations (mapcat :operations (:equations body))
          actual-operations (mapcat :operations (get-in emitted [:body :equations]))]
      (when-not (and (= (:target-dialect options) (get-in emitted [:provenance :target-dialect]))
                     (every? #(= (:target-dialect options)
                                  (get-in % [:operation :provenance :target-dialect]))
                             (get-in emitted [:graph :nodes])))
        (throw (ex-info "combine target emission differs from its retained target snapshot"
                        {:reason :collective-combine-target})))
      ;; SegSpace gives its private flattened index a fresh name. Rebind only that generated
      ;; binder; all logical dimensions, scalar regions, operand sets and launch grids must agree.
      (when-not (and (= (count expected-operations) (count actual-operations))
                     (every? true?
                             (map (fn [expected actual]
                                    (= (assoc-in expected [:space :flat-idx]
                                                 (get-in actual [:space :flat-idx])) actual))
                                  expected-operations actual-operations)))
        (throw (ex-info "combine schedule differs from the declared typed scalar algorithm"
                        {:reason :collective-combine-schedule}))))
    emitted))
