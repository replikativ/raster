(ns raster.ad.normalized-canonical-preparation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-map-element-test :as maps]
            [raster.ad.normalized-map-sum-runtime-test :as sums]
            [raster.ad.normalized-indexed-runtime-test :as indexed]
            [raster.ad.normalized-scalar-capture-test :as capture]
            [raster.ad.templates :as templates]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- plain [value] (if (number? value) value (vec value)))

(deftest normalized-producers-bootstrap-before-epoch-capture-but-queries-do-not
  (let [initialize effects/init-raster-context!
        revision templates/registry-revision
        _ (initialize)
        source (#'maps/map-program :double {})
        sum (#'sums/identity-source :double)]
    (doseq [construct [(fn [] (normalized/linearize-map-element source *ns*))
                       (fn [] (normalized/linearize-map-program source *ns*))
                       (fn [] (normalized/linearize-equation-program source *ns*))
                       (fn [] (normalized/linearize-map-sum-program sum *ns* {:active-input 'xs}))]]
      (let [events (atom [])]
        (with-redefs [effects/init-raster-context!
                      (fn [] (initialize) (swap! events conj :initialized) nil)
                      templates/registry-revision
                      (fn [] (swap! events conj :epoch) (revision))]
          (construct))
        (is (= :initialized (first @events)))
        (is (some #{:epoch} @events))))
    (let [expected (normalized/revisions)
          guard (normalized/execution-guard *ns* expected [{:parameters [] :body nil}])]
      (with-redefs [effects/init-raster-context!
                    (fn [] (throw (AssertionError. "a revision query initialized semantic context")))]
        (is (= expected (normalized/revisions)))
        (is (guard))))))

(defn- cases []
  ;; Preparing a later family may register lazy numeric specializations. Build each
  ;; witness immediately before using it; an eager vector would rightly stale earlier
  ;; witnesses at that definition boundary. Keep stale-rule refusal tests independent.
  [(fn [] [(normalized/linearize-map-program (#'maps/map-program :double {}) *ns*)
           canonical/adapt-map {'n (long 2) 'x (double-array [2.0 -3.0])}
           (double-array [0.5 -0.5]) (double-array [0.125 0.25])])
   (fn [] [(normalized/linearize-map-sum-program (#'sums/identity-source :double) *ns* {:active-input 'xs})
           canonical/adapt-map-sum {'size (long 2) 'xs (double-array [2.0 -3.0]) 'weights (double-array 2)}
           0.5 (double-array [0.125 0.25])])
   (fn [] [(#'indexed/bundle :double :long 'n 'm)
           canonical/adapt-indexed-map {'n (long 2) 'm (long 3) 'indices (long-array [1 1])
                                        'data (double-array [3.0 5.0 7.0])}
           (double-array [0.5 -0.5]) (double-array [0.125 0.25 0.5])])
   (fn [] [(normalized/linearize-map-sum-program (#'capture/source #'capture/gaussian-double :double)
                                                 *ns* {:active-input 'mu})
           canonical/adapt-map-sum {'size (long 3) 'mu 0.25 'sigma 1.7
                                    'observations (double-array [0.5 -0.25 1.0])}
           0.5 0.125])])

(deftest case-table-does-not-eagerly-prepare-revision-bound-evidence
  (let [premature (fn [& _] (throw (ex-info "eager evidence construction" {})))]
    (with-redefs [normalized/linearize-map-program premature
                  normalized/linearize-map-sum-program premature]
      (let [builders (cases)]
        (is (= 4 (count builders)))
        (is (every? fn? builders))))))

(deftest canonical-preparation-is-shared-by-every-existing-family
  (doseq [make-case (cases)]
    (let [[evidence named-adapter environment seed direction] (make-case)
          owner (canonical/adapt evidence)
          named (named-adapter evidence)
          prepared (with-redefs [canonical/adapt (fn [& _] (throw (ex-info "legacy admission consulted" {})))
                                 canonical/adapt-map (fn [& _] (throw (ex-info "map selector consulted" {})))
                                 canonical/adapt-map-sum (fn [& _] (throw (ex-info "sum selector consulted" {})))
                                 canonical/adapt-indexed-map (fn [& _] (throw (ex-info "indexed selector consulted" {})))]
                     (runtime/prepare-owner owner))
          compatibility (runtime/prepare evidence)
          args (mapv environment (:inputs prepared))
          [y pb] ((:vjp prepared) args)
          [old-y old-pb] ((:vjp compatibility) args)]
      (is (identical? (:source-ns owner) (:source-ns named)))
      (is (fingerprint/equivalent? (#'canonical/pure-envelope owner)
                                   (#'canonical/pure-envelope named)))
      (is (= (:certificate owner) (:certificate named)))
      (is (= (:inputs prepared) (:inputs compatibility)))
      (is (= (plain y) (plain old-y)))
      (is (= (mapv #(when % (plain %)) (pb seed))
             (mapv #(when % (plain %)) (old-pb seed))))
      (is (= (mapv plain ((:jvp prepared) args direction))
             (mapv plain ((:jvp compatibility) args direction)))))))

(deftest scope-and-owner-refusals-precede-materialization
  (let [evidence (normalized/linearize-map-program (#'maps/map-program :double {}) *ns*)
        projections (atom 0)]
    (with-redefs [route/materialize-host-program (fn [& _] (swap! projections inc))]
      (doseq [bad [(dissoc evidence :map-programs)
                   (assoc evidence :map-programs nil)
                   (assoc evidence :gather-programs {})
                   (assoc evidence :gather-programs {} :sum-programs {})]]
        (is (= :ad-program-bundle-scope (reason #(canonical/adapt bad))))
        (is (= :ad-program-bundle-scope (reason #(runtime/prepare bad)))))
      (is (= :ad-program-bundle-derivation (reason #(runtime/prepare-owner evidence))))
      (is (zero? @projections)))))

(deftest owner-admission-revalidates-after-an-independent-phase-boundary
  (let [evidence (normalized/linearize-map-program (#'maps/map-program :double {}) *ns*)
        owner (canonical/adapt evidence)
        revision (normalized/revisions) projections (atom 0)]
    (with-redefs [normalized/revisions (constantly (update revision :template-revision inc))
                  route/materialize-host-program (fn [& _] (swap! projections inc))]
      (is (= :ad-normalized-stale-rules (reason #(runtime/prepare-owner owner))))
      (is (zero? @projections)))))
