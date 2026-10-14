(ns raster.compiler.passes.parallel.public-scalar-placement-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]))

(defn- source-program []
  (frontend/form->program
   '(let* [out (clojure.core/float-array 1)
           written (raster.par/map-void! i 1
                     (clojure.core/aset out i (float (clojure.core/aget input i))))
           checkpoint (double (clojure.core/aget out 0))
           updated (raster.par/map-void! i 1
                     (clojure.core/aset state i
                                       (float (clojure.core/aget input i))))]
      checkpoint)
   {:dtype :float :array-types {'input :float 'state :float}}))

(deftest public-checkpoint-keeps-later-equations-and-source-placement
  (let [source (source-program)
        projected (resident/terminal-device-scalar-projection source)]
    (is projected)
    (is (= (dialect/equations source) (dialect/equations projected)))
    (is (= (get-in (dialect/facts source) [:equations 2 :provenance])
           (get-in (dialect/facts projected) [:equations 2 :provenance])))
    (is (= (get-in (dialect/facts source) [:values 'state])
           (get-in (dialect/facts projected) [:values 'state])))))

(deftest observable-state-outputs-do-not-hide-the-public-scalar
  (let [source (source-program)
        state-result (first (nth (last (dialect/equations source)) 2))
        result (first (dialect/outputs source))
        observed (dialect/make (dialect/facts source) (dialect/equations source)
                               [state-result result])
        projected (resident/terminal-device-scalar-projection observed)]
    (is projected)
    (is (= [state-result result] (dialect/outputs projected)))
    (is (= (dialect/equations source) (dialect/equations projected)))
    (is (resident/validated-terminal-resident-output (route/program-envelope projected) result))))

(deftest global-checkpoint-selection-precedes-component-splitting
  (let [source (source-program)
        result (first (dialect/outputs source))
        state-result (first (nth (last (dialect/equations source)) 2))
        entries (mapv #(hash-map :kind :typed-soac :equation-id (second %))
                      (dialect/equations source))
        select #'route/public-scalar-checkpoint
        request {:typed-program source :outputs [state-result result] :entries entries}]
    (is (= [result] (select request)))
    (let [later (list '= 10 ['later]
                      (list 'scalar {:dtypes [:double]} [result]
                            '(lambda [x] (region [] [x]))))
          facts (-> (dialect/facts source)
                    (assoc-in [:values 'later] (get-in (dialect/facts source) [:values result]))
                    (assoc-in [:equations 10] {:effects #{} :aliases {} :attributes {}
                                              :provenance {:source-binding-id 10}}))
          consumed (dialect/make facts (conj (dialect/equations source) later) [result])]
      (is (= [] (select (assoc request :typed-program consumed
                              :entries (conj entries {:kind :typed-soac :equation-id 10})))))
      (is (= [] (select (assoc request :typed-program consumed :outputs [result 'later]
                              :entries (conj entries {:kind :typed-soac :equation-id 10}))))))))

(deftest checkpoint-use-audit-includes-bookkeeping-and-storage
  (let [source (source-program)
        facts (dialect/facts source)
        equations (dialect/equations source)
        result (first (dialect/outputs source))
        unused? #(resident/public-scalar-unused-in-program? %1 %2 result)]
    (is (unused? facts equations))
    (testing "a later ordinary scalar capture is a use"
      (let [later (list '= 10 ['later]
                        (list 'scalar {:dtypes [:double]} [result]
                              '(lambda [x] (region [] [x]))))]
        (is (not (unused? facts (conj equations later))))))
    (doseq [changed [(assoc-in facts [:attributes :allocations 0 :extent] result)
                     (assoc-in facts [:values 'state :shape] [(list 'value result)])
                     (assoc-in facts [:attributes :host-read-values] [result])
                     (assoc-in facts [:equations 3 :aliases] {'alias result})
                     (assoc-in facts [:equations 3 :attributes :result-storage]
                               [{:destination result}])]]
      (is (not (unused? changed equations))))))

(deftest two-independent-public-scalars-do-not-select-a-checkpoint
  (let [source (source-program)
        result (first (dialect/outputs source))
        other (list '= 10 ['other]
                    (list 'scalar {:dtypes [:double]} []
                          '(lambda [] (region [] [1.0]))))
        facts (-> (dialect/facts source)
                  (assoc-in [:values 'other] (get-in (dialect/facts source) [:values result]))
                  (assoc-in [:equations 10] {:effects #{} :aliases {} :attributes {}
                                            :provenance {:source-binding-id 10}}))
        program (dialect/make facts (conj (dialect/equations source) other) [result 'other])]
    (is (resident/public-scalar-unused-in-program? facts (dialect/equations program) result))
    (is (resident/public-scalar-unused-in-program? facts (dialect/equations program) 'other))
    (is (nil? (resident/terminal-device-scalar-projection program)))))

(deftest backing-alias-host-observation-blocks-checkpoint-activation
  (let [source (source-program)
        facts (dialect/facts source)
        logical (first (nth (first (dialect/equations source)) 2))
        observed (dialect/make
                  (assoc-in facts [:attributes :host-read-values] [logical])
                  (dialect/equations source) (dialect/outputs source))]
    (is (vector? logical) "the alias closure must support canonical non-symbol value IDs")
    (is (nil? (resident/terminal-device-scalar-projection observed)))))

(deftest later-write-through-a-logical-alias-blocks-checkpoint-activation
  (let [source (source-program)
        equations (dialect/equations source)
        logical (first (nth (first equations) 2))
        later (last equations)
        later-result (first (nth later 2))
        facts (-> (dialect/facts source)
                  (assoc-in [:equations (second later) :aliases] {later-result logical})
                  (assoc-in [:equations (second later) :attributes :result-storage 0 :destination]
                            logical))
        alias-write (dialect/make facts equations (dialect/outputs source))]
    (is (nil? (resident/terminal-device-scalar-projection alias-write)))))

(deftest transported-selection-is-closed-and-does-not-fallback
  (let [source (source-program)
        facts (dialect/facts source)
        result (first (dialect/outputs source))
        select #(dialect/make (assoc-in facts [:attributes :public-scalar-checkpoint] %)
                              (dialect/equations source) (dialect/outputs source))]
    (is (resident/terminal-device-scalar-projection (select [result])))
    (doseq [selection [nil [] [result result] ['unknown] result {:result result}]]
      (is (nil? (resident/terminal-device-scalar-projection (select selection)))))))
