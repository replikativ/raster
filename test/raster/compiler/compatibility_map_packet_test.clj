(ns raster.compiler.compatibility-map-packet-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.par]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.backend.gpu.segop-opencl :as segop-cl]
            [raster.compiler.passes.parallel.segop-lower-pass :as slp]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(def options {:dtype :double :array-types {'a :double 'out :double 'indices :int 'values :double}
              :scalar-types {'n :long}})
(def stable '(raster.par/pmap i n double (clojure.core/aget a i)))
(def compound '(raster.par/pmap i (clojure.core/alength a) double (clojure.core/aget a i)))

(def float-map-sources
  {:float-result '(let* [left (raster.par/pmap i n float (clojure.core/aget a i))] left)
   :double-consumer '(let* [left (raster.par/pmap i n float (clojure.core/aget a i))
                            right (raster.par/pmap j n double (clojure.core/aget left j))] right)
   :legacy-consumer '(let* [left (raster.par/pmap i n float (clojure.core/aget a i))
                            result (raster.par/map! out j n double (clojure.core/aget left j))] result)})

(defn- without-legacy-maps [f]
  (let [original soac/par-form->soac]
    (with-redefs [soac/par-form->soac
                  (fn [& arguments]
                    (let [expression (second arguments)]
                      (when (par/par-map-pure-form? expression)
                        (throw (AssertionError. "admitted map reached legacy semantic inference")))
                      (apply original arguments)))]
      (f))))

(defn- map-equations [packet]
  (filterv #(= 'map (some-> % :algorithm dialect/equations first dialect/operation-kind))
           (:equations packet)))

(defn- assert-equation-contracts [packet]
  (is (= packet (program/validate! packet)))
  (doseq [equation (:equations packet)]
    (is (= (:operands equation) (:inputs (dialect/facts (:algorithm equation)))))
    (is (= (:results equation) (dialect/outputs (:algorithm equation))))
    (is (= equation (program/equation-for-binding packet (second (:site equation)) (:source equation))))))

(deftest pure-map-result-contract-carries-its-declared-storage-before-routing
  (let [source '(let* [result (raster.par/pmap i n float (clojure.core/aget a i))] result)
        typed (frontend/form->program source options)
        conversion (some #(when (dialect/scalar-convert-form? %) (dialect/scalar-convert-parts %))
                         (tree-seq coll? seq typed))]
    (is (= :float (get-in (dialect/facts typed) [:values 'result :dtype])))
    (is (= [:double :float]
           ((juxt :source-dtype :target-dtype) (:attributes conversion))))
    ;; The existing merge-value authority must still reject incompatible externally supplied
    ;; evidence, rather than resolving the contradiction by erasing the map's contract.
    (is (= :source-value-conflict
           (try (frontend/form->program
                 source (assoc options :values {'result (av/tensor {:dtype :double :shape ['n]
                                                                   :representation {:kind :plain}})}))
                nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftest pure-maps-import-complete-packets-without-legacy-inference
  (without-legacy-maps
   #(doseq [[expression expected] [[stable 1] [compound 2]]]
      (let [result (slp/segop-lower-pass (list 'let* ['result expression] 'result) options)
            packet (:form result)]
        (is (= expected (count (:equations packet))))
        (is (= 1 (count (map-equations packet))))
        (is (= 1 (get-in result [:stats :typed-soac-reused])))
        (is (= (dec expected) (get-in result [:stats :typed-scalar-equations])))
        (assert-equation-contracts packet)
        (doseq [n [0 1 3]]
          (let [a (double-array (take n [1.25 -3.5 7.0]))
                original (eval (list 'fn ['a 'n] (list 'let* ['result expression] 'result)))
                transformed (eval (list 'fn ['a 'n] (:source packet)))]
            (is (= (vec (original a n)) (vec (transformed a n)) (vec a)))))))))

(deftest map-prefix-and-return-stay-after-earlier-body-effects
  (without-legacy-maps
   #(let [source (list 'let* [] '(swap! events conj :before) compound)
          packet (:form (slp/segop-lower-pass source options))
          pairs (mapv vec (partition 2 (second (:source packet))))
          equation (first (map-equations packet))]
      (is (= '(swap! events conj :before) (second (first pairs))))
      (is (= [:body 1] (get-in equation [:provenance :compatibility-source-site])))
      (assert-equation-contracts packet)
      (doseq [expression [source (:source packet)]]
        (let [events (atom [])
              function (eval (list 'fn ['a 'events] expression))]
          (is (= [1.25 -3.5] (vec (function (double-array [1.25 -3.5]) events))))
          (is (= [:before] @events))))))
  (let [failure (ex-info "before extent" {})
        source (list 'let* [] '(throw failure) compound)
        packet (:form (slp/segop-lower-pass source options))]
    (doseq [expression [source (:source packet)]]
      (is (identical? failure
                     (try ((eval (list 'fn ['a 'failure] expression)) (double-array [1.0]) failure)
                          nil (catch Throwable error error)))))))

(deftest fresh-prefixes-and-repeated-body-maps-have-distinct-sites
  (without-legacy-maps
   #(doseq [source [(list 'let* ['rstr_extent_0 7 'left compound 'right compound] '[left right])
                    (list 'let* [] compound compound)]]
      (let [packet (:form (slp/segop-lower-pass source options))
            maps (map-equations packet)
            prefixes (filter (fn [equation] (get-in equation [:attributes :host-only])) (:equations packet))]
        (is (= 2 (count maps)))
        (is (= 2 (count (distinct (map :site maps)))))
        (is (= 2 (count (distinct (mapcat :results prefixes)))))
        (is (= (count (:equations packet)) (count (distinct (map :id (:equations packet))))))
        (assert-equation-contracts packet)))))

(deftest mixed-ode-control-keeps-admitted-map-packet
  ;; Retain the actual nested time-loop shape from structured_control_route_test, rather than
  ;; a plain arraycopy loop which the current frontend can already represent.
  (let [source
        '(let* [n (int (clojure.core/alength u0))
                u (clojure.core/aclone u0)
                scratch (clojure.core/double-array n)
                time-loop
                (dotimes [step steps]
                  (raster.par/map! scratch i n double (+ (clojure.core/aget u i) 1.0))
                  (let* [next (raster.par/pmap j n double
                                             (+ (clojure.core/aget u j)
                                                (clojure.core/aget scratch j)
                                                (* 0.0 step)))]
                    (java.lang.System/arraycopy next 0 u 0 n)))
                after (raster.par/pmap k n double (* 2.0 (clojure.core/aget u k)))]
           after)
        bodies '#{(+ (clojure.core/aget u i) 1.0)
                  (+ (clojure.core/aget u j) (clojure.core/aget scratch j) (* 0.0 step))
                  (* 2.0 (clojure.core/aget u k))}
        source (walk/postwalk (fn [form] (if (contains? bodies form)
                                          (with-meta form {:tag 'double :raster.type/tag 'double})
                                          form)) source)
        options (assoc options :array-types {'u0 :double 'u :double 'scratch :double}
                               :scalar-types {'steps :long 'n :int})
        whole (route/attempt source :double (:array-types options) {:scalar-types (:scalar-types options)})]
    (is (some? (:declined whole)))
    (without-legacy-maps
     #(let [packet (:form (slp/segop-lower-pass source options))]
        (is (= 1 (count (map-equations packet))))
        (is (= (get (into {} (map vec (partition 2 (second source)))) 'time-loop)
               (get (into {} (map vec (partition 2 (second (:source packet))))) 'time-loop)))
        (assert-equation-contracts packet)
        (doseq [expression [source (:source packet)]]
          (let [result ((eval (list 'fn ['u0 'steps] expression)) (double-array [2.0 -4.0]) 2)]
            (is (= [22.0 -26.0] (vec result)))))))))

(deftest map-storage-conversion-is-retained-and-consumed-by-backend
  (without-legacy-maps
   #(doseq [storage ['float 'double]]
      (let [source (list 'let* ['result (list 'raster.par/pmap 'i '(clojure.core/alength a)
                                             storage '(clojure.core/aget a i))] 'result)
            packet (:form (slp/segop-lower-pass source options))
            compile (requiring-resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)
            emitted (compile packet :dtype :double :compile-spirv? false :min-elements 0
                             :array-types {'a :double})
            equation (first (map-equations packet))]
        (is (= (keyword (name storage)) (get-in packet [:values (first (:results equation)) :dtype])))
        (is (= 1 (count (:kernels emitted))))
        (is (zero? (get-in emitted [:stats :fallback] 0)))
        (is (nil? (get-in emitted [:stats :segop-relowered])))
        (assert-equation-contracts packet)))))

(deftest malformed-map-success-cannot-enter-legacy-fallback
  (let [original route/attempt]
    (without-legacy-maps
     #(doseq [change [(fn [_] {})
                      (fn [packet] (assoc packet :source '(let* [dangling] result)))
                      (fn [packet] (update-in packet [:equations 0] assoc :site [:binding 'missing]))
                      (fn [packet] (update-in packet [:equations 0] assoc :site []))]]
        (with-redefs [route/attempt (fn [& arguments]
                                     (update (apply original arguments) :program change))]
          (is (= :raster/bug
                 (try (slp/segop-lower-pass (list 'let* ['result stable] 'result) options)
                      nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))))

(deftest later-map-consumes-retained-earlier-array-storage
  (let [source (:double-consumer float-map-sources)]
    (without-legacy-maps
     #(let [packet (:form (slp/segop-lower-pass source options))
            right (last (map-equations packet))]
        (is (= :float (get-in packet [:values [:binding 'left] :dtype])))
        (is (= :float (get-in (dialect/facts (:physical-algorithm right)) [:values 'left :dtype])))
        (is (some #{[:binding 'left]} (:operands right)))
        (assert-equation-contracts packet)
        (let [a (double-array [1.00000001 -3.5])
              original (eval (list 'fn ['a 'n] source))
              transformed (eval (list 'fn ['a 'n] (:source packet)))]
          (is (= (vec (original a 2)) (vec (transformed a 2)))))))))

(deftest legacy-consumer-retains-the-admitted-map-storage-contract
  ;; map! remains outside this singleton migration. Its compatibility equation must consume
  ;; the preceding pmap's Float storage, rather than reconstructing it as the Double default.
  (let [source (:legacy-consumer float-map-sources)]
    (without-legacy-maps
     #(let [packet (:form (slp/segop-lower-pass source options))
            consumer (last (:equations packet))
            retained-dtype (get-in packet [:values [:binding 'left] :dtype])
            operation (first (:operations consumer))
            artifact (segop-cl/generate-scheduled-segmap-kernel
                      operation :dtype :double :scalar-types {'n :long}
                      :array-types {'left retained-dtype 'out :double})
            abi (into {} (map (juxt :name :dtype)) (:abi artifact))]
        (is (nil? (:algorithm consumer)) "legacy consumer is not relabelled as typed admission")
        (is (= :float retained-dtype))
        (is (some #{[:binding 'left]} (:operands consumer)))
        (is (= :float (get abi 'left)))
        (is (= :double (get abi 'out)))
        (doseq [expression [source (:source packet)]]
          (let [a (double-array [1.00000001 -3.5])
                out (double-array 2)
                expected (mapv (fn [x] (double (float x))) (vec a))
                returned ((eval (list 'fn ['a 'out 'n] expression)) a out 2)]
            (is (identical? out returned))
            (is (= expected (vec returned)))))))))
