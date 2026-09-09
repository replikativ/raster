(ns raster.compiler.passes.parallel.packed-stage-fragment
  "Shared typed byte-product stage fragment. Its owner must prove packed-stage admission."
  (:require [clojure.walk :as walk]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.kernel-body :as body]))

(defn lower
  "Build the admitted packed inner fold with the owner's scalar builder and index proof.
   `plan` is the successful inner-dp4a-plan; identities are allocated by the owning schedule.
   This fragment neither recognizes source arithmetic nor invents an overflow policy."
  [{:keys [builder lower-index scope operands inner plan packed-index carry result]}]
  (let [pack (fn [{:keys [sym map]}]
               (reduce
                (fn [word offset]
                  (let [coordinate (walk/postwalk-replace
                                    {(:axis inner) (list 'clojure.core/+
                                                        (list 'clojure.core/* packed-index 4) offset)}
                                    (am/index-expr map))
                        loaded ((:cast builder)
                                ((:load builder) sym [(lower-index coordinate scope)]) :int coordinate)
                        byte-word ((:compute builder) :bit-and :int
                                   [(:result loaded) (body/literal 255 :int)] {})
                        shifted ((:compute builder) :shl :int
                                 [(:result byte-word) (body/literal (* offset 8) :int)] {})
                        joined ((:compute builder) :bit-or :int [(:result word) (:result shifted)] {})]
                    {:operations (vec (concat (:operations word) (:operations loaded)
                                              (:operations byte-word) (:operations shifted)
                                              (:operations joined)))
                     :result (:result joined)}))
                {:operations [] :result (body/literal 0 :int)} (range 4)))
        [a b] (mapv pack operands)
        dot ((:compute builder) :dp4a :int [(:result a) (:result b) carry] {})]
    {:result result :dtype :int
     :operations [(body/->ForLoop
                   (body/value packed-index :int) 0 (:packed-extent plan) 1
                   [(body/->LoopArg (body/value carry :int) (body/literal 0 :int))]
                   (vec (concat (:operations a) (:operations b) (:operations dot)
                                [(body/->Yield [(:result dot)])]))
                   [(body/value result :int)] {})]}))
