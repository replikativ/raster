(ns raster.compiler.passes.parallel.segmented-weighted-reduction-route
  "Backend routing for schedule-neutral segmented weighted reductions.

   Each leaf proves its own representability from plan descriptors. The router composes decline
   trails and never infers a model-level semantic operation from the source spelling."
  (:require [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.passes.parallel.indexed-attention-route :as indexed-leaf]))

(def dynamic-leaves
  [{:id :indexed-edge-list-reference :route indexed-leaf/route-dynamic}])

(defn route-dynamic
  ([plan] (route-dynamic plan nil))
  ([plan desc]
   (let [plan (swr/validate! plan)]
     (loop [[leaf & more] dynamic-leaves
            declines []]
       (if-not leaf
         {:operation plan :strategy nil :reference? false :declines declines}
         (let [result ((:route leaf) plan desc)]
           (if (:strategy result)
             (assoc result :leaf (:id leaf))
             (recur more (into declines (:declines result))))))))))

(defn route-dynamic!
  ([plan] (route-dynamic! plan nil))
  ([plan desc]
   (let [result (route-dynamic plan desc)]
     (if (:strategy result)
       result
       (throw (ex-info "no executable segmented weighted-reduction kernel route"
                       {:reason :segmented-weighted-reduction-no-kernel-route
                        :route result}))))))
