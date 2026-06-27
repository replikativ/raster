(ns raster.compiler.passes.placement
  "Device-placement / scheduling pass — the raster analogue of llama.cpp's
  ggml-backend-sched. Over a closed-core IR form it:

    1. assigns each op a CONCRETE device, from the op's :placement device-TYPE tag
       (:jvm | :cpu-quant | :gpu, in the op-descriptor registry) mapped through a
       device-binding POLICY (device-type -> concrete device-id), and
    2. inserts explicit cross-device TRANSFER markers wherever a value produced on one
       device is consumed on another.

  Placement is a TWO-level decision, matching the rest of the design: the op carries an
  abstract device-type tag (portable); the policy binds it to a concrete instance
  (:ze:0 / :ze:1 / :cpu:0 — several GPUs possible). A future version will run
  microbenchmarks to drive the policy (cost-model assignment); for now it is STATIC —
  the tag + the policy map. Ops with no :placement tag default to the CPU.

  This pass is analysis + transfer insertion only; the backend legs (bytecode / CPU-C /
  opencl-pass) consume the per-binding :device annotations and the transfer markers."
  (:require [raster.compiler.core.op-descriptor :as opd]))

(def default-policy
  "Static device-binding policy: op device-TYPE -> concrete device-id, plus :default for
  untagged ops. The single place the 'run this op-class on the GPU' choice lives (future:
  microbenchmark-driven). Override per compile via the policy arg."
  {:gpu :ze:0 :cpu-quant :cpu:0 :jvm :cpu:0 :default :cpu:0})

(defn- let*-form? [form]
  (and (seq? form) (= 'let* (first form))))

(defn op-device
  "Concrete device for a call expr under `policy`: its op's :placement device-type mapped
  to a device, defaulting to the CPU when the op is untagged / the expr is not a call."
  [expr policy]
  (let [dtype (when (seq? expr)
                (when-let [op (opd/semantic-op expr)]
                  (opd/get-placement op)))]
    (get policy dtype (:default policy))))

(defn- arg-syms
  "The symbols referenced by a call expr's semantic arguments (shallow — direct sym args,
  the data-flow edges between bindings)."
  [expr]
  (when (seq? expr)
    (filter symbol? (opd/call-args expr))))

(defn xfer
  "A cross-device transfer marker: move `sym` from device `from` to device `to`. A
  canonical IR node the backend lowers to the device runtime's copy (zero-copy on a
  unified arena, an explicit upload/download on discrete)."
  [from to sym]
  (list `xfer from to sym))

(defn place
  "Place a closed-core (let* [s e ...] body) form on devices under `policy`.
  Returns {:form form' :devices {sym device} :transfers [{:sym :from :to :at}]}, where
  form' annotates each binding's value with ^{:device d} metadata and wraps cross-device
  input syms in `xfer` markers. Non-let* forms are returned unplaced."
  ([form] (place form default-policy))
  ([form policy]
   (if-not (let*-form? form)
     {:form form :devices {} :transfers []}
     (let [[_ bindings & body] form
           pairs (partition 2 bindings)]
       (loop [pairs pairs, dev {}, out-binds [], transfers []]
         (if-let [[sym expr] (first pairs)]
           (let [d (op-device expr policy)
                 ;; an input produced on another device needs a transfer to d
                 needed (for [s (arg-syms expr)
                              :let [sd (get dev s)]
                              :when (and sd (not= sd d))]
                          {:sym s :from sd :to d :at sym})
                 xmap (into {} (map (juxt :sym identity) needed))
                 expr' (if (seq xmap)
                         (with-meta
                           (map (fn [a]
                                  (if-let [{:keys [from to]} (and (symbol? a) (get xmap a))]
                                    (xfer from to a)
                                    a))
                                expr)
                           (meta expr))
                         expr)
                 expr' (vary-meta expr' assoc :device d)]
             (recur (rest pairs)
                    (assoc dev sym d)
                    (into out-binds [sym expr'])
                    (into transfers needed)))
           {:form (with-meta (list* 'let* out-binds body) (meta form))
            :devices dev
            :transfers transfers}))))))

(defn summary
  "Human-readable placement summary: device per binding + the transfer cuts."
  [form policy]
  (let [{:keys [devices transfers]} (place form policy)]
    {:devices devices
     :transfers (mapv (fn [{:keys [sym from to]}] [sym from '-> to]) transfers)
     :n-transfers (count transfers)}))
