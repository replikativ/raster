(ns raster.ad.purity
  "Purity analysis bridge between beichte and Raster.

  Beichte 0.2+ uses a four-level effect lattice:
    :pure < :local < :mutation < :io

  Raster uses this to validate code for AD, GPU, and other compilation
  targets via beichte's compilable? check.

  Usage:
    (init-raster-registry!)         ;; call once at startup
    (pure-sexp? '(Math/sin x))    ;; -> true
    (validate-for-ad! body params) ;; warns on impure ops in active flow"
  (:require [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.core.types :as types]))

(defn init-raster-registry!
  "Initialize the shared Raster Beichte context."
  []
  (effects/init-raster-context!))

;; ================================================================
;; S-expression purity check
;; ================================================================

(defn pure-sexp?
  "Check if an S-expression is pure.
  Returns true if pure, or a map with the effect level if not."
  [sexp]
  (try
    (let [effect (effects/analyze-effect sexp)]
      (if (= :pure effect)
        true
        {:effect effect :sexp sexp}))
    (catch Exception e
      {:problem :analysis-error
       :message (.getMessage e)
       :sexp sexp})))

(defn pure-var?
  "Check if a var is pure according to beichte."
  [v]
  (= :pure (effects/analyze-var-effect v)))

;; ================================================================
;; Symbol-level purity check (for walked S-expressions)
;; ================================================================

(def ^:private known-impure-ops
  "Ops known to be impure in deftm bodies."
  #{'println 'print 'prn 'printf 'newline 'flush
    'clojure.core/println 'clojure.core/print 'clojure.core/prn
    'swap! 'reset! 'clojure.core/swap! 'clojure.core/reset!
    'aset 'clojure.core/aset
    'spit 'clojure.core/spit
    'alter-var-root 'clojure.core/alter-var-root})

(def ^:private known-pure-ops
  "Ops known to be pure in deftm bodies."
  (into #{'+ '- '* '/ 'inc 'dec 'min 'max 'abs
          'double 'float 'long 'int 'byte 'short
          'zero? 'pos? 'neg? 'even? 'odd? 'not
          'nil? 'some? 'identical? 'instance?
          'aget 'alength 'aclone
          'Math/sin 'Math/cos 'Math/tan 'Math/exp 'Math/log 'Math/sqrt
          'Math/pow 'Math/abs 'Math/max 'Math/min 'Math/atan2
          'Math/floor 'Math/ceil 'Math/round 'Math/fma
          '.invk}
        (map (fn [s] (symbol "clojure.core" (name s)))
             '[+ - * / inc dec min max abs
               double float long int byte short
               zero? pos? neg? even? odd? not
               nil? some? identical? instance?
               aget alength aclone])))

(defn- impure-mutating-op?
  "Check if a qualified op is impure (convention: name ends with !)."
  [sym]
  (when-let [n (name sym)]
    (.endsWith ^String n "!")))

(defn pure-op?
  "Check if an op symbol is known to be pure.
  Returns :pure, :impure, or :unknown."
  [op-sym]
  (cond
    (known-impure-ops op-sym)                          :impure
    (known-pure-ops op-sym)                            :pure
    (and (qualified-symbol? op-sym)
         (impure-mutating-op? op-sym))                 :impure
    ;; Check beichte registry for vars
    (and (qualified-symbol? op-sym)
         (try (= :pure (effects/analyze-var-effect (resolve op-sym)))
              (catch Exception _ false)))
    :pure
    :else                                              :unknown))

;; ================================================================
;; Pre-AD validation
;; ================================================================

(defn- collect-calls
  "Collect all function call heads from an S-expression."
  [expr]
  (cond
    (and (seq? expr) (seq expr))
    (let [head (first expr)]
      (cons {:head head :form expr}
            (mapcat collect-calls (rest expr))))

    (vector? expr)
    (mapcat collect-calls expr)

    :else nil))

(defn validate-for-ad!
  "Pre-AD validation: warn on impure ops in active data flow.
  Checks op symbols against known pure/impure registries.
  Returns nil if clean, or a seq of warning maps.

  walked-body: the walked S-expression body
  active-params: set of parameter symbols that are active for AD"
  ([walked-body active-params]
   (validate-for-ad! walked-body active-params nil))
  ([walked-body _active-params warning-meta]
   (let [calls (collect-calls walked-body)
         warnings (keep (fn [{:keys [head form]}]
                          (when (symbol? head)
                            (let [purity (pure-op? head)]
                              (when (= :impure purity)
                                {:warning :impure-op-in-ad-path
                                 :op head
                                 :form form}))))
                        calls)]
     (when (seq warnings)
       (doseq [w warnings]
         (types/emit-warning! warning-meta
                              :impure-ad
                              (str "WARNING: Impure op in AD path: " (:op w))))
       warnings))))
