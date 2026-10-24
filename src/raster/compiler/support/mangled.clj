(ns raster.compiler.support.mangled
  "Shared helpers for compiler-facing deftm and impl symbol mangling."
  (:require [clojure.string :as str]))

(def op-name-map
  "Operator-character encoding shared by deftm and specialized names.
   This is a structural identifier codec, never semantic or dispatch evidence."
  {\+ "plus", \- "minus", \* "star", \/ "div", \< "lt", \> "gt",
   \= "eq", \! "bang", \? "pred", \& "amp", \| "pipe", \~ "tilde"})

(defn encode-operator-name
  "Encode a wholly operator-character name; ordinary helper names stay intact."
  [base-name]
  (let [n (name base-name)]
    (if (every? #(contains? op-name-map %) n)
      (str "_" (apply str (map op-name-map n)) "_")
      n)))

(def ^:private operator-tokens
  (mapv (fn [[character token]] [token character]) op-name-map))

(defn decode-operator-name
  "Decode a COMPLETE canonical operator encoding, or return nil.
   No partial token match changes an ordinary source helper's name."
  [encoded]
  (when (and (string? encoded) (> (count encoded) 2)
             (str/starts-with? encoded "_") (str/ends-with? encoded "_"))
    (loop [remaining (subs encoded 1 (dec (count encoded))) decoded ""]
      (if (empty? remaining)
        decoded
        (when-let [[token character] (some #(when (str/starts-with? remaining (first %)) %) operator-tokens)]
          (recur (subs remaining (count token)) (str decoded character)))))))

(def ^:private legacy-name-replacements
  ;; Preserve previously accepted partial/legacy spellings without making
  ;; them the codec for new multi-character operators. Canonical replacements
  ;; still derive from the one forward table; qmark predates its pred token.
  (concat (map (fn [operator] [(encode-operator-name operator) operator])
               ["+" "-" "*" "/" "<" ">" "<=" ">=" "="])
          [[(str "_" (op-name-map \!)) "!"] ["_qmark" "?"]]))

(defn strip-impl-suffix
  "Strip the -impl suffix from a function name string."
  [name-str]
  (if (.endsWith ^String name-str "-impl")
    (.substring ^String name-str 0 (- (count name-str) 5))
    name-str))

(defn strip-deftm-mangling
  "Strip deftm mangling suffixes from a function name string.
	Example: dense_m_Object_Object -> dense"
  [name-str]
  (let [impl-free (strip-impl-suffix name-str)
        mangled-idx (.indexOf ^String impl-free "_m_")]
    (if (neg? mangled-idx)
      impl-free
      (.substring ^String impl-free 0 mangled-idx))))

(defn unqualified-base-name
  "Return the unqualified base name for a possibly mangled symbol."
  [sym]
  (when (symbol? sym)
    (strip-deftm-mangling (name sym))))

(defn extract-deftm-base
  "Extract the base symbol from a possibly deftm-mangled symbol.
	Preserves the original namespace when present."
  [sym]
  (when (symbol? sym)
    (let [base-name (unqualified-base-name sym)
          ns-str (namespace sym)]
      (if ns-str
        (symbol ns-str base-name)
        (symbol base-name)))))

(defn impl->op
  "Recover a qualified source operation from a deftm implementation identity."
  [sym]
  (when (and (symbol? sym) (namespace sym)
             (pos? (.indexOf ^String (name sym) "_m_")))
    (let [base (unqualified-base-name sym)
          decoded (or (decode-operator-name base)
                      (reduce (fn [s [encoded original]]
                                (.replace ^String s ^CharSequence encoded ^CharSequence original))
                              base legacy-name-replacements))]
      (symbol (namespace sym) decoded))))
