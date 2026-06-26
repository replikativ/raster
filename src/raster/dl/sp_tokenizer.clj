(ns raster.dl.sp-tokenizer
  "SentencePiece-style BPE tokenizer for HuggingFace tokenizer.json — the scheme
  used by Gemma, Llama, Mistral, Qwen and most SPM-derived models (distinct from the
  GPT-2 byte-level BPE in raster.dl.bpe-tokenizer). Implements:

    - Metaspace normalization: literal space -> U+2581, no byte-to-unicode map
    - BPE merges applied in rank order over the whole normalized sequence
    - byte_fallback: characters absent from the vocab decompose to their UTF-8
      bytes as <0xNN> tokens
    - ignore_merges: a piece already present in the vocab is emitted directly
    - TemplateProcessing post-processor: prepend <bos>
    - decoder: U+2581 -> space, ByteFallback (fuse consecutive <0xNN> to UTF-8)

  Usage:
    (def t (load-tokenizer \"...tokenizer.json\"))
    (encode t \"Hello, world!\")   ;; => [2 9259 236764 1902 236888]
    (decode t ids)"
  (:require [clojure.data.json :as json]
            [clojure.string :as str])
  (:import [java.util HashMap]
           [java.io ByteArrayOutputStream]))

(def ^:private META "▁")   ;; SentencePiece metaspace
(def ^:private SEP "\u0020") ;; merge-pair key separator (literal space)

(defn- byte-token-value
  "If token is a byte-fallback token <0xNN>, return the byte value 0-255, else nil."
  [^String tok]
  (when (and (= 6 (.length tok)) (.startsWith tok "<0x") (.endsWith tok ">"))
    (try (Integer/parseInt (subs tok 3 5) 16) (catch Exception _ nil))))

(defn load-tokenizer
  "Load a HF tokenizer.json (SentencePiece/BPE family). Returns a map with the
  vocab/merge structures + special ids and the byte-fallback maps."
  [path]
  (let [raw (json/read-str (slurp path))
        model (get raw "model")
        vocab-raw (get model "vocab")
        merges-raw (get model "merges")
        added (get raw "added_tokens")
        vocab (HashMap. (count vocab-raw))
        id->tok (HashMap. (count vocab-raw))
        _ (doseq [[tok id] vocab-raw]
            (.put vocab tok (int id))
            (.put id->tok (int id) tok))
        ;; added/special tokens override into the vocab too
        _ (doseq [a (or added [])]
            (.put vocab (get a "content") (int (get a "id")))
            (.put id->tok (int (get a "id")) (get a "content")))
        ;; merge rank: "left<SEP>right" -> priority (lower = higher priority).
        ;; merges are either ["left" "right"] (new format) or "left right" (old).
        merge-rank (HashMap. (count merges-raw))
        _ (reduce (fn [^long i m]
                    (let [[l r] (if (string? m) (str/split m #"\s" 2) m)]
                      (.put merge-rank (str l SEP r) (int i)))
                    (inc i))
                  0 merges-raw)
        special-ids (set (map #(int (get % "id")) (or added [])))]
    {:vocab vocab :id->tok id->tok :merge-rank merge-rank
     :special-ids special-ids
     :bos-id (.get ^HashMap vocab "<bos>")
     :eos-id (.get ^HashMap vocab "<eos>")
     :ignore-merges (boolean (get model "ignore_merges"))
     :byte-fallback (boolean (get model "byte_fallback"))}))

(defn- initial-symbols
  "Decompose the normalized string into the starting BPE symbols: each codepoint as
  its own vocab token, or — when absent and byte_fallback is on — its UTF-8 bytes as
  <0xNN> tokens."
  [{:keys [^HashMap vocab byte-fallback]} ^String s]
  (let [out (java.util.ArrayList.)
        n (.length s)]
    (loop [i 0]
      (when (< i n)
        (let [cp (.codePointAt s i)
              ch (String. (Character/toChars cp))]
          (if (.containsKey vocab ch)
            (.add out ch)
            (if byte-fallback
              (doseq [b (.getBytes ch "UTF-8")] (.add out (format "<0x%02X>" (bit-and b 0xFF))))
              (.add out ch)))   ;; non-byte-fallback unknown -> <unk> at lookup
          (recur (+ i (Character/charCount cp))))))
    out))

(defn- bpe
  "Apply BPE merges to one normalized piece, returning the list of token strings."
  [{:keys [^HashMap vocab ^HashMap merge-rank ignore-merges] :as tk} ^String s]
  (if (and ignore-merges (.containsKey vocab s))
    [s]
    (let [^java.util.ArrayList syms (initial-symbols tk s)]
      (loop []
        (let [n (.size syms)]
          (when (> n 1)
            (let [best (loop [i 0 bi -1 br Integer/MAX_VALUE]
                         (if (< i (dec n))
                           (let [r (.get merge-rank (str (.get syms i) SEP (.get syms (int (inc i)))))]
                             (if (and r (< (int r) br)) (recur (inc i) i (int r)) (recur (inc i) bi br)))
                           bi))]
              (when (>= (int best) 0)
                (.set syms (int best) (str (.get syms (int best)) (.get syms (int (inc best)))))
                (.remove syms (int (inc best))) ;; int -> remove(index), not remove(Object)
                (recur))))))
      (vec syms))))

(defn encode
  "Encode text to token ids. Prepends <bos> (TemplateProcessing). Space -> U+2581."
  ([tk text] (encode tk text true))
  ([{:keys [^HashMap vocab bos-id] :as tk} ^String text add-bos?]
   (let [norm (.replace text " " META)
         pieces (when (pos? (.length norm)) (bpe tk norm))
         unk (.getOrDefault vocab "<unk>" (int 0))
         ids (map (fn [s] (let [v (.get vocab s)] (if v (int v) unk))) pieces)]
     (vec (if (and add-bos? bos-id) (cons (int bos-id) ids) ids)))))

(defn decode
  "Decode token ids back to text. Reverses the decoder Sequence: U+2581 -> space, and
  fuses consecutive <0xNN> byte-fallback tokens into UTF-8. Skips special tokens."
  ([tk ids] (decode tk ids true))
  ([{:keys [^HashMap id->tok special-ids]} ids skip-special?]
   (let [sb (StringBuilder.)
         bb (ByteArrayOutputStream.)
         flush! (fn [] (when (pos? (.size bb))
                         (.append sb (String. (.toByteArray bb) "UTF-8")) (.reset bb)))]
     (doseq [id ids]
       (when-not (and skip-special? (contains? special-ids (int id)))
         (let [tok (.get id->tok (int id))]
           (when tok
             (if-let [b (byte-token-value tok)]
               (.write bb (int b))
               (do (flush!) (.append sb (.replace ^String tok META " "))))))))
     (flush!)
     (.toString sb))))
