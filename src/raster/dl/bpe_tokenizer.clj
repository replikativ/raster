(ns raster.dl.bpe-tokenizer
  "Byte-level BPE tokenizer compatible with HuggingFace's tokenizer.json format.

  Supports GPT-2, LLaMA, and other models that use byte-level BPE with the
  standard HuggingFace JSON tokenizer format. Implements:
    1. GPT-2 byte-to-unicode mapping (fixed bijection)
    2. Regex-based pre-tokenization (GPT-2 pattern)
    3. BPE merge application in priority order
    4. Special token handling (BOS/EOS/PAD)

  Usage:
    (def tok (load-bpe-tokenizer \"path/to/tokenizer.json\"))
    (vec (encode tok \"Hello world\"))
    (decode tok (encode tok \"Hello world\"))"
  (:require [clojure.data.json :as json])
  (:import [java.util HashMap]
           [java.util.regex Pattern]))

;; ---------------------------------------------------------------------------
;; GPT-2 byte <-> unicode mapping
;; ---------------------------------------------------------------------------

(defn- build-byte->unicode
  "Build the GPT-2 byte-to-unicode mapping. Printable ASCII/Latin bytes map to
  themselves as chars; remaining bytes map to chars starting at 256."
  []
  (let [mapping (HashMap.)
        ;; Printable ranges that map to themselves
        ;; 33-126 (ASCII printable), 161-172, 174-255 (Latin-1 supplement)
        identity-ranges [[33 126] [161 172] [174 255]]
        identity-bytes (into #{}
                             (mapcat (fn [[lo hi]] (range lo (inc hi))))
                             identity-ranges)]
    ;; Identity-mapped bytes
    (doseq [b identity-bytes]
      (.put mapping (int b) (char b)))
    ;; Remaining bytes map to chars starting at 256
    (loop [b 0 offset 0]
      (when (<= b 255)
        (if (contains? identity-bytes b)
          (recur (inc b) offset)
          (do
            (.put mapping (int b) (char (+ 256 offset)))
            (recur (inc b) (inc offset))))))
    mapping))

(def ^:private byte->unicode (build-byte->unicode))

(def ^:private unicode->byte
  "Inverse mapping: unicode char -> byte value."
  (let [m (HashMap.)]
    (doseq [entry (.entrySet ^HashMap byte->unicode)]
      (.put m (.getValue entry) (.intValue (.getKey entry))))
    m))

(defn- bytes->bpe-string
  "Convert a byte array to a string using the GPT-2 byte-to-unicode mapping."
  ^String [^bytes bs]
  (let [sb (StringBuilder. (alength bs))]
    (dotimes [i (alength bs)]
      (let [b (bit-and (aget bs i) 0xFF)]
        (.append sb (.get ^HashMap byte->unicode (int b)))))
    (.toString sb)))

(defn- bpe-string->bytes
  "Convert a BPE-encoded string back to raw bytes."
  ^bytes [^String s]
  (let [n (.length s)
        out (byte-array n)]
    (dotimes [i n]
      (let [c (.charAt s i)
            b (.get ^HashMap unicode->byte c)]
        (aset out i (unchecked-byte (int b)))))
    out))

;; ---------------------------------------------------------------------------
;; Pre-tokenization (GPT-2 regex)
;; ---------------------------------------------------------------------------

(def ^:private ^Pattern gpt2-pattern
  "GPT-2 pre-tokenization regex pattern."
  (Pattern/compile
   (str "'s|'t|'re|'ve|'m|'ll|'d"
        "| ?\\p{L}+"
        "| ?\\p{N}+"
        "| ?[^\\s\\p{L}\\p{N}]+"
        "|\\s+")))

(defn- pre-tokenize
  "Split text into pre-tokens using the GPT-2 regex pattern.
  Returns a vector of strings."
  [^String text]
  (let [m (.matcher gpt2-pattern text)]
    (loop [tokens (transient [])]
      (if (.find m)
        (recur (conj! tokens (.group m)))
        (persistent! tokens)))))

;; ---------------------------------------------------------------------------
;; BPE merge logic
;; ---------------------------------------------------------------------------

(defn- get-pairs
  "Get all adjacent symbol pairs from a sequence of symbols.
  Returns a set of [left right] pairs."
  [symbols]
  (when (> (count symbols) 1)
    (let [n (count symbols)]
      (loop [i 0 pairs (transient #{})]
        (if (>= i (dec n))
          (persistent! pairs)
          (recur (inc i) (conj! pairs [(nth symbols i) (nth symbols (inc i))])))))))

(defn- apply-merges
  "Apply BPE merges to a sequence of symbols (each a BPE-encoded string).
  merge-ranks is a HashMap from [left right] pair to priority (lower = merge first)."
  [symbols ^HashMap merge-ranks]
  (loop [symbols (vec symbols)]
    (if (<= (count symbols) 1)
      symbols
      (let [pairs (get-pairs symbols)]
        (if (nil? pairs)
          symbols
          ;; Find the highest priority (lowest rank) pair
          (let [best-pair (reduce (fn [best pair]
                                    (let [rank (.get merge-ranks pair)]
                                      (if (nil? rank)
                                        best
                                        (if (nil? best)
                                          [pair rank]
                                          (let [[_ best-rank] best]
                                            (if (< (long rank) (long best-rank))
                                              [pair rank]
                                              best))))))
                                  nil
                                  pairs)]
            (if (nil? best-pair)
              symbols ;; No more merges possible
              (let [[merge-pair _] best-pair
                    [left right] merge-pair
                    merged (str left right)
                    n (count symbols)]
                ;; Merge all occurrences of the pair
                (recur
                 (loop [i 0 new-syms (transient [])]
                   (if (>= i n)
                     (persistent! new-syms)
                     (if (and (< i (dec n))
                              (= (nth symbols i) left)
                              (= (nth symbols (inc i)) right))
                       (recur (+ i 2) (conj! new-syms merged))
                       (recur (inc i) (conj! new-syms (nth symbols i)))))))))))))))

;; ---------------------------------------------------------------------------
;; Tokenizer loading
;; ---------------------------------------------------------------------------

(defn- parse-merges
  "Parse merge list from tokenizer.json model.merges into a HashMap of
  [left right] -> rank (index)."
  [merges-list]
  (let [m (HashMap. (count merges-list))]
    (dotimes [i (count merges-list)]
      (let [merge-str (nth merges-list i)
            space-idx (.indexOf ^String merge-str " ")]
        (when (pos? space-idx)
          (let [left (.substring ^String merge-str 0 space-idx)
                right (.substring ^String merge-str (inc space-idx))]
            (.put m [left right] (int i))))))
    m))

(defn- parse-vocab
  "Parse vocab map from tokenizer.json model.vocab into a HashMap of token->id."
  [vocab-map]
  (let [m (HashMap. (count vocab-map))]
    (doseq [[token id] vocab-map]
      (.put m (name token) (long id)))
    m))

(defn- parse-added-tokens
  "Parse added_tokens list into maps for lookup."
  [added-tokens-list]
  (let [token->id (HashMap.)
        id->token (HashMap.)]
    (doseq [entry added-tokens-list]
      (let [content (get entry "content")
            id (long (get entry "id"))]
        (.put token->id content id)
        (.put id->token id content)))
    {:token->id token->id
     :id->token id->token}))

(defn load-bpe-tokenizer
  "Load a HuggingFace tokenizer.json and return a tokenizer map.

  The returned map contains:
    :vocab       - HashMap of token string -> token id
    :id->token   - HashMap of token id -> token string
    :merges      - HashMap of [left right] pair -> merge priority
    :added-tokens - special tokens from added_tokens list
    :bos-id      - beginning-of-sequence token id (or nil)
    :eos-id      - end-of-sequence token id (or nil)
    :pad-id      - padding token id (or nil)"
  [path]
  (let [raw (json/read-str (slurp path))
        model (get raw "model")
        vocab-raw (get model "vocab")
        merges-raw (get model "merges")
        added-tokens-raw (get raw "added_tokens")
        vocab (parse-vocab vocab-raw)
        ;; Build reverse map id -> token
        id->token (let [m (HashMap. (.size vocab))]
                    (doseq [entry (.entrySet vocab)]
                      (.put m (.longValue ^Number (.getValue entry))
                            (.getKey entry)))
                    m)
        merges (parse-merges merges-raw)
        added (parse-added-tokens (or added-tokens-raw []))
        ;; Merge added tokens into vocab/id->token
        _ (doseq [entry (.entrySet ^HashMap (:token->id added))]
            (let [tok (.getKey entry)
                  id (.longValue ^Number (.getValue entry))]
              (.put vocab tok id)
              (.put id->token id tok)))
        ;; Find common special token ids
        bos-id (.get vocab "<|endoftext|>")  ;; GPT-2 uses this as BOS
        eos-id (.get vocab "<|endoftext|>")  ;; GPT-2 uses this as EOS
        ;; Also check for LLaMA-style tokens
        bos-id (or bos-id (.get vocab "<s>"))
        eos-id (or eos-id (.get vocab "</s>"))
        pad-id (or (.get vocab "<pad>") (.get vocab "<|padding|>"))]
    {:vocab vocab
     :id->token id->token
     :merges merges
     :added-tokens added
     :bos-id (some-> bos-id long)
     :eos-id (some-> eos-id long)
     :pad-id (some-> pad-id long)}))

;; ---------------------------------------------------------------------------
;; Encoding
;; ---------------------------------------------------------------------------

(defn- bpe-encode-word
  "BPE-encode a single pre-token string into a sequence of token IDs.
  Converts the string to bytes, maps through byte->unicode, then applies merges."
  [^String word ^HashMap vocab ^HashMap merges]
  (let [;; Convert word to bytes, then to BPE unicode chars
        bs (.getBytes word "UTF-8")
        ;; Each byte becomes a single-char symbol
        symbols (mapv (fn [i]
                        (let [b (bit-and (aget bs i) 0xFF)]
                          (String/valueOf (.get ^HashMap byte->unicode (int b)))))
                      (range (alength bs)))
        ;; Apply BPE merges
        merged (apply-merges symbols merges)
        ;; Look up each merged symbol in vocab
        unk-id (or (.get vocab "<unk>")
                   (.get vocab "<|endoftext|>")
                   0)]
    (mapv (fn [sym]
            (let [id (.get vocab sym)]
              (if (nil? id) unk-id (long id))))
          merged)))

(defn encode
  "Encode text into token IDs using byte-level BPE. Returns long[].

  The text is pre-tokenized using the GPT-2 regex pattern, then each pre-token
  is BPE-encoded. Special tokens (BOS/EOS) are NOT automatically added;
  use encode-with-special to include them."
  ^longs [tokenizer ^String text]
  (let [{:keys [vocab merges added-tokens]} tokenizer
        ^HashMap added-tok->id (:token->id added-tokens)
        pre-tokens (pre-tokenize text)
        ids (transient [])]
    ;; Encode each pre-token
    (doseq [pre-token pre-tokens]
      ;; Check if this matches an added token exactly
      (let [added-id (when (and added-tok->id (pos? (.size added-tok->id)))
                       (.get added-tok->id pre-token))]
        (if added-id
          (conj! ids (long added-id))
          (doseq [id (bpe-encode-word pre-token vocab merges)]
            (conj! ids id)))))
    (long-array (persistent! ids))))

(defn encode-with-special
  "Encode text with BOS/EOS special tokens prepended/appended.
  Returns long[]."
  ^longs [tokenizer ^String text]
  (let [{:keys [bos-id eos-id]} tokenizer
        core-ids (vec (encode tokenizer text))
        ids (cond-> []
              bos-id (conj bos-id)
              true   (into core-ids)
              eos-id (conj eos-id))]
    (long-array ids)))

;; ---------------------------------------------------------------------------
;; Decoding
;; ---------------------------------------------------------------------------

(defn decode
  "Decode token IDs back to a string. Handles byte-level BPE decoding:
  converts BPE unicode chars back to bytes, then decodes as UTF-8."
  ^String [tokenizer ids]
  (let [{:keys [^HashMap id->token]} tokenizer
        ;; Collect all BPE-encoded token strings
        sb (StringBuilder.)]
    (doseq [id (if (instance? (Class/forName "[J") ids)
                 (seq ids)
                 ids)]
      (let [token (.get id->token (long id))]
        (when token
          (.append sb ^String token))))
    ;; Convert BPE unicode string back to bytes, then to UTF-8
    (let [bpe-str (.toString sb)]
      (String. (bpe-string->bytes bpe-str) "UTF-8"))))

;; ---------------------------------------------------------------------------
;; Convenience
;; ---------------------------------------------------------------------------

(defn vocab-size
  "Return the total vocabulary size."
  [tokenizer]
  (.size ^HashMap (:vocab tokenizer)))

(defn token->id
  "Look up a single token string to its ID, or nil."
  [tokenizer ^String token]
  (.get ^HashMap (:vocab tokenizer) token))

(defn id->token
  "Look up a single token ID to its string, or nil."
  [tokenizer ^long id]
  (.get ^HashMap (:id->token tokenizer) id))

;; ---------------------------------------------------------------------------
;; Testing
;; ---------------------------------------------------------------------------

;; (def tok (load-bpe-tokenizer "path/to/gpt2/tokenizer.json"))
;; (vec (encode tok "Hello world"))
;; (decode tok (encode tok "Hello world"))
;; (decode tok (encode tok "The quick brown fox jumps over the lazy dog."))
;; (vocab-size tok)
