(ns raster.dl.tokenizer
  "Minimal WordPiece tokenizer for BERT-style models.

  Loads vocab.txt (one token per line, 0-indexed) and performs:
    1. Lowercasing (if do_lower_case)
    2. Basic whitespace + punctuation splitting
    3. WordPiece subword tokenization
    4. [CLS] / [SEP] wrapping

  This is sufficient for inference. For production, use HuggingFace's
  Rust tokenizer via DJL bindings.")

(defn load-vocab
  "Load vocab.txt → {token-string → id}."
  [path]
  (let [lines (clojure.string/split (slurp path) #"\n")]
    (into {} (map-indexed (fn [i line] [line i]) lines))))

(defn- punctuation? [c]
  (or (Character/isWhitespace c)
      (contains? #{\. \, \; \: \! \? \- \' \" \( \) \[ \] \{ \} \/ \\ \@ \# \$ \% \^ \& \* \~ \` \+ \= \< \> \|} c)))

(defn- basic-tokenize
  "Split on whitespace and punctuation, keeping punctuation as tokens."
  [^String text]
  (let [text (.trim text)
        n (.length text)]
    (loop [i 0 tokens [] current (StringBuilder.)]
      (if (>= i n)
        (let [s (.toString current)]
          (if (pos? (.length s))
            (conj tokens s)
            tokens))
        (let [c (.charAt text i)]
          (if (punctuation? c)
            (let [prev (.toString current)
                  tokens (if (pos? (.length prev)) (conj tokens prev) tokens)
                  tokens (if (not (Character/isWhitespace c))
                           (conj tokens (str c))
                           tokens)]
              (recur (inc i) tokens (StringBuilder.)))
            (recur (inc i) tokens (.append current c))))))))

(defn- wordpiece-tokenize
  "WordPiece tokenization of a single word token."
  [^String token vocab ^long max-word-len]
  (if (> (.length token) max-word-len)
    ["[UNK]"]
    (loop [start 0 sub-tokens []]
      (if (>= start (.length token))
        sub-tokens
        (let [;; Find longest matching subword
              result (loop [end (.length token)]
                       (if (<= end start)
                         nil
                         (let [substr (if (== start 0)
                                        (.substring token start end)
                                        (str "##" (.substring token start end)))]
                           (if (contains? vocab substr)
                             [substr end]
                             (recur (dec end))))))]
          (if (nil? result)
            ["[UNK]"]
            (let [[sub-tok new-start] result]
              (recur (long new-start) (conj sub-tokens sub-tok)))))))))

(defn tokenize
  "Tokenize text into token IDs. Returns long[].
  Adds [CLS] at start, [SEP] at end."
  ^longs [vocab ^String text & {:keys [max-length lower-case max-word-len]
                                :or {max-length 512 lower-case true max-word-len 200}}]
  (let [text (if lower-case (.toLowerCase text) text)
        basic-tokens (basic-tokenize text)
        wp-tokens (mapcat #(wordpiece-tokenize % vocab max-word-len) basic-tokens)
        ;; Truncate to max-length - 2 (room for [CLS] and [SEP])
        wp-tokens (take (- max-length 2) wp-tokens)
        cls-id (get vocab "[CLS]" 101)
        sep-id (get vocab "[SEP]" 102)
        unk-id (get vocab "[UNK]" 100)
        ids (into [cls-id]
                  (map #(get vocab % unk-id))
                  wp-tokens)
        ids (conj ids sep-id)]
    (long-array ids)))

(defn decode
  "Convert token IDs back to text (for debugging)."
  [vocab ^longs ids]
  (let [id->token (into {} (map (fn [[k v]] [v k]) vocab))]
    (clojure.string/join " "
                         (map #(get id->token % "[?]")
                              (seq ids)))))
