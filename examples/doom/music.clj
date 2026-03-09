(ns doom.music
  "MUS-to-MIDI converter and playback for Doom music.
  Converts WAD MUS lumps to standard MIDI and plays via javax.sound.midi."
  (:require [doom.wad :as wad])
  (:import [javax.sound.midi MidiSystem Sequencer]
           [java.io ByteArrayInputStream ByteArrayOutputStream]))

(set! *warn-on-reflection* true)

;; ================================================================
;; MUS format constants
;; ================================================================

;; MUS controller map → MIDI controller numbers
(def ^:private mus-controllers
  [0x00 0x20 0x01 0x07 0x0A 0x0B 0x5B 0x5D 0x40 0x43 0x78 0x7B 0x7E 0x7F 0x79])

;; MUS channel 15 → MIDI channel 9 (percussion)
(defn- mus-channel->midi [^long ch]
  (cond (< ch 9) ch
        (= ch 9) 10
        (< ch 15) ch
        :else 9))

;; ================================================================
;; MUS parser
;; ================================================================

(defn- read-mus-delay
  "Read variable-length delay from MUS data. Returns [delay new-offset]."
  [^bytes data ^long offset]
  (loop [off offset delay 0]
    (let [b (bit-and (aget data off) 0xFF)]
      (if (not= (bit-and b 0x80) 0)
        (recur (inc off) (bit-or (bit-shift-left delay 7) (bit-and b 0x7F)))
        [(bit-or (bit-shift-left delay 7) b) (inc off)]))))

;; ================================================================
;; MIDI writer helpers
;; ================================================================

(defn- write-var-len
  "Write a MIDI variable-length quantity."
  [^ByteArrayOutputStream out ^long value]
  (if (zero? value)
    (.write out 0)
    (let [buf (byte-array 4)
          n (loop [v value i 0]
              (if (zero? v)
                i
                (do (aset buf i (byte (bit-and v 0x7F)))
                    (recur (unsigned-bit-shift-right v 7) (inc i)))))]
      ;; Write in reverse order, setting continuation bits
      (loop [i (dec n)]
        (when (>= i 0)
          (let [b (aget buf i)]
            (.write out (if (pos? i)
                          (bit-or (bit-and b 0xFF) 0x80)
                          (bit-and b 0xFF))))
          (recur (dec i)))))))

(defn- write-u16-be [^ByteArrayOutputStream out ^long v]
  (.write out (bit-and (unsigned-bit-shift-right v 8) 0xFF))
  (.write out (bit-and v 0xFF)))

(defn- write-u32-be [^ByteArrayOutputStream out ^long v]
  (.write out (bit-and (unsigned-bit-shift-right v 24) 0xFF))
  (.write out (bit-and (unsigned-bit-shift-right v 16) 0xFF))
  (.write out (bit-and (unsigned-bit-shift-right v 8) 0xFF))
  (.write out (bit-and v 0xFF)))

;; ================================================================
;; MUS → MIDI conversion
;; ================================================================

(defn mus->midi
  "Convert MUS data bytes to MIDI data bytes.
  mus-data: raw bytes of the MUS lump (including header)."
  [^bytes mus-data]
  (let [;; Parse MUS header
        _score-len (bit-or (bit-and (aget mus-data 4) 0xFF)
                           (bit-shift-left (bit-and (aget mus-data 5) 0xFF) 8))
        score-start (bit-or (bit-and (aget mus-data 6) 0xFF)
                            (bit-shift-left (bit-and (aget mus-data 7) 0xFF) 8))
        ;; Track volume state per channel
        channel-vols (long-array 16 127)
        ;; Output MIDI track data
        track-out (ByteArrayOutputStream. 8192)]
    ;; Convert events
    (loop [off (int score-start)]
      (when (< off (alength mus-data))
        (let [event-byte (bit-and (aget mus-data off) 0xFF)
              event-type (bit-and (unsigned-bit-shift-right event-byte 4) 0x07)
              channel (bit-and event-byte 0x0F)
              last? (not= (bit-and event-byte 0x80) 0)
              midi-ch (mus-channel->midi channel)
              off (inc off)]
          (case (int event-type)
            ;; 0: Release note
            0 (let [note (bit-and (aget mus-data off) 0x7F)]
                (write-var-len track-out 0)
                (.write track-out (bit-or 0x80 midi-ch))
                (.write track-out note)
                (.write track-out 0)
                (let [off (inc off)
                      [delay off] (if last? (read-mus-delay mus-data off) [0 off])]
                  (when (pos? delay) (write-var-len track-out delay))
                  (recur off)))

            ;; 1: Play note
            1 (let [b (bit-and (aget mus-data off) 0xFF)
                    note (bit-and b 0x7F)
                    has-vol? (not= (bit-and b 0x80) 0)
                    off (inc off)
                    [vol off] (if has-vol?
                                [(bit-and (aget mus-data off) 0x7F) (inc off)]
                                [(aget channel-vols midi-ch) off])]
                (when has-vol? (aset channel-vols midi-ch (long vol)))
                (write-var-len track-out 0)
                (.write track-out (bit-or 0x90 midi-ch))
                (.write track-out note)
                (.write track-out (int vol))
                (let [[delay off] (if last? (read-mus-delay mus-data off) [0 off])]
                  (when (pos? delay) (write-var-len track-out delay))
                  (recur off)))

            ;; 2: Pitch wheel
            2 (let [pw (bit-and (aget mus-data off) 0xFF)
                    midi-pw (* pw 64)] ;; MUS 0-255 → MIDI 0-16383
                (write-var-len track-out 0)
                (.write track-out (bit-or 0xE0 midi-ch))
                (.write track-out (bit-and midi-pw 0x7F))
                (.write track-out (bit-and (unsigned-bit-shift-right midi-pw 7) 0x7F))
                (let [off (inc off)
                      [delay off] (if last? (read-mus-delay mus-data off) [0 off])]
                  (when (pos? delay) (write-var-len track-out delay))
                  (recur off)))

            ;; 3: System event (ignored for playback)
            3 (let [off (inc off)
                    [delay off] (if last? (read-mus-delay mus-data off) [0 off])]
                (when (pos? delay) (write-var-len track-out delay))
                (recur off))

            ;; 4: Controller change
            4 (let [ctrl-num (bit-and (aget mus-data off) 0xFF)
                    ctrl-val (bit-and (aget mus-data (inc off)) 0xFF)
                    off (+ off 2)]
                (if (zero? ctrl-num)
                  ;; Controller 0 = instrument change
                  (do (write-var-len track-out 0)
                      (.write track-out (bit-or 0xC0 midi-ch))
                      (.write track-out (bit-and ctrl-val 0x7F)))
                  ;; Regular controller
                  (let [midi-ctrl (if (< ctrl-num (count mus-controllers))
                                    (nth mus-controllers ctrl-num)
                                    ctrl-num)]
                    (write-var-len track-out 0)
                    (.write track-out (bit-or 0xB0 midi-ch))
                    (.write track-out (bit-and midi-ctrl 0x7F))
                    (.write track-out (bit-and ctrl-val 0x7F))))
                (let [[delay off] (if last? (read-mus-delay mus-data off) [0 off])]
                  (when (pos? delay) (write-var-len track-out delay))
                  (recur off)))

            ;; 5: End of measure (ignored)
            5 (let [[delay off] (if last? (read-mus-delay mus-data off) [0 off])]
                (when (pos? delay) (write-var-len track-out delay))
                (recur off))

            ;; 6: Score end
            6 nil

            ;; Default: skip
            (let [[delay off] (if last? (read-mus-delay mus-data off) [0 off])]
              (when (pos? delay) (write-var-len track-out delay))
              (recur off))))))

    ;; End of track event
    (write-var-len track-out 0)
    (.write track-out 0xFF)
    (.write track-out 0x2F)
    (.write track-out 0x00)

    ;; Build complete MIDI file
    (let [track-data (.toByteArray track-out)
          midi-out (ByteArrayOutputStream. (+ 22 (alength track-data)))]
      ;; MIDI header: MThd
      (.write midi-out (.getBytes "MThd"))
      (write-u32-be midi-out 6) ;; header length
      (write-u16-be midi-out 0) ;; format 0
      (write-u16-be midi-out 1) ;; 1 track
      (write-u16-be midi-out 140) ;; 140 ticks per quarter (Doom ≈ 140Hz)
      ;; Track header: MTrk
      (.write midi-out (.getBytes "MTrk"))
      (write-u32-be midi-out (alength track-data))
      (.write midi-out track-data)
      (.toByteArray midi-out))))

;; ================================================================
;; Playback
;; ================================================================

(defn play-music!
  "Load and play MUS music from WAD. Returns Sequencer (for later stop) or nil."
  [wad-data ^String lump-name]
  (try
    (when-let [lump (wad/find-lump wad-data lump-name)]
      (let [buf (:buf wad-data)
            offset (long (:offset lump))
            size (long (:size lump))
            mus-bytes (byte-array size)
            _ (let [b ^java.nio.ByteBuffer (.duplicate buf)]
                (.position b (int offset))
                (.get b mus-bytes))
            ;; Verify MUS header
            _ (when-not (and (= (aget mus-bytes 0) (byte 0x4D))
                             (= (aget mus-bytes 1) (byte 0x55))
                             (= (aget mus-bytes 2) (byte 0x53))
                             (= (aget mus-bytes 3) (byte 0x1A)))
                (throw (ex-info "Not a MUS file" {:header [(aget mus-bytes 0)
                                                            (aget mus-bytes 1)]})))
            midi-bytes (mus->midi mus-bytes)
            stream (ByteArrayInputStream. midi-bytes)
            sequence (MidiSystem/getSequence stream)
            ^Sequencer sequencer (MidiSystem/getSequencer)]
        (.open sequencer)
        (.setSequence sequencer sequence)
        (.setLoopCount sequencer Sequencer/LOOP_CONTINUOUSLY)
        (.start sequencer)
        (println "Music playing:" lump-name)
        sequencer))
    (catch Exception e
      (println "Music playback failed:" (.getMessage e))
      nil)))

(defn stop-music!
  "Stop music playback."
  [^Sequencer sequencer]
  (when sequencer
    (try
      (.stop sequencer)
      (.close sequencer)
      (catch Exception _))))

(defn set-music-volume!
  "Set music volume. vol is 0-15 (Doom scale)."
  [^Sequencer sequencer ^long vol]
  (when sequencer
    (try
      (let [midi-vol (int (* (/ (double (max 0 (min vol 15))) 15.0) 127.0))
            synth (when (instance? javax.sound.midi.Synthesizer sequencer)
                    sequencer)
            channels (if synth
                       (.getChannels ^javax.sound.midi.Synthesizer synth)
                       ;; For default sequencer, get synthesizer
                       (when-let [^javax.sound.midi.Synthesizer s (javax.sound.midi.MidiSystem/getSynthesizer)]
                         (when-not (.isOpen s) (.open s))
                         (.getChannels s)))]
        (when channels
          (dotimes [i (alength channels)]
            (when-let [^javax.sound.midi.MidiChannel ch (aget channels i)]
              (.controlChange ch 7 midi-vol))))) ;; CC 7 = channel volume
      (catch Exception _))))
