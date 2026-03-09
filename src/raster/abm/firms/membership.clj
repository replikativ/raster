(ns raster.abm.firms.membership
  "CSR membership management and decision execution for the Firms ABM.

   All operations use deftm for walker type inference and
   zero-overhead primitive dispatch."
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength aclone]]))

;; ================================================================
;; CSR Rebuild (deftm)
;; ================================================================

(deftm rebuild-csr!
  "Rebuild CSR membership from current-firm assignments.
   Returns count of alive firms."
  [current-firm :- (Array int), firm-size :- (Array int),
   alive :- (Array int), members :- (Array int),
   member-offsets :- (Array int), n :- Long, max-firms :- Long] :- Long
  (let [pos (int-array max-firms)]
    ;; Pass 1: Zero sizes and count per firm
    (dotimes [f max-firms]
      (aset firm-size f (int 0)))
    (dotimes [i n]
      (let [f (aget current-firm i)]
        (when (>= f 0)
          (aset firm-size f (int (inc (aget firm-size f)))))))
    ;; Prefix sum + alive update
    (aset member-offsets 0 (int 0))
    (let [n-alive
          (loop [f (int 0) cnt (int 0)]
            (if (>= f max-firms)
              (long cnt)
              (let [sz (aget firm-size f)
                    offset (aget member-offsets f)]
                (aset member-offsets (inc f) (int (+ offset sz)))
                (aset pos f (int offset))
                (if (> sz 0)
                  (do (aset alive f (int 1))
                      (recur (unchecked-add-int f 1) (unchecked-add-int cnt 1)))
                  (do (aset alive f (int 0))
                      (recur (unchecked-add-int f 1) cnt))))))]
      ;; Pass 2: Fill members
      (dotimes [i n]
        (let [f (aget current-firm i)]
          (when (>= f 0)
            (let [p (aget pos f)]
              (aset members p (int i))
              (aset pos f (int (inc p)))))))
      n-alive)))

;; ================================================================
;; Queue operations (deftm)
;; ================================================================

(deftm enqueue!
  "Add a decision to the queue. Mutates count in place."
  [agent-ids :- (Array int), decision-types :- (Array int),
   target-firms :- (Array int), new-efforts :- (Array float),
   count-arr :- (Array int),
   agent-id :- Long, dtype :- Long, target :- Long, effort :- Float] :- Long
  (let [c (aget count-arr 0)]
    (aset agent-ids c (int agent-id))
    (aset decision-types c (int dtype))
    (aset target-firms c (int target))
    (aset new-efforts c effort)
    (aset count-arr 0 (int (inc c)))
    (long (inc c))))

;; ================================================================
;; Decision Execution (deftm, serial)
;; ================================================================

(deftm allocate-firm-slot!
  "Find a dead firm slot and initialize it.
   Returns firm index or -1 if full."
  [alive :- (Array int), param-a :- (Array float), param-b :- (Array float),
   param-beta :- (Array float), total-effort :- (Array float),
   output :- (Array float), firm-size :- (Array int),
   max-firms :- Long, a :- Double, b :- Double, beta :- Double] :- Long
  (loop [f (int 0)]
    (cond
      (>= f max-firms) -1
      (== (aget alive f) 0)
      (do (aset alive f (int 1))
          (aset param-a f (float a))
          (aset param-b f (float b))
          (aset param-beta f (float beta))
          (aset total-effort f (float 0.0))
          (aset output f (float 0.0))
          (aset firm-size f (int 0))
          (long f))
      :else (recur (unchecked-add-int f 1)))))

(deftm execute-decisions-impl!
  "Execute queued decisions. Returns number of newly created firms."
  [agent-ids :- (Array int), decision-types :- (Array int),
   target-firms :- (Array int), new-efforts :- (Array float),
   cnt :- Long,
   effort :- (Array float), current-firm :- (Array int),
   total-effort :- (Array float), firm-size :- (Array int),
   alive :- (Array int),
   param-a :- (Array float), param-b :- (Array float),
   param-beta :- (Array float), output :- (Array float),
   max-firms :- Long, rng-seeds :- (Array long)] :- Long
  (loop [q (int 0) new-firms (int 0)]
    (if (>= q cnt)
      (long new-firms)
      (let [ai    (aget agent-ids q)
            dtype (aget decision-types q)
            tgt   (aget target-firms q)
            new-e (double (aget new-efforts q))
            old-fi (aget current-firm ai)
            old-e  (double (aget effort ai))]
        (case (int dtype)
          ;; STAY
          0 (do
              (aset effort ai (float new-e))
              (when (>= old-fi 0)
                (aset total-effort old-fi
                      (float (+ (double (aget total-effort old-fi))
                                (- new-e old-e)))))
              (recur (unchecked-add-int q 1) new-firms))

          ;; SWITCH
          1 (do
              (when (and (>= tgt 0) (== 1 (aget alive tgt)))
                ;; Leave old
                (when (>= old-fi 0)
                  (aset total-effort old-fi
                        (float (- (double (aget total-effort old-fi)) old-e)))
                  (aset firm-size old-fi (int (dec (aget firm-size old-fi))))
                  (when (<= (aget firm-size old-fi) 0)
                    (aset alive old-fi (int 0))))
                ;; Join target
                (aset current-firm ai (int tgt))
                (aset effort ai (float new-e))
                (aset total-effort tgt
                      (float (+ (double (aget total-effort tgt)) new-e)))
                (aset firm-size tgt (int (inc (aget firm-size tgt)))))
              (recur (unchecked-add-int q 1) new-firms))

          ;; STARTUP — leave old firm FIRST (frees its slot), then allocate.
          ;; Matches C11 order: LeaveFirm() before FoundFirm()/NewFirm().
          ;; This allows max-firms = n (not 2n): the vacated singleton slot
          ;; is immediately available for the new firm.
          2 (let [seed (aget rng-seeds ai)
                  a-new (+ 0.1 (* 1.4 (/ (double (bit-and seed 0xFFFF)) 65536.0)))
                  b-new (+ 0.1 (* 1.4 (/ (double (bit-and (unsigned-bit-shift-right seed 16) 0xFFFF)) 65536.0)))
                  beta-new (+ 1.0 (* 0.99 (/ (double (bit-and (unsigned-bit-shift-right seed 32) 0xFFFF)) 65536.0)))
                  ;; Leave old firm first — frees its slot if it was a singleton
                  _ (when (>= old-fi 0)
                      (aset total-effort old-fi
                            (float (- (double (aget total-effort old-fi)) old-e)))
                      (aset firm-size old-fi (int (dec (aget firm-size old-fi))))
                      (when (<= (aget firm-size old-fi) 0)
                        (aset alive old-fi (int 0))))
                  new-fi (allocate-firm-slot! alive param-a param-b param-beta
                                              total-effort output firm-size
                                              max-firms a-new b-new beta-new)]
              (if (>= new-fi 0)
                (do
                  ;; Join new firm
                  (aset current-firm ai (int new-fi))
                  (aset effort ai (float new-e))
                  (aset total-effort new-fi (float new-e))
                  (aset firm-size new-fi (int 1))
                  (recur (unchecked-add-int q 1) (unchecked-add-int new-firms 1)))
                (do
                  ;; Slot exhausted — agent stays unaffiliated (current-firm = old-fi already freed)
                  ;; In practice never happens when max-firms >= n
                  (aset current-firm ai (int -1))
                  (recur (unchecked-add-int q 1) new-firms)))))))))
