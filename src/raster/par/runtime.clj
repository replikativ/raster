(ns raster.par.runtime
  "CPU thread-parallel execution for raster's side-effecting parallel loops.

  `parallel-for!` runs a chunked index range across a dedicated ForkJoinPool
  sized by `*par-threads*`. The compiler lowers parallel loops to a single
  `parallel-for!` call (non-parametric emit); the thread count is a runtime
  policy here, NOT baked into compiled code.

  Intended for side-effecting bodies with benign races (SGD-style embedding
  updates): chunks are contiguous, so on a head-sorted edge list each thread
  owns a disjoint vertex range and RNG/embedding collisions are rare. Bind
  `*par-threads*` to 1 for the serial, deterministic path (tests / baselines)."
  (:import [java.util.concurrent ForkJoinPool Callable Future]
           [java.util ArrayList]))

(def ^:dynamic *par-threads*
  "Thread count for parallel-for!. nil ⇒ availableProcessors; 1 ⇒ serial
  (deterministic). Bind to match a benchmark's core count or to coexist with
  BLAS threads (OPENBLAS_NUM_THREADS)."
  nil)

(defn- default-threads ^long []
  (.availableProcessors (Runtime/getRuntime)))

(defonce ^:private pools (atom {}))

(defn- pool-for ^ForkJoinPool [^long p]
  (or (get @pools p)
      (get (swap! pools (fn [m] (if (contains? m p) m (assoc m p (ForkJoinPool. (int p)))))) p)))

(defn parallel-for!
  "Run (body-fn lo hi) over contiguous chunks covering [0,n) across
  *par-threads* threads (one chunk per thread). Blocks until all chunks
  complete; propagates the first task exception. Returns nil.

  Serial fast path (*par-threads*=1, or n<=1, or one chunk): a single
  (body-fn 0 n) call on the calling thread — identical to a sequential loop."
  [n body-fn]
  (let [n (long n)
        p (long (or *par-threads* (default-threads)))]
    (if (or (<= p 1) (<= n 1))
      (do (body-fn 0 n) nil)
      (let [chunk (long (quot (+ n (dec p)) p))          ; ceil(n/p)
            tasks (ArrayList.)]
        (loop [lo 0]
          (when (< lo n)
            (let [lo* lo
                  hi  (min n (+ lo chunk))]
              (.add tasks ^Callable (reify Callable (call [_] (body-fn lo* hi) nil)))
              (recur (+ lo chunk)))))
        (let [futures (.invokeAll (pool-for p) tasks)]
          (doseq [^Future f futures] (.get f)))          ; surface task exceptions
        nil))))
