(ns raster.tree
  "Generic structural operations on HMap/HVec values with leaf wrappers
  (Param/Frozen).

  Operates on any value that has a TC spec (HMap/HVec). Knows nothing about
  trainability — that lives in the leaf wrappers (raster.compiler.core.params).

  ## Two views

  A spec-shaped value has two views:

  - **Structured view**: nested map/vec at runtime. Read via `(:k x)` / `(nth x i)`.
  - **Flat view**: a vector of leaf values in canonical (sorted-key, ascending-index)
    order. Read via `(nth x flat-idx)`.

  The conversion is bijective and depends only on the spec.

  ## Operations

  - `flatten` / `unflatten` — runtime view conversion.
  - `leaves` — leaf descriptors (path + kind + type) for a spec.
  - `flat-view` — declare a binding is a flat view (compile-time tag, no runtime
    cost). Used by the pre-flatten pass to resolve subsequent path access on the
    binding to flat-index lookups.
  - `walk!` — in-place per-leaf operation across K parallel structured trees.
    Compile-time unrolled to one named-deftm call per leaf.
  - `scan-vec` — sequential fold over a known-length HVec sub-tree. Compile-time
    unrolled to a let-chain.

  These are macros at the source level. The pre-flatten pass recognizes them
  and expands using compile-time spec info."
  (:refer-clojure :exclude [flatten])
  (:require [raster.compiler.core.params-flatten :as pf]))

;; ----------------------------------------------------------------------
;; Runtime view conversion (delegate to params-flatten internals)
;; ----------------------------------------------------------------------

(defn leaves
  "Return canonical leaf descriptors for a spec. Each entry:
  {:path [keys/idxs] :type T :kind :param/:frozen/:plain}."
  [spec]
  (pf/flatten-spec spec))

(defn flatten
  "Walk a structured value matching spec; return the flat vector of leaves
  in canonical order."
  [spec value]
  (pf/flatten-value spec value))

(defn unflatten
  "Given spec and a flat vector of leaves, reconstruct the structured value."
  [spec flat-leaves]
  (pf/unflatten-value spec flat-leaves))

;; ----------------------------------------------------------------------
;; Compile-time forms — recognized by the pre-flatten pass.
;; The macro expansions are runtime fallbacks for cases where the spec
;; isn't compile-time visible (e.g. when used outside a defmodel body).
;; ----------------------------------------------------------------------

(defmacro flat-view
  "Declare that `flat-vec-form` is a flat-view of a known tree spec.

  Inside a defmodel body, the pre-flatten pass recognizes this and records
  the binding in its env so subsequent `(:k binding)` / `(nth binding i)`
  access resolves to `(nth flat-vec-form <flat-idx>)` at compile time —
  no Map allocation.

  Outside a defmodel body, this is a runtime pass-through returning the
  flat vector unchanged.

  Options:
    :spec-of var          -- the var whose first tree arg's spec is the source
    :spec spec-form       -- explicit spec literal (alternative to :spec-of)
    :starting-at idx      -- skip the first idx entries (default 0).
                             For value+grad output, use :starting-at 1.

  Examples:
    (let [vg ((p/value+grad #'gsdm-loss) w x y)
          grads (tree/flat-view vg :spec-of #'gsdm-loss :starting-at 1)]
      (tree/walk! :param adam-step! [w grads m v] lr ...))"
  [flat-vec-form & _opts]
  ;; Runtime-only expansion. The pre-flatten pass intercepts this form
  ;; before macro expansion and handles it specially.
  flat-vec-form)

(defmacro walk!
  "In-place per-leaf application of f across K parallel structured trees,
  filtered by leaf kind. Compile-time unrolled via the pre-flatten pass.

  f is a NAMED deftm var. Per leaf path P matching kind, emits
    (f tree-args[0]@P tree-args[1]@P ... tree-args[K-1]@P extras...)

  Outside a defmodel body, falls back to a runtime walk via
  `flatten`/spec lookup of the first tree-arg.

  Args:
    kind         -- :param, :frozen, :plain, or :any
    f            -- a deftm function (passed as a var, e.g. #'adam-step!,
                    or as a bare symbol resolvable in scope)
    tree-args  -- vector of K tree expressions, each of compatible shape
                    (the first determines structure)
    extras       -- additional positional args passed to every f call

  Example:
    (tree/walk! :param adam-leaf-step! [w grads m v] lr beta1 beta2 eps t)"
  [kind f tree-args & extras]
  ;; Runtime fallback: walk via the FIRST tree-arg's spec at runtime.
  ;; Slow but correct. The pre-flatten pass replaces this with a compile-time
  ;; unrolled per-leaf call sequence when the spec is statically known.
  (let [first-pt (first tree-args)
        spec-sym (gensym "spec__")
        leaf-descs-sym (gensym "leaves__")
        ld-sym (gensym "ld__")]
    `(let [~spec-sym (or (some-> ~first-pt meta :raster.tree/spec)
                         (throw (ex-info "tree/walk!: cannot determine spec at runtime; use inside a defmodel body or attach :raster.tree/spec metadata"
                                         {:value ~first-pt})))
           ~leaf-descs-sym (raster.tree/leaves ~spec-sym)]
       (doseq [~ld-sym ~leaf-descs-sym
               :when (or (= ~kind :any) (= (:kind ~ld-sym) ~kind))]
         (let [path# (:path ~ld-sym)]
           (~f ~@(map (fn [pt]
                        `(reduce (fn [v# step#]
                                   (if (keyword? step#) (get v# step#) (nth v# step#)))
                                 ~pt path#))
                      tree-args)
               ~@extras))))))

(defmacro scan-vec
  "Sequential fold over a known-length HVec sub-tree. Compile-time unrolled
  via the pre-flatten pass.

  f is a NAMED deftm taking (acc, *<leaves-of-element>, *<extras>) -> Acc.
  Compile-time emits a let-chain of N applications, where N is the HVec
  length from the spec.

  Outside a defmodel body, falls back to a runtime reduce.

  Args:
    f         -- a deftm function
    init      -- initial accumulator
    xs        -- expression that resolves to an HVec sub-tree
    extras    -- additional positional args passed to every f call

  Example:
    (tree/scan-vec transformer-block-step h-init (:layers w)
                   temb seq-len d-model n-head)"
  [f init xs & extras]
  ;; Runtime fallback: reduce by calling f with element flattened.
  (let [acc-sym (gensym "acc__")
        elt-sym (gensym "elt__")]
    `(reduce (fn [~acc-sym ~elt-sym]
               ;; Caller must arrange for f to accept an unflattened element
               ;; OR for the structured element to be accepted directly.
               ;; Runtime fallback assumes f takes (acc, elt, extras...).
               (~f ~acc-sym ~elt-sym ~@extras))
             ~init
             ~xs)))
