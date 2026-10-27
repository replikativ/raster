(ns raster.compiler.backend.jvm.field-read-emission-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.types :as types]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

;; A generic nominal class, not a distribution-specific AD fixture.
(deftype FieldReadFixture [^double coefficient ^float gain payload])

(defn- emit-field [access field]
  (let [class-tag (symbol (.getName FieldReadFixture))
        receiver (with-meta 'value {:tag class-tag})
        node (case access
               :dot (list '. receiver (symbol (str "-" field)))
               :keyword (list (keyword field) 'value))
        body [node]
        ;; A raw Object parameter exercises the actual member-local CHECKCAST.
        ;; The keyword fast path instead selects its field from the local hint.
        tags (if (= :dot access) '[Object] [class-tag])
        interface (types/ensure-fn-interface! tags)
        emitted (core/emit-owned-bytecode!
                 bytecode/compile-typed-impl! "raster.compiled.FieldReadObservation"
                 '[value] tags 'Object body
                 (the-ns 'raster.compiler.backend.jvm.field-read-emission-test)
                 (:iface-name interface) 'field-read-observation)
        emission (:emission emitted)
        original (first (filter #(bytecode/retained-field-read emission %)
                                (vals (:invocation-catalog emission))))]
    (assoc emitted :input-body body :source node :original original
           :observation (when original (bytecode/retained-field-read emission original)))))

(deftest actual-primitive-fields-share-dot-and-keyword-read-evidence
  (doseq [access [:dot :keyword]
          [field carrier value] [["coefficient" Double/TYPE 2.5]
                                 ["gain" Float/TYPE (float 0.75)]]]
    (let [{:keys [implementation observation]} (emit-field access field)
          ^java.lang.reflect.Field selected (:field observation)
          owner (:owner-class observation)
          instance (FieldReadFixture. 2.5 (float 0.75) :opaque)]
      (is (some? observation))
      (is (instance? java.lang.reflect.Field selected))
      (is (identical? FieldReadFixture owner))
      (is (identical? FieldReadFixture (:declaring-class observation)))
      (is (identical? owner (Class/forName (.getName ^Class owner) false
                                         (.getClassLoader ^Class owner))))
      (is (identical? carrier (.getType selected)))
      (is (= field (.getName selected) (:field-name observation)))
      (is (= (.descriptorString carrier)
             (.descriptorString ^java.lang.constant.ClassDesc (:field-descriptor observation))))
      (is (identical? carrier (:raw-result-carrier observation)))
      (is (true? (:closed-result? observation)))
      (is (= {:kind :identity :result-carrier carrier
              :result-type (if (identical? Double/TYPE carrier) :double :float)}
             (:final-completion observation)))
      (is (= #{'value} (set (keys (get-in observation [:receiver-input :ports])))))
      (is (= :ref (:receiver-type observation)))
      (is (= value (implementation instance)))
      (is (thrown? NullPointerException (implementation nil)))
      (is (thrown? ClassCastException (implementation (Object.)))))))

(deftest raw-object-receiver-retains-its-actual-member-checkcast
  (let [{:keys [observation]} (emit-field :dot "coefficient")]
    (is (= :checkcast (:receiver-adaptation observation)))
    (is (identical? FieldReadFixture (:receiver-checkcast-class observation)))
    (is (identical? Object (get-in observation [:receiver-input :ports 'value :reference-class])))
    (is (false? (get-in observation [:receiver-input :ports 'value :verified]))))
  (let [{:keys [observation]} (emit-field :keyword "coefficient")]
    (is (= :identity (:receiver-adaptation observation)))
    (is (nil? (:receiver-checkcast-class observation)))
    (is (true? (get-in observation [:receiver-local :verified])))
    (is (identical? FieldReadFixture (get-in observation [:receiver-local :reference-class])))))

(deftest object-field-observation-does-not-certify-registry-postprocessing
  ;; The real dot emitter unboxes this Object field through Number.doubleValue.
  ;; Observing GETFIELD must not mistake that registry-selected instruction for
  ;; a closed primitive field read, or reacquire the registry during later use.
  (with-redefs [inference/field-type-registry
                (atom {'FieldReadFixture {"payload" 'double}})]
    (let [{:keys [implementation observation]} (emit-field :dot "payload")]
      (is (identical? Object (:raw-result-carrier observation)))
      (is (= :ref (:raw-result-type observation)))
      (is (false? (:closed-result? observation)))
      (is (not (contains? observation :final-completion)))
      (is (= 3.0 (implementation (FieldReadFixture. 0.0 (float 0.0) 3.0))))))
  (let [{:keys [implementation observation]} (emit-field :keyword "payload")
        payload (Object.)]
    (is (identical? Object (:raw-result-carrier observation)))
    (is (false? (:closed-result? observation)))
    (is (identical? payload (implementation (FieldReadFixture. 0.0 (float 0.0) payload))))))

(deftest field-selection-stays-with-the-exact-occurrence-and-historical-owner
  (let [{:keys [emission input-body source original]} (emit-field :dot "coefficient")
        foreign (emit-field :dot "coefficient")
        token (:raster.compiler.backend.jvm.bytecode/call-occurrence (meta original))
        input (bytecode/retained-input-field-read emission input-body source)]
    (is (some? (bytecode/retained-field-read emission original)))
    (is (identical? source (:source-input input)))
    (is (identical? (:field input) (:field (bytecode/retained-field-read emission original))))
    (is (nil? (bytecode/retained-field-read emission
                                          (with-meta (apply list original) (meta original)))))
    (is (nil? (bytecode/retained-field-read (:emission foreign) original)))
    (is (nil? (bytecode/retained-input-field-read emission (mapv identity input-body) source)))
    (is (nil? (bytecode/retained-input-field-read emission input-body
                                                (with-meta (apply list source) (meta source)))))
    (is (nil? (bytecode/retained-field-read
               (assoc-in emission [:field-read-contexts token :status] :ambiguous) original)))
    (is (= :unstable-emission-inputs
           (try (bytecode/retained-field-read
                 (update emission :completed-definition-revision inc) original)
                nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
