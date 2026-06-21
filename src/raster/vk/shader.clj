(ns raster.vk.shader
  "GLSL → SPIR-V compilation via LWJGL shaderc, and shader module creation."
  (:import
   [org.lwjgl.util.shaderc Shaderc]
   [org.lwjgl.vulkan VK13 VkDevice VkShaderModuleCreateInfo]
   [org.lwjgl.system MemoryStack MemoryUtil]
   [java.nio ByteBuffer]))

(set! *warn-on-reflection* true)

;; ================================================================
;; GLSL → SPIR-V
;; ================================================================

(def ^:private stage-map
  {:vertex   Shaderc/shaderc_vertex_shader
   :fragment Shaderc/shaderc_fragment_shader
   :compute  Shaderc/shaderc_compute_shader
   :geometry Shaderc/shaderc_geometry_shader})

(defn compile-glsl
  "Compile a GLSL source string to SPIR-V bytes.
  stage: :vertex, :fragment, :compute, :geometry
  Returns a direct ByteBuffer containing SPIR-V binary.
  Throws on compilation error with error message."
  ^ByteBuffer [^String source stage & {:keys [filename optimize?]
                                       :or {filename "shader.glsl" optimize? true}}]
  (let [compiler (Shaderc/shaderc_compiler_initialize)
        options (Shaderc/shaderc_compile_options_initialize)]
    (try
      (when optimize?
        (Shaderc/shaderc_compile_options_set_optimization_level
         options Shaderc/shaderc_optimization_level_performance))
      (Shaderc/shaderc_compile_options_set_target_env
       options Shaderc/shaderc_target_env_vulkan Shaderc/shaderc_env_version_vulkan_1_3)
      (let [kind (get stage-map stage)
            _ (when-not kind
                (throw (ex-info (str "Unknown shader stage: " stage)
                                {:stage stage :valid (keys stage-map)})))
            result (Shaderc/shaderc_compile_into_spv
                    compiler source (int kind) ^String filename "main" (long options))
            status (Shaderc/shaderc_result_get_compilation_status result)]
        (when-not (= status Shaderc/shaderc_compilation_status_success)
          (let [msg (Shaderc/shaderc_result_get_error_message result)]
            (Shaderc/shaderc_result_release result)
            (throw (ex-info (str "GLSL compilation failed: " msg)
                            {:status status :stage stage}))))
        (let [spirv-bytes (Shaderc/shaderc_result_get_bytes result)
              ;; Copy to our own buffer since result will be released
              copy (MemoryUtil/memAlloc (.remaining spirv-bytes))]
          (.put copy spirv-bytes)
          (.flip copy)
          (Shaderc/shaderc_result_release result)
          copy))
      (finally
        (Shaderc/shaderc_compile_options_release options)
        (Shaderc/shaderc_compiler_release compiler)))))

;; ================================================================
;; Shader module
;; ================================================================

(defn create-shader-module
  "Create a VkShaderModule from SPIR-V ByteBuffer.
  Returns the shader module handle (long)."
  ^long [^VkDevice device ^ByteBuffer spirv]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [ci (doto (VkShaderModuleCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
                 (.pCode spirv))
            p-module (.mallocLong stack 1)
            result (VK13/vkCreateShaderModule device ci nil p-module)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create shader module" {:result result})))
        (.get p-module 0))
      (finally
        (MemoryStack/stackPop)))))

(defn destroy-shader-module! [^VkDevice device ^long module]
  (VK13/vkDestroyShaderModule device module nil))

(defn compile-shader-module
  "Convenience: compile GLSL string and create shader module in one step.
  Returns {:module long :spirv ByteBuffer}.
  Caller should free spirv with MemoryUtil/memFree when done."
  [^VkDevice device ^String glsl-source stage]
  (let [spirv (compile-glsl glsl-source stage)
        module (create-shader-module device spirv)]
    {:module module :spirv spirv}))
