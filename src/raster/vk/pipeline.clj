(ns raster.vk.pipeline
  "Vulkan graphics pipeline creation using dynamic rendering (Vulkan 1.3).
  No VkRenderPass/VkFramebuffer objects needed."
  (:require [raster.vk.shader :as shader])
  (:import
   [org.lwjgl.vulkan
    VK13 VkDevice VkCommandBuffer
    VkGraphicsPipelineCreateInfo
    VkPipelineShaderStageCreateInfo
    VkPipelineVertexInputStateCreateInfo
    VkPipelineInputAssemblyStateCreateInfo
    VkPipelineViewportStateCreateInfo
    VkPipelineRasterizationStateCreateInfo
    VkPipelineMultisampleStateCreateInfo
    VkPipelineDepthStencilStateCreateInfo
    VkPipelineColorBlendStateCreateInfo
    VkPipelineColorBlendAttachmentState
    VkPipelineDynamicStateCreateInfo
    VkPipelineLayoutCreateInfo
    VkPipelineRenderingCreateInfo
    VkViewport VkRect2D VkOffset2D VkExtent2D
    VkVertexInputBindingDescription VkVertexInputAttributeDescription
    VkPushConstantRange]
   [org.lwjgl.system MemoryStack MemoryUtil]
   [java.nio ByteBuffer]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Pipeline layout
;; ================================================================

(defn create-graphics-pipeline-layout
  "Create a pipeline layout with optional push constants.
  push-constants: {:size int :stages int} or nil."
  [^VkDevice device push-constants]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [ci (doto (VkPipelineLayoutCreateInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO))
            _ (when push-constants
                (let [buf (VkPushConstantRange/calloc 1 stack)
                      ^VkPushConstantRange pc (.get buf 0)]
                  (doto pc
                    (.stageFlags (int (:stages push-constants)))
                    (.offset 0)
                    (.size (int (:size push-constants))))
                  (.pPushConstantRanges ci buf)))
            p-layout (.mallocLong stack 1)
            result (VK13/vkCreatePipelineLayout device ci nil p-layout)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create pipeline layout" {:result result})))
        (.get p-layout 0))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Graphics pipeline with dynamic rendering
;; ================================================================

(defn create-graphics-pipeline
  "Create a graphics pipeline with dynamic rendering.
  config map:
    :vert-glsl    - vertex shader GLSL string
    :frag-glsl    - fragment shader GLSL string
    :topology     - :triangles (default) or :lines
    :color-format - swapchain color format (int)
    :depth-format - depth format (int), 0 for no depth
    :depth-test?  - enable depth test (default: true when depth-format > 0)
    :depth-write? - enable depth write (default: true when depth-format > 0)
    :push-constants - {:size int :stages int} or nil
    :vertex-bindings - [{:binding int :stride int}] or nil (vertex pulling)
    :vertex-attributes - [{:location int :binding int :format int :offset int}] or nil
    :cull-mode    - :none (default), :back, :front
    :polygon-mode - :fill (default), :line
    :blend?       - enable alpha blending (default false)
    :layout       - pre-created pipeline layout (long), skips auto-creation
  Returns {:pipeline long :layout long :vert-module long :frag-module long
           :vert-spirv ByteBuffer :frag-spirv ByteBuffer}."
  [^VkDevice device config]
  (let [{:keys [vert-glsl frag-glsl topology color-format depth-format
                depth-test? depth-write?
                push-constants vertex-bindings vertex-attributes
                cull-mode polygon-mode blend? layout]
         :or {topology :triangles depth-format 0
              cull-mode :none polygon-mode :fill blend? false}} config
        depth-test? (if (some? depth-test?) depth-test? (pos? depth-format))
        depth-write? (if (some? depth-write?) depth-write? (pos? depth-format))
        stack (MemoryStack/stackPush)]
    (try
      (let [;; Compile shaders
            {vert-mod :module vert-spirv :spirv} (shader/compile-shader-module device vert-glsl :vertex)
            {frag-mod :module frag-spirv :spirv} (shader/compile-shader-module device frag-glsl :fragment)

            ;; Shader stages
            stages (VkPipelineShaderStageCreateInfo/calloc 2 stack)
            ^VkPipelineShaderStageCreateInfo vs (.get stages 0)
            ^VkPipelineShaderStageCreateInfo fs (.get stages 1)
            main-name (.UTF8 stack "main")
            _ (doto vs
                (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                (.stage VK13/VK_SHADER_STAGE_VERTEX_BIT)
                (.module vert-mod)
                (.pName main-name))
            _ (doto fs
                (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                (.stage VK13/VK_SHADER_STAGE_FRAGMENT_BIT)
                (.module frag-mod)
                (.pName main-name))

            ;; Vertex input
            vi-ci (doto (VkPipelineVertexInputStateCreateInfo/calloc stack)
                    (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO))
            _ (when (seq vertex-bindings)
                (let [binds (VkVertexInputBindingDescription/calloc (count vertex-bindings) stack)]
                  (dotimes [i (count vertex-bindings)]
                    (let [b (nth vertex-bindings i)
                          ^VkVertexInputBindingDescription bd (.get binds i)]
                      (doto bd
                        (.binding (int (:binding b)))
                        (.stride (int (:stride b)))
                        (.inputRate VK13/VK_VERTEX_INPUT_RATE_VERTEX))))
                  (.pVertexBindingDescriptions vi-ci binds)))
            _ (when (seq vertex-attributes)
                (let [attrs (VkVertexInputAttributeDescription/calloc (count vertex-attributes) stack)]
                  (dotimes [i (count vertex-attributes)]
                    (let [a (nth vertex-attributes i)
                          ^VkVertexInputAttributeDescription ad (.get attrs i)]
                      (doto ad
                        (.location (int (:location a)))
                        (.binding (int (:binding a)))
                        (.format (int (:format a)))
                        (.offset (int (:offset a))))))
                  (.pVertexAttributeDescriptions vi-ci attrs)))

            ;; Input assembly
            ia-ci (doto (VkPipelineInputAssemblyStateCreateInfo/calloc stack)
                    (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO)
                    (.topology (case topology
                                 :triangles VK13/VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                                 :lines VK13/VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                                 :points VK13/VK_PRIMITIVE_TOPOLOGY_POINT_LIST))
                    (.primitiveRestartEnable false))

            ;; Dynamic viewport/scissor
            dyn-states (.mallocInt stack 2)
            _ (.put dyn-states 0 VK13/VK_DYNAMIC_STATE_VIEWPORT)
            _ (.put dyn-states 1 VK13/VK_DYNAMIC_STATE_SCISSOR)
            dyn-ci (doto (VkPipelineDynamicStateCreateInfo/calloc stack)
                     (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO)
                     (.pDynamicStates dyn-states))

            ;; Viewport (dynamic, just need count)
            vp-ci (doto (VkPipelineViewportStateCreateInfo/calloc stack)
                    (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO)
                    (.viewportCount 1)
                    (.scissorCount 1))

            ;; Rasterization
            rast-ci (doto (VkPipelineRasterizationStateCreateInfo/calloc stack)
                      (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO)
                      (.depthClampEnable false)
                      (.rasterizerDiscardEnable false)
                      (.polygonMode (case polygon-mode
                                      :fill VK13/VK_POLYGON_MODE_FILL
                                      :line VK13/VK_POLYGON_MODE_LINE))
                      (.lineWidth (float 1.0))
                      (.cullMode (case cull-mode
                                   :none VK13/VK_CULL_MODE_NONE
                                   :back VK13/VK_CULL_MODE_BACK_BIT
                                   :front VK13/VK_CULL_MODE_FRONT_BIT))
                      (.frontFace VK13/VK_FRONT_FACE_COUNTER_CLOCKWISE)
                      (.depthBiasEnable false))

            ;; Multisampling (none)
            ms-ci (doto (VkPipelineMultisampleStateCreateInfo/calloc stack)
                    (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO)
                    (.sampleShadingEnable false)
                    (.rasterizationSamples VK13/VK_SAMPLE_COUNT_1_BIT))

            ;; Depth/stencil
            ds-ci (doto (VkPipelineDepthStencilStateCreateInfo/calloc stack)
                    (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO)
                    (.depthTestEnable (boolean depth-test?))
                    (.depthWriteEnable (boolean depth-write?))
                    (.depthCompareOp VK13/VK_COMPARE_OP_LESS)
                    (.depthBoundsTestEnable false)
                    (.stencilTestEnable false))

            ;; Color blend (alpha blend)
            cb-att (VkPipelineColorBlendAttachmentState/calloc 1 stack)
            ^VkPipelineColorBlendAttachmentState cb0 (.get cb-att 0)
            _ (doto cb0
                (.colorWriteMask (bit-or VK13/VK_COLOR_COMPONENT_R_BIT
                                         VK13/VK_COLOR_COMPONENT_G_BIT
                                         VK13/VK_COLOR_COMPONENT_B_BIT
                                         VK13/VK_COLOR_COMPONENT_A_BIT))
                (.blendEnable (boolean blend?)))
            _ (when blend?
                (doto cb0
                  (.srcColorBlendFactor VK13/VK_BLEND_FACTOR_SRC_ALPHA)
                  (.dstColorBlendFactor VK13/VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
                  (.colorBlendOp VK13/VK_BLEND_OP_ADD)
                  (.srcAlphaBlendFactor VK13/VK_BLEND_FACTOR_ONE)
                  (.dstAlphaBlendFactor VK13/VK_BLEND_FACTOR_ZERO)
                  (.alphaBlendOp VK13/VK_BLEND_OP_ADD)))
            cb-ci (doto (VkPipelineColorBlendStateCreateInfo/calloc stack)
                    (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO)
                    (.logicOpEnable false)
                    (.pAttachments cb-att))

            ;; Pipeline layout (use provided or create from push constants)
            layout (or layout (create-graphics-pipeline-layout device push-constants))

            ;; Dynamic rendering info (Vulkan 1.3)
            rendering-ci (doto (VkPipelineRenderingCreateInfo/calloc stack)
                           (.sType VK13/VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO)
                           (.colorAttachmentCount 1)
                           (.pColorAttachmentFormats (.ints stack (int color-format)))
                           (.depthAttachmentFormat (int depth-format)))

            ;; The pipeline
            pipe-buf (VkGraphicsPipelineCreateInfo/calloc 1 stack)
            ^VkGraphicsPipelineCreateInfo pipe0 (.get pipe-buf 0)
            _ (doto pipe0
                (.sType VK13/VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO)
                (.pNext (.address rendering-ci))
                (.stageCount 2)
                (.pStages stages)
                (.pVertexInputState vi-ci)
                (.pInputAssemblyState ia-ci)
                (.pViewportState vp-ci)
                (.pRasterizationState rast-ci)
                (.pMultisampleState ms-ci)
                (.pDepthStencilState ds-ci)
                (.pColorBlendState cb-ci)
                (.pDynamicState dyn-ci)
                (.layout layout)
                (.renderPass 0)  ;; no render pass — dynamic rendering
                (.subpass 0))

            p-pipeline (.mallocLong stack 1)
            result (VK13/vkCreateGraphicsPipelines device 0 pipe-buf nil p-pipeline)]
        (when-not (= result VK13/VK_SUCCESS)
          (throw (ex-info "Failed to create graphics pipeline" {:result result})))
        {:pipeline (.get p-pipeline 0)
         :layout layout
         :vert-module vert-mod
         :frag-module frag-mod
         :vert-spirv vert-spirv
         :frag-spirv frag-spirv})
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Dynamic rendering commands
;; ================================================================

(defn cmd-begin-rendering!
  "Begin dynamic rendering. attachments: {:color-view long :depth-view long or nil}
  extent: [w h], clear-color: [r g b a]."
  [^VkCommandBuffer cb attachments extent clear-color]
  (let [w (int (nth extent 0))
        h (int (nth extent 1))
        stack (MemoryStack/stackPush)]
    (try
      (let [color-att (doto (org.lwjgl.vulkan.VkRenderingAttachmentInfo/calloc 1 stack)
                        (-> (.get 0)
                            (.sType VK13/VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                            (.imageView (long (:color-view attachments)))
                            (.imageLayout VK13/VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                            (.loadOp VK13/VK_ATTACHMENT_LOAD_OP_CLEAR)
                            (.storeOp VK13/VK_ATTACHMENT_STORE_OP_STORE)))
            ;; Set clear color
            ^org.lwjgl.vulkan.VkRenderingAttachmentInfo ca0 (.get color-att 0)
            _ (-> (.clearValue ca0) .color
                  (.float32 0 (float (nth clear-color 0)))
                  (.float32 1 (float (nth clear-color 1)))
                  (.float32 2 (float (nth clear-color 2)))
                  (.float32 3 (float (nth clear-color 3))))
            ri (doto (org.lwjgl.vulkan.VkRenderingInfo/calloc stack)
                 (.sType VK13/VK_STRUCTURE_TYPE_RENDERING_INFO)
                 (.pColorAttachments color-att)
                 (.layerCount 1))
            _ (doto (.renderArea ri)
                (-> .offset (.x 0) (.y 0))
                (-> .extent (.width w) (.height h)))]
        ;; Add depth attachment if provided
        (when-let [dv (:depth-view attachments)]
          (let [depth-att (doto (org.lwjgl.vulkan.VkRenderingAttachmentInfo/calloc 1 stack)
                            (-> (.get 0)
                                (.sType VK13/VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO)
                                (.imageView (long dv))
                                (.imageLayout VK13/VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                (.loadOp VK13/VK_ATTACHMENT_LOAD_OP_CLEAR)
                                (.storeOp VK13/VK_ATTACHMENT_STORE_OP_DONT_CARE)))
                ^org.lwjgl.vulkan.VkRenderingAttachmentInfo da0 (.get depth-att 0)]
            (-> (.clearValue da0) (.depthStencil) (.depth (float 1.0)) (.stencil 0))
            (.pDepthAttachment ri da0)))
        (VK13/vkCmdBeginRendering cb ri))
      (finally
        (MemoryStack/stackPop)))))

(defn cmd-end-rendering! [^VkCommandBuffer cb]
  (VK13/vkCmdEndRendering cb))

(defn cmd-set-viewport!
  "Set dynamic viewport."
  [^VkCommandBuffer cb ^long w ^long h]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [vp (doto (VkViewport/calloc 1 stack)
                 (-> (.get 0)
                     (.x (float 0))
                     (.y (float 0))
                     (.width (float w))
                     (.height (float h))
                     (.minDepth (float 0))
                     (.maxDepth (float 1))))]
        (VK13/vkCmdSetViewport cb 0 vp))
      (finally
        (MemoryStack/stackPop)))))

(defn cmd-set-scissor!
  "Set dynamic scissor."
  [^VkCommandBuffer cb ^long w ^long h]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [sc (VkRect2D/calloc 1 stack)
            ^VkRect2D s0 (.get sc 0)]
        (doto (.offset s0) (.x 0) (.y 0))
        (doto (.extent s0) (.width (int w)) (.height (int h)))
        (VK13/vkCmdSetScissor cb 0 sc))
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Cleanup
;; ================================================================

(defn destroy-graphics-pipeline!
  [^VkDevice device pipeline-info]
  (VK13/vkDestroyPipeline device (:pipeline pipeline-info) nil)
  (VK13/vkDestroyPipelineLayout device (:layout pipeline-info) nil)
  (shader/destroy-shader-module! device (:vert-module pipeline-info))
  (shader/destroy-shader-module! device (:frag-module pipeline-info))
  (when-let [s (:vert-spirv pipeline-info)] (MemoryUtil/memFree ^ByteBuffer s))
  (when-let [s (:frag-spirv pipeline-info)] (MemoryUtil/memFree ^ByteBuffer s)))
