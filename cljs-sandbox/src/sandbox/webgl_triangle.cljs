(ns sandbox.webgl-triangle
  "Experiment 3 — WebGL2 hello triangle. The render path raster.vk does NOT port
   to; this is the greenfield WebGL2 target (WebGPU-render can come later). Kept
   deliberately minimal: a single VBO + shader program, no abstraction — just to
   confirm the cljs↔WebGL2 interop seam before any renderer design.")

(def vs "#version 300 es
in vec2 pos; in vec3 col; out vec3 vcol;
void main(){ vcol = col; gl_Position = vec4(pos, 0.0, 1.0); }")

(def fs "#version 300 es
precision highp float; in vec3 vcol; out vec4 o;
void main(){ o = vec4(vcol, 1.0); }")

(defn compile-shader [gl type src]
  (let [s (.createShader gl type)]
    (.shaderSource gl s src)
    (.compileShader gl s)
    (when-not (.getShaderParameter gl s (.-COMPILE_STATUS gl))
      (js/console.error "shader compile error:" (.getShaderInfoLog gl s)))
    s))

(defn run! []
  (let [canvas (.getElementById js/document "gl")
        gl     (.getContext canvas "webgl2")]
    (if-not gl
      (js/console.error "WebGL2 not available")
      (let [prog (.createProgram gl)
            vbo  (.createBuffer gl)
            ;; interleaved: x y r g b
            data (js/Float32Array. #js [ 0.0  0.8  1.0 0.2 0.2
                                        -0.8 -0.8  0.2 1.0 0.2
                                         0.8 -0.8  0.2 0.2 1.0])]
        (.attachShader gl prog (compile-shader gl (.-VERTEX_SHADER gl) vs))
        (.attachShader gl prog (compile-shader gl (.-FRAGMENT_SHADER gl) fs))
        (.linkProgram gl prog)
        (.useProgram gl prog)
        (.bindBuffer gl (.-ARRAY_BUFFER gl) vbo)
        (.bufferData gl (.-ARRAY_BUFFER gl) data (.-STATIC_DRAW gl))
        (let [stride (* 5 4)
              ploc (.getAttribLocation gl prog "pos")
              cloc (.getAttribLocation gl prog "col")]
          (.enableVertexAttribArray gl ploc)
          (.vertexAttribPointer gl ploc 2 (.-FLOAT gl) false stride 0)
          (.enableVertexAttribArray gl cloc)
          (.vertexAttribPointer gl cloc 3 (.-FLOAT gl) false stride (* 2 4)))
        (.clearColor gl 0.1 0.1 0.12 1.0)
        (.clear gl (.-COLOR_BUFFER_BIT gl))
        (.drawArrays gl (.-TRIANGLES gl) 0 3)
        (js/console.log "WebGL2 triangle drawn.")))))
