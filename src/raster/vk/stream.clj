(ns raster.vk.stream
  "Stream Vulkan frames to browser clients via WebSocket + H.264 encoding.

  Usage:
    (stream/start! ctx w h)           ;; Start streaming server on port 8090
    (stream/start! ctx w h :port 9000 :fps 30)
    ;; Open http://localhost:8090/ in browser
    (stream/stop!)

  Integrates with raster.vk.frame via post-frame-hook atom.
  Captures swapchain images, H.264-encodes via ffmpeg subprocess,
  sends to all connected browsers via WebSocket.
  Browser input (keyboard/mouse) is relayed back and injected into
  raster.vk.input's event atom."
  (:require
   [raster.vk.core :as vk]
   [raster.vk.mem :as mem]
   [org.httpkit.server :as http])
  (:import
   [org.lwjgl.vulkan VK13 KHRSwapchain VkCommandBuffer VkBufferImageCopy]
   [org.lwjgl.system MemoryStack]
   [java.io OutputStream BufferedOutputStream BufferedInputStream]
   [java.util.concurrent ConcurrentLinkedQueue]))

(set! *warn-on-reflection* true)

;; ================================================================
;; State
;; ================================================================

(defonce state (atom nil))

;; Input events from browser clients — injected into input system
(defonce ^ConcurrentLinkedQueue input-queue (ConcurrentLinkedQueue.))

;; Post-frame hook — set by start!, cleared by stop!
;; frame.clj calls this after each submit
(defonce post-frame-hook (atom nil))

;; ================================================================
;; Input injection
;; ================================================================

(defn inject-browser-events!
  "Drain browser input queue and inject events into the GLFW input system's
  events atom. Call from the game's update function.
  input-sys: the InputSystem record from raster.vk.input."
  [input-sys]
  (let [events-atom (:events input-sys)]
    (loop []
      (when-let [raw (.poll input-queue)]
        (let [evt (try (read-string raw) (catch Exception _ nil))]
          (when evt
            (case (:type evt)
              "key"
              (let [k (keyword (:key evt))]
                (if (:down evt)
                  (swap! events-atom conj [:key-down k])
                  (swap! events-atom conj [:key-up k])))
              "mouse"
              ;; Browser sends raw deltas from pointer lock.
              ;; The input system computes delta = pos - prev_pos.
              ;; We inject a fake mouse-move that shifts the current pos by dx,dy.
              (let [dx (double (or (:dx evt) 0))
                    dy (double (or (:dy evt) 0))
                    prev-pos (:mouse-pos @(:state input-sys))
                    px (double (nth prev-pos 0))
                    py (double (nth prev-pos 1))]
                (swap! events-atom conj [:mouse-move (+ px dx) (+ py dy)]))
              "click"
              (let [b (keyword (:button evt))]
                (swap! events-atom conj [:mouse-down b])
                ;; Auto-release after 1 event (edge trigger)
                (swap! events-atom conj [:mouse-up b]))
              ;; unknown event type — ignore
              nil)))
        (recur)))))

;; ================================================================
;; FFmpeg H.264 encoder
;; ================================================================

(defn- start-encoder! [w h fps]
  (let [cmd ["ffmpeg" "-y"
             "-f" "rawvideo" "-pix_fmt" "bgra"
             "-s" (str w "x" h)
             "-r" (str fps)
             "-i" "pipe:0"
             "-pix_fmt" "yuv420p"      ;; baseline needs 4:2:0
             "-c:v" "libx264"
             "-preset" "ultrafast"
             "-tune" "zerolatency"
             "-crf" "23"
             "-g" (str (* fps 2))     ;; keyframe every 2 sec
             "-profile:v" "baseline"
             "-level" "3.1"
             "-f" "h264"              ;; raw Annex B NAL units
             "-an"                    ;; no audio
             "pipe:1"]
        pb (doto (ProcessBuilder. ^java.util.List cmd)
             (.redirectErrorStream false)
             (.environment))
        proc (.start pb)
        stdin (BufferedOutputStream. (.getOutputStream proc) (* 256 1024))
        stdout (BufferedInputStream. (.getInputStream proc) (* 256 1024))
        stderr (.getErrorStream proc)]
    ;; Drain stderr in background to prevent blocking
    (let [t (Thread.
             (fn []
               (try
                 (let [buf (byte-array 4096)]
                   (loop []
                     (let [n (.read stderr buf)]
                       (when (pos? n)
                          ;; Uncomment for debugging:
                          ;; (print (String. buf 0 n))
                         (recur)))))
                 (catch Exception _)))
             "ffmpeg-stderr")]
      (.setDaemon t true)
      (.start t))
    {:process proc :stdin stdin :stdout stdout}))

(defn- start-reader-thread!
  "Read raw H.264 NAL units from ffmpeg stdout and broadcast to WebSocket clients."
  [stdout clients]
  (let [t (Thread.
           (fn []
             (try
               (let [buf (byte-array 65536)]
                 (loop []
                   (let [n (.read ^java.io.InputStream stdout buf)]
                     (when (pos? n)
                       (let [data (byte-array n)]
                         (System/arraycopy buf 0 data 0 n)
                         (doseq [ch @clients]
                           (try
                             (http/send! ch data false)
                             (catch Exception _
                               (swap! clients disj ch)))))
                       (recur)))))
               (catch Exception e
                 (when-not (instance? java.io.IOException e)
                   (println "Stream reader error:" (.getMessage e))))))
           "stream-reader")]
    (.setDaemon t true)
    (.start t)
    t))

;; ================================================================
;; Frame capture
;; ================================================================

(defn- read-staging-to-stream!
  "Copy BGRA pixels from staging buffer directly to ffmpeg stdin."
  [allocator staging ^OutputStream out dims]
  (let [w (long (first dims))
        h (long (second dims))
        nbytes (* w h 4)
        ^java.nio.ByteBuffer bb (mem/map-buffer allocator staging)
        arr (byte-array nbytes)]
    ;; Bulk copy from direct ByteBuffer to byte array
    (.get bb arr)
    (mem/unmap-buffer allocator staging)
    (.write out arr)
    (.flush out)))

(defn- capture-frame!
  "Capture current swapchain image to staging buffer, pipe to ffmpeg."
  [ctx image-and-dims]
  (when-let [st @state]
    (let [image (long (nth image-and-dims 0))
          w (long (nth image-and-dims 1))
          h (long (nth image-and-dims 2))
          device (:device ctx)
          allocator (long (:allocator ctx))
          staging (:staging st)
          encoder-stdin (:encoder-stdin st)]
      (try
        ;; No vkDeviceWaitIdle — the queue processes commands in submission
        ;; order, so submit-and-wait! will naturally wait for the render
        ;; submit to complete before our copy executes.
        (let [cb (vk/allocate-command-buffer device (:command-pool ctx))]
          (vk/begin-command-buffer cb)
          ;; Transition to TRANSFER_SRC
          (vk/cmd-image-barrier! cb image
                                 [KHRSwapchain/VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                  VK13/VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL]
                                 [VK13/VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
                                  VK13/VK_PIPELINE_STAGE_TRANSFER_BIT]
                                 [0 VK13/VK_ACCESS_TRANSFER_READ_BIT])
          ;; Copy command
          (let [stack (MemoryStack/stackPush)]
            (try
              (let [region (VkBufferImageCopy/calloc 1 stack)
                    ^VkBufferImageCopy r (.get region 0)]
                (.bufferOffset r 0)
                (.bufferRowLength r 0)
                (.bufferImageHeight r 0)
                (doto (.imageSubresource r)
                  (.aspectMask VK13/VK_IMAGE_ASPECT_COLOR_BIT)
                  (.mipLevel 0)
                  (.baseArrayLayer 0)
                  (.layerCount 1))
                (doto (.imageOffset r) (.set 0 0 0))
                (doto (.imageExtent r) (.set (int w) (int h) 1))
                (VK13/vkCmdCopyImageToBuffer cb image
                                             VK13/VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                             (long (:buffer staging)) region))
              (finally (MemoryStack/stackPop))))
          ;; Transition back
          (vk/cmd-image-barrier! cb image
                                 [VK13/VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                  KHRSwapchain/VK_IMAGE_LAYOUT_PRESENT_SRC_KHR]
                                 [VK13/VK_PIPELINE_STAGE_TRANSFER_BIT
                                  VK13/VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT]
                                 [VK13/VK_ACCESS_TRANSFER_READ_BIT 0])
          (vk/end-command-buffer cb)
          (vk/submit-and-wait! device (:queue ctx) cb))
        ;; Read pixels and pipe to ffmpeg
        (read-staging-to-stream! allocator staging encoder-stdin [w h])
        (catch Exception e
          (println "Stream capture error:" (.getMessage e)))))))

;; ================================================================
;; WebSocket server + HTML client
;; ================================================================

(def client-html
  "<!DOCTYPE html>
<html><head>
<meta charset='utf-8'>
<title>Raster Stream Viewer</title>
<style>
* { margin: 0; padding: 0; }
body { background: #111; overflow: hidden; display: flex; justify-content: center; align-items: center; height: 100vh; }
video { background: #000; }
#info { position: fixed; top: 8px; left: 8px; color: #0f0; font: 14px monospace; z-index: 10; text-shadow: 0 0 4px #000; }
</style>
</head><body>
<div id='info'>Connecting...</div>
<video id='v' autoplay muted playsinline></video>
<script src='/jmuxer.js'></script>
<script>
const video = document.getElementById('v');
const info = document.getElementById('info');
let byteCount = 0, lastStatTime = performance.now();

function connect() {
  const ws = new WebSocket('ws://' + location.host + '/ws');
  ws.binaryType = 'arraybuffer';

  const jmuxer = new JMuxer({
    node: 'v',
    mode: 'video',
    flushingTime: 0,
    maxDelay: 0,
    fps: 20,
    debug: false
  });

  ws.onopen = () => {
    info.textContent = 'Connected — waiting for video...';
  };

  ws.onclose = () => {
    info.textContent = 'Disconnected — reconnecting...';
    jmuxer.destroy();
    setTimeout(connect, 1000);
  };

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      jmuxer.feed({ video: new Uint8Array(e.data) });
      byteCount += e.data.byteLength;
      const now = performance.now();
      if (now - lastStatTime > 1000) {
        const kbps = (byteCount * 8 / 1000) / ((now - lastStatTime) / 1000);
        byteCount = 0;
        lastStatTime = now;
        const w = video.videoWidth || '?';
        const h = video.videoHeight || '?';
        info.textContent = w + 'x' + h + ' | ' + kbps.toFixed(0) + ' kbps';
      }
    }
  };

  // Input handling
  const keyMap = {
    'KeyW': 'w', 'KeyA': 'a', 'KeyS': 's', 'KeyD': 'd',
    'Space': 'space', 'ShiftLeft': 'left-shift', 'ShiftRight': 'right-shift',
    'Escape': 'escape', 'KeyF': 'f', 'KeyE': 'e',
    'Digit1': '1', 'Digit2': '2', 'Digit3': '3', 'Digit4': '4',
    'Digit5': '5', 'Digit6': '6', 'Digit7': '7', 'Digit8': '8', 'Digit9': '9'
  };

  function sendInput(msg) {
    let edn = '{';
    for (const [k, v] of Object.entries(msg)) {
      edn += ':' + k + ' ';
      if (typeof v === 'string') edn += '\"' + v + '\"';
      else if (typeof v === 'boolean') edn += v.toString();
      else edn += v;
      edn += ' ';
    }
    edn += '}';
    if (ws.readyState === 1) ws.send(edn);
  }

  document.addEventListener('keydown', (e) => {
    const key = keyMap[e.code];
    if (key) { e.preventDefault(); sendInput({type:'key', key:key, down:true}); }
  });
  document.addEventListener('keyup', (e) => {
    const key = keyMap[e.code];
    if (key) { e.preventDefault(); sendInput({type:'key', key:key, down:false}); }
  });

  let pointerLocked = false;
  video.addEventListener('click', () => {
    if (!pointerLocked) video.requestPointerLock();
    else sendInput({type:'click', button:'left'});
  });
  video.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    if (pointerLocked) sendInput({type:'click', button:'right'});
  });
  document.addEventListener('pointerlockchange', () => {
    pointerLocked = document.pointerLockElement === video;
  });
  document.addEventListener('mousemove', (e) => {
    if (pointerLocked) sendInput({type:'mouse', dx:e.movementX, dy:e.movementY});
  });
}

function fitVideo() {
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const scale = Math.min(window.innerWidth / vw, window.innerHeight / vh);
  video.style.width = (vw * scale) + 'px';
  video.style.height = (vh * scale) + 'px';
}
video.addEventListener('loadedmetadata', fitVideo);
window.addEventListener('resize', fitVideo);
fitVideo();

connect();
</script>
</body></html>")

(def ^:private jmuxer-js-path
  (str (System/getProperty "user.home")
       "/Development/jmuxer/dist/jmuxer.min.js"))

(defn- ws-handler [req]
  (case (:uri req)
    "/ws"
    (http/with-channel req ch
      (when-let [st @state]
        (swap! (:clients st) conj ch)
        (println "Stream client connected:" (:remote-addr req))
        (http/on-close ch
                       (fn [_]
                         (when-let [st @state]
                           (swap! (:clients st) disj ch))
                         (println "Stream client disconnected:" (:remote-addr req))))
        (http/on-receive ch
                         (fn [data]
                           (when (string? data)
                             (.offer input-queue ^String data))))))
    "/jmuxer.js"
    {:status 200
     :headers {"Content-Type" "application/javascript"
               "Cache-Control" "public, max-age=86400"}
     :body (java.io.File. ^String jmuxer-js-path)}
    ;; Default: serve HTML client page
    {:status 200
     :headers {"Content-Type" "text/html; charset=utf-8"}
     :body client-html}))

;; ================================================================
;; Public API
;; ================================================================

(defn start!
  "Start streaming server. Captures frames via ffmpeg H.264 encoding.
  Options:
    :port — WebSocket server port (default 8090)
    :fps  — Target capture FPS (default 20)"
  [ctx w h & {:keys [port fps] :or {port 8090 fps 20}}]
  (when @state
    (println "Streaming already active. Call (stop!) first.")
    (throw (ex-info "Already streaming" {})))
  (let [allocator (long (:allocator ctx))
        w (long w) h (long h)
        buf-size (* w h 4)
        staging (mem/create-buffer allocator buf-size :staging :gpu-to-cpu)
        encoder (start-encoder! w h fps)
        clients (atom #{})
        reader-thread (start-reader-thread! (:stdout encoder) clients)
        frame-skip (max 1 (long (/ 60.0 (double fps))))
        frame-counter (atom 0)
        stop-fn (http/run-server ws-handler {:port port})]
    (reset! state
            {:staging staging
             :encoder encoder
             :encoder-stdin (:stdin encoder)
             :clients clients
             :reader-thread reader-thread
             :ws-stop-fn stop-fn
             :frame-skip frame-skip
             :frame-counter frame-counter
             :w w :h h})
    ;; Register post-frame hook
    (reset! post-frame-hook
            (fn [ctx image w h]
              (let [fc (swap! frame-counter inc)]
                (when (zero? (mod fc frame-skip))
                  (when (seq @clients)
                    (capture-frame! ctx [image w h]))))))
    (println (str "Streaming started on http://localhost:" port "/"))
    (println (str "H.264 encoding at " fps " fps via ffmpeg"))
    :started))

(defn stop!
  "Stop streaming server and clean up resources."
  []
  (when-let [st @state]
    (reset! post-frame-hook nil)
    ;; Close encoder stdin → ffmpeg exits
    (try (.close ^OutputStream (:encoder-stdin st)) (catch Exception _))
    ;; Wait for ffmpeg to exit
    (try (.waitFor ^Process (:process (:encoder st)) 3 java.util.concurrent.TimeUnit/SECONDS)
         (catch Exception _))
    (try (.destroyForcibly ^Process (:process (:encoder st))) (catch Exception _))
    ;; Close all WebSocket connections
    (doseq [ch @(:clients st)]
      (http/close ch))
    ;; Stop HTTP server
    ((:ws-stop-fn st))
    (reset! state nil)
    (.clear input-queue)
    (println "Streaming stopped.")
    :stopped))

(defn poll-input!
  "Poll accumulated input events from browser clients.
  Returns a vector of parsed input maps, or empty vector."
  []
  (loop [events (transient [])]
    (if-let [raw (.poll input-queue)]
      (recur (conj! events
                    (try (read-string raw)
                         (catch Exception _ nil))))
      (persistent! events))))
