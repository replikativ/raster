(ns raster.dl.lenet
  "LeNet-5 loss function for compiled pipeline benchmarking.

  Architecture: Conv(1→6,5x5) → ReLU → MaxPool(2x2) →
                Conv(6→16,5x5) → ReLU → MaxPool(2x2) →
                Dense(256→120) → ReLU → Dense(120→10) → Softmax → CrossEntropy

  Input: 28x28 grayscale (MNIST), flat double[784]"
  (:require [raster.core :refer [deftm]]
            [raster.nn :as nn]
            [raster.dl.nn :as dl]))

(deftm lenet-loss-fn
  [conv1-W :- (Array double) conv1-b :- (Array double)
   conv2-W :- (Array double) conv2-b :- (Array double)
   fc1-W :- (Array double) fc1-b :- (Array double)
   fc2-W :- (Array double) fc2-b :- (Array double)
   x :- (Array double) y :- (Array double)] :- Double
  (let [;; Conv1: 1→6, 5x5, no padding → [1,6,24,24]
        c1 (dl/conv2d x conv1-W conv1-b 1 1 28 28 6 5 5 1 1 0 0)
        ;; ReLU + MaxPool 2x2 → [1,6,12,12]
        r1 (nn/relu c1)
        p1 (dl/maxpool2d r1 1 6 24 24 2 2)
        ;; Conv2: 6→16, 5x5 → [1,16,8,8]
        c2 (dl/conv2d p1 conv2-W conv2-b 1 6 12 12 16 5 5 1 1 0 0)
        r2 (nn/relu c2)
        p2 (dl/maxpool2d r2 1 16 8 8 2 2)
        ;; Flatten is implicit (p2 is already flat double[256])
        ;; FC1: 256→120
        f1 (nn/dense fc1-W p2 fc1-b)
        a1 (nn/relu f1)
        ;; FC2: 120→10
        f2 (nn/dense fc2-W a1 fc2-b)
        p  (nn/softmax f2)]
    (nn/cross-entropy p y)))

(deftm lenet-loss-fn
  [conv1-W :- (Array float) conv1-b :- (Array float)
   conv2-W :- (Array float) conv2-b :- (Array float)
   fc1-W :- (Array float) fc1-b :- (Array float)
   fc2-W :- (Array float) fc2-b :- (Array float)
   x :- (Array float) y :- (Array float)] :- Double
  (let [c1 (dl/conv2d x conv1-W conv1-b 1 1 28 28 6 5 5 1 1 0 0)
        r1 (nn/relu c1)
        p1 (dl/maxpool2d r1 1 6 24 24 2 2)
        c2 (dl/conv2d p1 conv2-W conv2-b 1 6 12 12 16 5 5 1 1 0 0)
        r2 (nn/relu c2)
        p2 (dl/maxpool2d r2 1 16 8 8 2 2)
        f1 (nn/dense fc1-W p2 fc1-b)
        a1 (nn/relu f1)
        f2 (nn/dense fc2-W a1 fc2-b)
        p  (nn/softmax f2)]
    (nn/cross-entropy p y)))

(deftm lenet-predict-fn
  [conv1-W :- (Array double) conv1-b :- (Array double)
   conv2-W :- (Array double) conv2-b :- (Array double)
   fc1-W :- (Array double) fc1-b :- (Array double)
   fc2-W :- (Array double) fc2-b :- (Array double)
   x :- (Array double)] :- (Array double)
  (let [c1 (dl/conv2d x conv1-W conv1-b 1 1 28 28 6 5 5 1 1 0 0)
        r1 (nn/relu c1)
        p1 (dl/maxpool2d r1 1 6 24 24 2 2)
        c2 (dl/conv2d p1 conv2-W conv2-b 1 6 12 12 16 5 5 1 1 0 0)
        r2 (nn/relu c2)
        p2 (dl/maxpool2d r2 1 16 8 8 2 2)
        f1 (nn/dense fc1-W p2 fc1-b)
        a1 (nn/relu f1)]
    (nn/dense fc2-W a1 fc2-b)))

(deftm lenet-predict-fn
  [conv1-W :- (Array float) conv1-b :- (Array float)
   conv2-W :- (Array float) conv2-b :- (Array float)
   fc1-W :- (Array float) fc1-b :- (Array float)
   fc2-W :- (Array float) fc2-b :- (Array float)
   x :- (Array float)] :- (Array float)
  (let [c1 (dl/conv2d x conv1-W conv1-b 1 1 28 28 6 5 5 1 1 0 0)
        r1 (nn/relu c1)
        p1 (dl/maxpool2d r1 1 6 24 24 2 2)
        c2 (dl/conv2d p1 conv2-W conv2-b 1 6 12 12 16 5 5 1 1 0 0)
        r2 (nn/relu c2)
        p2 (dl/maxpool2d r2 1 16 8 8 2 2)
        f1 (nn/dense fc1-W p2 fc1-b)
        a1 (nn/relu f1)]
    (nn/dense fc2-W a1 fc2-b)))
