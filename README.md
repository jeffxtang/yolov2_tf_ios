# What
This repo contains a sample iOS app that runs with a 80-class COCO trained tiny-yolo model and the TensorFlow library to do object detection and show the detected results as well as bounding boxes.

# How

1. Download the TensorFlow 1.2 source [here](https://github.com/tensorflow/tensorflow/releases/tag/v1.2.0);

2. Build the TensorFlow iOS libraries by following the instructions [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile). After you complete it successfully, you'll see `libprotobuf.a` and `libprotobuf-lite.a` in `tensorflow/contrib/makefile/gen/protobuf_ios/lib/`, and `libtensorflow-core.a` in `tensorflow/contrib/makefile/gen/lib`;

3. Clone this repo and then move the whole folder `yolov2_tf_ios` to `tensorflow/contrib/ios_examples/` (so the TF libraries built in step 2 will be available) and open the project in Xcode;

4. Download the OpenCV 3.2.0 iOS pack [here](http://opencv.org/releases.html) and then drag and drop the `opencv2.framework` to the Xcode project - this is to draw bounding boxes but of course you can do so also (easily) with the iOS API. Still I included OpenCV in the project to be ready for other more advanced OpenCV based CV processing;

5. Run the app in Xcode on simulator or device and tap the button to see results as the screenshot at the top.

# Notes

1. This repo includes two pre-trained and TF-quantized models (to reduce the model size significantly): quantized-tiny-yolo.pb and quantized-tiny-yolo-voc.pb. The first one was originally trained using the PASCAL VOC dataset for 20-class object detection, and the second one the COCO dataset for 80-class object detection. I created the pb files using [Darkflow](https://github.com/thtrieu/darkflow) then ran `bazel-bin/tensorflow/tools/quantization/quantize_graph --input=darkflow/built_graph/tiny-yolo.pb --output_node_names=output --output=quantized_tiny-yolo.pb --mode=weights` and `bazel-bin/tensorflow/tools/quantization/quantize_graph --input=darkflow/built_graph/tiny-yolo-voc.pb --output_node_names=output --output=quantized_tiny-yolo-voc.pb --mode=weights`. Both `tiny-yolo(-vos).pb` and its quantized versions can run successfully on device;

2. I also tested the `yolo.pb` and its quantized version - they can run OK on simulator but would crash on actual device.

# Credits
The preprocessing and postprocessing of the input image tensor and output tensor in iOS with TensorFlow C++ API was finally made right after many hours of debugging, review of the original Yolo9000 paper, and the following sources:

1. [Darkflow](https://github.com/thtrieu/darkflow) source and README;

2. The [Real-time object detection with YOLO](http://machinethink.net/blog/object-detection-with-yolo/) blog;

3. The Tensorflow [Android source code of TF Detect](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android).


