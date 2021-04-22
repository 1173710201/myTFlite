cd /home/xmy/tensorflow/bazel-bin/tensorflow/lite/java
cp -f classes.jar /home/xmy/Android/image_classification/android/lib_support/libs
cp -f classes.jar /home/xmy/Android/image_classification/android/app/libs
cp -f libtensorflowlite_gpu_jni.so /home/xmy/Android/image_classification/android/lib_support/src/main/jniLibs/arm64-v8a
cp -f libtensorflowlite_gpu_jni.so /home/xmy/Android/image_classification/android/app/src/main/jniLibs/arm64-v8a
