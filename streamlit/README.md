# Streamlit

This app is the deployment of the ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8_otters.tflite customed trained with a dataset of 500 trained images and 50 validation images labelled on Roboflow. 

It allows users to upload images or take photos using the webcam or camera of a mobile phone, to return an image of bounding boxes identifying the otters as well as the total count of otters within the image. 

The app consists of the webrtc snapshot component and the tf lite object detection for otters, adapted and updated from various sources.

## References

TensorFlow 2 Object Detection API ([source])(https://github.com/tensorflow/models/tree/master/research/object_detection)

Streamlitwebcam by soft-nougat ([source])(https://github.com/soft-nougat/streamlitwebcam)

Discussion of webrtc by the author of the component ([source])(https://discuss.streamlit.io/t/new-component-streamlit-webrtc-a-new-way-to-deal-with-real-time-media-streams/8669?u=whitphx)