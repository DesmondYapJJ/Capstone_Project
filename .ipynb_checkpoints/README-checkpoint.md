<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# Capstone: Tensorflow 2 Object Detection API - Detecting Otters in Singapore

Content:-
- Background
- Problem Statement
- What is Object Detection
- Mean Average Precision
- Data Preparation
    - Flickr scraper API
    - Roboflow
- Tensorflow 2 Object Detection API
- Streamlit
- Summary

# Background

## What is happening in Singapore? 

Singapore's otter population has more than doubled since 2019 - with the current estimate at around 170, roughly 17 families. In 2020, NParks received 208 citizen reports about otters, followed by 305 in 2021, and more than 300 as of this August. Although most reports are sightings, otters have been known to lash out when threatened. ([source](https://www.theguardian.com/environment/2022/oct/23/slippery-hungry-sometimes-angry-singapore-struggles-with-unparalleled-otter-boom)) 

There has been many news article coverage on them locally on Straits Times, Today, Mothership and even on international news such as CNA, The Guardian, BBC News and SCMP. 

With the increase in population, there has also been an increase in incidents of otters attacking people at Kallang Riverside Park([source](https://www.straitstimes.com/singapore/man-bitten-by-otter-after-trailing-pack-of-30-during-morning-run)) and Botanic Gardens([source](https://www.bbc.com/news/world-asia-59592355)). This has also created some trouble with home owners([source](https://mothership.sg/2022/10/bukit-timah-otter-eat-40-koi-fish/)) and building management for condominiums([source](https://www.todayonline.com/singapore/otters-seen-eating-fish-condominium-along-alexandra-canal-upsetting-residents)) as the otters roam freely, killing and feeding on koi and other fishes. 

This has prompted local authority NParks to relocate the otters away from local residential areas([source1](https://www.channelnewsasia.com/singapore/otters-seletar-relocation-last-resort-pups-holts-3012831))([source2](https://www.scmp.com/week-asia/health-environment/article/3196279/singapore-move-otters-out-residential-areas-more-hunt)). This has also led to discussions on managing the local otter population. Sightings were rare up until late 1998 when a pair of otters were spotted at Sungei Buloh Wetlands Reserve. It is also saddening that culling is the first thought of many keyboard warriors comments online, however the thoughts differ on ground. Experts also do not agree with culling with the current population, it is still manageable([source](https://www.todayonline.com/singapore/otter-population-sharply-still-manageable-say-experts-who-urge-public-learn-co-exist-them-1767076)). The priority should be to co-exist as the nation progresses with its [City in Nature goal](https://www.todayonline.com/singapore/cutting-landfill-waste-mandating-cleaner-vehicles-among-slew-goals-unveiled-singapore) as well. The public will also need to be educated, especially that although the otters may be cute and seemingly approachable, that they are nontheless wild animals and adult otters could potentially become aggressive with pups around. 

# Problem Statement

The bottomline is that things have to be managed. Singapore has to learn to co-exist with the otters and culling should not be an option for a City in Nature unless absolutely necessary. 

This project aims to detect the number of otters from an image, video or live stream and return the number of counts of otters identified. With that information, it can then be translated into many other uses. For example, security for the home owners and building management - a warning sound could be activate when the number of otters detected is above a threshold number or tracking of the otters by NParks just by counts. 

# What is Objection Detection

## Which brings us to the question - how can this be done?

A typical image classification would not be sufficient which is why we look to objection detection. 

Object recognition is a general term to describe a collection of related computer vision tasks that involve identifying objects in digital photographs([source](https://machinelearningmastery.com/object-recognition-with-deep-learning/)). 

Image classification involves assigning a class label to an image, whereas object localization involves drawing a bounding box around one or more objects in an image. Object detection is more challenging and combines these two tasks and draws a bounding box around each object of interest in the image and assigns them a class label. Together, all of these problems are referred to as object recognition.

![Object-Recognition.png](ppt_images/Object-Recognition.png)

- Image Classification: Predict the type or class of an object in an image.
Input: An image with a single object, such as a photograph.
Output: A class label (e.g. one or more integers that are mapped to class labels).

- Object Localization: Locate the presence of objects in an image and indicate their location with a bounding box.
Input: An image with one or more objects, such as a photograph.
Output: One or more bounding boxes (e.g. defined by a point, width, and height).

- Object Detection: Locate the presence of objects with a bounding box and types or classes of the located objects in an image.
Input: An image with one or more objects, such as a photograph.
Output: One or more bounding boxes (e.g. defined by a point, width, and height), and a class label for each bounding box.

![Capture17.JPG](ppt_images/Capture17.JPG)
([source](https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2))

## Types of techniques

### R-CNN
The R-CNN family of methods refers to the R-CNN, which may stand for “Regions with CNN Features” or “Region-Based Convolutional Neural Network,” developed by Ross Girshick, et al.

This includes the techniques R-CNN, Fast R-CNN, and Faster-RCNN designed and demonstrated for object localization and object recognition.

One of the latest SOTA model is the Faster R-CNN.

Although it is a single unified model, the architecture is comprised of two modules:

Module 1: Region Proposal Network. Convolutional neural network for proposing regions and the type of object to consider in the region.
Module 2: Fast R-CNN. Convolutional neural network for extracting features from the proposed regions and outputting the bounding box and class labels. 

### YOLO 
YOLO or “You Only Look Once,” developed by Joseph Redmon, et al.

The R-CNN models may be generally more accurate, yet the YOLO family of models are fast, much faster than R-CNN, achieving object detection in real-time.

### SSD MobileNet
Single Shot Detectors

As the name applied, the MobileNet model is designed to be used in mobile applications, and it is TensorFlow’s first mobile computer vision model([source](https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470)).

MobileNet is an object detector released in 2017 as an efficient CNN architecture designed for mobile and embedded vision application. This architecture uses proven depth-wise separable convolutions to build lightweight deep neural networks([source](https://arxiv.org/pdf/1704.04861.pdf)).

### Model Selection
R-CNN can be classified as a two stage object detection model while YOLO and SSD are one-stage. The major difference between the two is that in the two-stage object detection models, the region of interest is first determined and the detection is then performed only on the region of interest. This implies that the two-stage object detection models are generally more accurate than the one-stage ones but require more computational resources and are slower([source](https://vidishmehta204.medium.com/object-detection-using-ssd-mobilenet-v2-7ff3543d738d)).

In this project we will be utilizing the pretrained model **_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8_** from Tensorflow 2 Object Detection API([source](https://github.com/tensorflow/models/tree/master/research/object_detection))

There are many available pretrained models in the Tensorflow 2 Object Detection Model Zoo([source](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)), with a range of speed(speed(ms)), score(COCO mAP) and output, however due to hardware limitations and for deployment, a lighter model is chosen in this case. It is also possible to utilize google colab or other cloud platforms for the GPU to do the customized training of the model. 

An extraction from the Tensorflow 2 Object Detection Model Zoo for comparison purposes (not the full list). The model training can be computationally expensive, given if there are large amount of images dataset to work with.

![Capture16.jpg](ppt_images/Capture16.jpg)

# Mean Average Precision (mAP) 

Object detection systems make predictions in terms of a bounding box and a class label. For each bounding box, we measure an overlap between the predicted bounding box and the ground truth bounding box. This is measured by IoU (intersection over union)([source](https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2)).

![Capture9.JPG](ppt_images/Capture9.JPG)

For object detection tasks, we calculate Precision and Recall using IoU value for a given IoU threshold.

For example, if IoU threshold is 0.5, and the IoU value for a prediction is 0.7, then we classify the prediction as True Positive (TF). On the other hand, if IoU is 0.3, we classify it as False Positive (FP).

![Capture10.JPG](ppt_images/Capture10.JPG)

That means that we may get different binary true or false positives by changing the IoU threshold. In general a high IoU would be a more stringent requirement to get a good mAP score - the predicted box must have a high overlap with the ground truth box in order to be a true positive. 

![Capture11.JPG](ppt_images/Capture11.JPG)

The general definition for the Average Precision (AP) is finding the area under the precision-recall curve.

mAP (mean average precision) is the average of AP.

The mean Average Precision or mAP score is calculated by taking the mean AP over all classes and/or overall IoU thresholds, depending on different detection challenges that exist.

![622fd6ac6796060c0f599a02_i3TWRDyqCH81SirZy0KCbTXqrJgGyL_b4PeyZNdzgnt9sZRKrgBh1LsYTtZioV7g4VDi5d-09YcG8F14MQ91JuML-13OrJyP5MvDz2OVpjG3UFNJ9qz2EqQfMA0UNX3igaeWdXI3.png](ppt_images/622fd6ac6796060c0f599a02_i3TWRDyqCH81SirZy0KCbTXqrJgGyL_b4PeyZNdzgnt9sZRKrgBh1LsYTtZioV7g4VDi5d-09YcG8F14MQ91JuML-13OrJyP5MvDz2OVpjG3UFNJ9qz2EqQfMA0UNX3igaeWdXI3.png)([source](https://www.v7labs.com/blog/mean-average-precision#:~:text=let's%20dive%20in!-,What%20is%20Mean%20Average%20Precision%20(mAP)%3F,values%20from%200%20to%201.))

Precision is a measure of when ""your model predicts how often does it predicts correctly?"" It indicates how much we can rely on the model's positive predictions. 

![Capture12.JPG](ppt_images/Capture12.JPG)

Recall is a measure of ""has your model predicted every time that it should have predicted?"" It indicates any predictions that it should not have missed if the model is missing. 

![Capture13.JPG](ppt_images/Capture13.JPG)

# Data Preparation

In order to do a customized training on the pretrained model selected from Tensorflow 2 Object Detection Model Zoo, we will need to prepare a training and validation set of images with labelling. 

## Flickr scraper API 

For collection of images, we would utilize the Flickr image-scraping software developed by Ultralytics LLC.

https://github.com/ultralytics/flickr_scraper

It requires you to have a Flickr account and a personal API key and secret which can be obtained from https://www.flickr.com/services/apps/create/apply

A total of 700 images of otters were extracted with the API. 

## Roboflow

Traditionally hand [labelling](https://github.com/heartexlabs/labelImg) is required for preparing the images data. Labelimg is one of the first few tools that many have utilized. Since then many other platforms have also been developed and labelimg itself has also been updated. 

In this project, the platform that we utilized is [Roboflow](https://roboflow.com/). It has an option of utilizing a pretrained model to assist with the labelling, however more specific objects still have to be hand labelled (in our case, otters were not able to be detected by the AI assist). Roboflow also allows for segregation to train/validation/test sets, preprocessing options as well as augmentation if required. Since our final model ultilizes 320x320 images, we can use Roboflow to resize the images accordingly to reduce the workload on our model training (without GPU). Of the 700 images scrapped from Flickr, we would utilize 500 for training and 50 for validation. 

![Capture14.JPG](ppt_images/Capture14.JPG)

![Capture15.JPG](ppt_images/Capture15.JPG)

# Tensorflow 2 Object Detection API

The notebook currently shows the final version utilizing *ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8* which is exported to tf lite for deployment on streamlit. 

- Tensorflow 2 Object Detection API ([source](https://github.com/tensorflow/models/tree/master/research/object_detection))

- Installation Guide ([source1](https://github.com/tensorflow/models)) ([source2](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html))

- Installation Guide and runthrough by nicknochnack ([source1](https://www.youtube.com/watch?v=yqkISICHH-U)) ([source2](https://github.com/nicknochnack/TFODCourse))

Label Map and TF records can be exported from Roboflow

## Train the model 

A pretrained model based on *ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8* was also tested with 2000 train steps only. However this gave an undesirable result. Due to the time required for training, we kept with the *ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8* and increased the number of training steps to 25000. 

320x320 with 25000 steps took ~6 hours for training. There are various tunings possible including increasing the number of training steps, changing of batch size (depends on computational power), increasing the number of images for training dataset. 

We will look at both results afterwards.

Things to note 
- Batch size is changed to 4 
- We have 500 training images and 50 validation images 
- 500/4 = 125 
- number of train steps should be adjusted according to reach ~200 epoch 
- 125 X 200 = 25000 steps required

## Evaluate the Model with Tensorboard

### Following results extracted from terminal

#### *ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8* with 2000 training steps, 500 training images, 50 validation images

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
 
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.661
 
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.252
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.302
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.309
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.514
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.576
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.576

#### *ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8* with 25000 training steps, 500 training images, 50 validation images (~6 hours training time) 

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.530
 
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.822
 
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.617
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.531
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.543
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.647
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.683
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.683
 
#### Findings 
 
The final model shows a significant improvement to achieve decent results. Average Precision at IoU=0.50 increase from 0.661 to 0.822. Average Precision at IoU=0.75 increased from 0.252 to 0.617. It is also noted that as the IoU threshold increases, the average precision falls. This is expected as the the conditions for true positive become more stringent as IoU threshold increases. Still the results were decent for usage. 

Average recall also improved overall. 

The other possible tuning that can be done is to increase the number of training images dataset. However, this was kept as a constant due to time and computational power limitations. 

#### Using Tensorboard

In the trained model train and evaluation folders

Run the following in terminal, tensorboard --logdir=. then copy http://localhost:6006/ paste in browser

From Tensorboard, you are able to see the loss from the training folder 

![Capture8.JPG](ppt_images/Capture8.JPG)

As well as the mAP and AP from the evaluation folder 

![Capture2.JPG](ppt_images/Capture2.JPG)

Tensorboard also shows the result of the validation image data set

![Capture.JPG](ppt_images/Capture.JPG)

## Detect from images, webcam and video

Webcam example

![otters.gif](ppt_images/otters.gif)

Images example

![1otter_predict.jpg](Tensorflow/workspace/images/test/1otter_predict.jpg)

![2otter_predict.jpg](Tensorflow/workspace/images/test/2otter_predict.jpg)

![3otter_predict.jpg](Tensorflow/workspace/images/test/3otter_predict.jpg)

![4otter_predict.jpg](Tensorflow/workspace/images/test/4otter_predict.jpg)

Video example - refer to video_output2.mp4

# Streamlit

The deployment to Streamlit takes the input from users as images or snapshots from live webcam using the TF Lite model converted from the above code. One of the main reasons for choosing the mobilenet model is for the benefit of being a lighter weight model in comparison with YOLO or Faster R-CNN. It also allows for a lighter deployment that can be used on mobile and websites such as Streamlit. Object detection of the otters via live stream on webcam was excluded from the Streamlit and could be an improvement to be added on. 

However, it is also noted that the performance of the model after conversion to TF Lite format is not as capable as the original customized model. 

The final deployment can be found here https://desmondyapjj-capstone-project-streamlitweb-app-wmmrox.streamlit.app/

# Summary

As the otters' population grow, it would be necessary to find methods to co-exist with them. There are two main concerns to be addressed, from the residents and building managements' point of view - a deterrence, from a local authority such as NParks' point of view - a way to track and identify the otters. 

There are many techniques for object detection and we have briefly covered some of them. These techniques can be computationally taxing but nonetheless, hold the potential to do many things. 

Our final customized model is based on pretrained model ssd_mobilenet_v2_fpnlite_320x320. A total of 700 images were scraped from Flickr via Flickr scraper API and a data set of 500 training images and 50 validation images were prepared using Roboflow. Running through 25000 training steps returned decent results to be used in model prediction and deployment. We tested the model's prediction on images, videos and live webcam streaming. Average Precision at IoU=0.5 and IoU=0.75 were 0.822 and 0.617 respectively. The model was then exported to TF Lite to be deployed on streamlit. 

In this end, we hope to simulate the experience and show the possibility of incorporating object detection to other platforms, such as security cameras. The Streamlit app is made to display a warning sound when 3 or more otters are detected in the image input by the users. This is akin to a deterrence in a real world situation perhaps to scare off the otters and prevent them from entering the residential premises. Although there are other preventive measures, this can be viewed as an added on feature as a form of deterrence and automation since humans cannot be on guard 24hours a day and even physical barriers may be breached. Object detection can even be extended to other use cases in the future as well. 

### Recommendations

It is possible to utilize other pretrained models such as YOLO and Faster R-CNN to see which gives a better mAP score. There are also cloud platforms such as Google colab which allows certain hours of GPU usage in a week. 

The number of images used for training can also be increased as part of training model tuning. Other alternative tuning methods are augmentation (ssd_mobilenet_v2_fpnlite_320x320 has some built in augmentation in its base configuration, but more can be explored with Roboflow as well), training more epoch and training step. 

Streamlit can also do object detection based on a live webcam similar to the one ran in the code notebook. These would mimic a real world scenario more closely. 
