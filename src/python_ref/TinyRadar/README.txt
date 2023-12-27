This is a repository containing the dataset, extraction code and training code
used in "TinyRadarNN: Combining Spatial and Temporal Convolutional Neural Networks for Embedded Gesture Recognition with Short Range Radars" (https://arxiv.org/pdf/2006.16281.pdf).
This project is brought to you by Moritz Scherer, Michele Magno, Jonas Erb, Philipp Mayer, Manuel Eggimann, Luca Benini of Digital Circuits and Systems Group of ETH Zurich (https://iis.ee.ethz.ch/), with the support of Project Based Learning Center (https://pbl.ee.ethz.ch/). 

Please visit our project www page for more details https://tinyradar.ethz.ch/

License

The dataset found in the zipped folders, i.e. 11G.zip and 5G.zip is distributed under the
Creative Commons Attribution Non-Commercial 4.0 (CC-BY-NC) license. All code on this webpage
is distributed under the Apache-2.0 license.

Download the Dataset

Tinyradar comes with two large data sets that you can use to run the experiments. These can be downloaded from:

https://tinyradar.ethz.ch/wp-content/uploads/2020/09/5G.zip
https://tinyradar.ethz.ch/wp-content/uploads/2020/09/11G.zip

The 5G dataset contains recordings of 5 gestures using one sensor, the 11G dataset contains recordings of 11 gestures.
More information can be found on https://tinyradar.ethz.ch/.

Getting started

To get started, please unpack the 11G.zip or 5G.zip raw binary dataset.
Following this, you may unpack the data using the extract_11G.py or extract_5G.zip files, respectively.

To apply the preprocessing mentioned in the paper, please use extract_numpy2feat.py.
For the 11G dataset, please set:

gestures = ["PinchIndex", "PinchPinky", "FingerSlider", "FingerRub",
                     "SlowSwipeRL", "FastSwipeRL", "Push", "Pull", "PalmTilt",
                     "Circle", "PalmHold", "NoHand", "RandomGesture"]
numpyFiles = '../11G/data_npy/'
featureFiles = '../11G/data_feat/'
people = list(range(1,26,1))
singleuserlist = list(map(lambda x: "0_" + str(x) , list(range(1,21,1))))
freq = 160
windowSize = 32
people += singleuserlist

For the 5G dataset, please set:

gestures = ["FingerSlider", "SwipeRL", "PushDown", "PullUp", "PalmTilt"]
numpyFiles = '../5G/data_npy/'
featureFiles = '../5G/data_feat/'
people = [0]
freq = 256
windowSize = 32

Please note that while the dataset is quite large, with 5G.zip and 11G.zip both requiring 11 GB of storage, the unrolled datasets require substantially more space,
with the 11G dataset taking up 219 GB and the 5G dataset taking up 43 GB, using the standard extraction settings.
