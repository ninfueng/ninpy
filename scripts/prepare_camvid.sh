#!/bin/bash

# Modified: https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html
IMAGE_URL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip'
LABEL_URL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip'

echo "Your download path: $1."
mkdir -p $1
cd $1

echo "Downloading CamVid Image."
wget $IMAGE_URL

echo "Downloading CamVid Label."
wget $LABEL_URL

cd -
