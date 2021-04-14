#!/bin/bash

# Modified: https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html
IMAGE_URL="http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip"
LABEL_URL="http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip"
IMAGE_ZIP="701_StillsRaw_full.zip"
LABEL_ZIP="LabeledApproved_full.zip"

echo "Your download path: $1."
mkdir -p $1
cd $1

echo "Downloading CamVid Image (557 MB)."
wget $IMAGE_URL

echo "Downloading CamVid Label (16 MB)."
wget $LABEL_URL

echo "Unzip images."
unzip $IMAGE_URL
mv "701_StillsRaw_full" images

echo "Unzip labels."
mkdir labels
mv $LABEL_ZIP labels/
unzip "./labels/$LABEL_ZIP"
cd -
