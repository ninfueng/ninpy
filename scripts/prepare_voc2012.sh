#!/bin/bash

set -e
echo "Create ./datasets."
mkdir ./datasets
cd ./datasets

echo "Install Pascal VOC 2012."
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
echo "Install Pascal VOC augmentation 2012."
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

echo "Untar Pascal VOC 2012."
tar -xvf VOCtrainval_11-May-2012.tar
echo "Untar Pascal VOC augmentation 2012."
tar -xvf benchmark.tgz

echo "Rename folder benchmark_RELEASE to VOCaug."
mv benchmark_RELEASE VOCaug
echo "Create ./VOCaug/dataset/trainval.txt."
cat ./VOCaug/dataset/train.txt ./VOCaug/dataset/val.txt > ./VOCaug/dataset/trainval.txt
echo "Run successfully."

cd -
