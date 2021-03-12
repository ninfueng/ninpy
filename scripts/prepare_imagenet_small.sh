#!/bin/bash
# Modified: https://raw.githubusercontent.com/openai/vdvae/main/setup_imagenet.sh
# Used for generative model. Therefore, this does not contain labels.

if [ "$1" == "imagenet32" ]; then
    echo "Downloading imagenet32"
    wget http://www.image-net.org/small/train_32x32.tar
    wget http://www.image-net.org/small/valid_32x32.tar
    tar -xvf train_32x32.tar
    tar -xvf valid_32x32.tar
    python files_to_npy.py train_32x32/ imagenet32-train.npy
    python files_to_npy.py valid_32x32/ imagenet32-valid.npy
    echo "Removing train valid .tar and folders"
    rm train_32x32.tar
    rm valid_32x32.tar
    rm -rf train_32x32
    rm -rf valid_32x32


elif [ "$1" == "imagenet64" ]; then
    echo "Downloading imagenet64"
    wget http://www.image-net.org/small/train_64x64.tar
    wget http://www.image-net.org/small/valid_64x64.tar
    tar -xvf train_64x64.tar
    tar -xvf valid_64x64.tar
    python files_to_npy.py train_64x64/ imagenet64-train.npy
    python files_to_npy.py valid_64x64/ imagenet64-valid.npy
    echo "Removing train valid .tar and folders"
    rm train_64x64.tar
    rm valid_64x64.tar
    rm -rf train_32x32
    rm -rf valid_32x32

else
    echo "Please pass the string imagenet32 or imagenet64 as an argument"
fi

