#!/bin/bash
echo "Download CINIC-10"
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz --no-check-certificate
echo "Unzip CINIC-10"
mkdir CINIC10
tar -xvf CINIC-10.tar.gz -C CINIC10
echo "Remove zip file."
rm CINIC-10.tar.gz