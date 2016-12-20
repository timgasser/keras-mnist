#!/bin/bash

declare -a files=("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
       "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
       "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" 
       "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")

echo "Deleting existing files"
rm -f *.gz
rm -f *-ubyte
rm -f *.pkl

# Download gzip files from Yann LeCun's website
for i in "${files[@]}"
do
    wget "$i"
#    echo "Downloading $i"
done

# Extract the files and move downloads to backup
gunzip -k *.gz
mkdir -p archive
mv -f *.gz archive

