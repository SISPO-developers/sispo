#!/bin/bash

echo "Download ucac4 star catalogue start"

# Creating dir
cd ../..
[[ -d data ]] || mkdir data
cd data

[[ -d ucac4 ]] || mkdir ucac4
cd ucac4

# Download data
wget -r --no-parent -P . http://casdc.china-vo.org/mirror/UCAC/UCAC4/u4b/ &
wget -r --no-parent -P . http://casdc.china-vo.org/mirror/UCAC/UCAC4/u4i/
wait

# Moving files
mv -r casdc.china-vo.org/mirror/UCAC/UCAC4/u4b u4b
mv -r casdc.china-vo.org/mirror/UCAC/UCAC4/u4i u4i

rm -r casdc.china-vo.org/

echo "Download ucac4 star catalogue done"
