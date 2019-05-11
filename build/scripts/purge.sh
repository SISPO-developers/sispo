#!/bin/bash

echo "Purging software directory: Proceed [y/n]"
read confirmation

if [ $confirmation = y ]
    then rm -r ../software & echo "Purging software directory: done"
    else echo "Nothing done"
fi
