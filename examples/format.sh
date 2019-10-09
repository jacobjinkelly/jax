#!/bin/bash

# format results in separate *.o files into one results.txt file

dir=$1

for f in ls $dir/*.o; do
    # for some reason "ls" is included in this; ignore it
    if [ $f != "ls" ]; then
        cat $f >> $dir/results.txt
    fi
done
