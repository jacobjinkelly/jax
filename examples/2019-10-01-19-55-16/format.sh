#!/bin/bash

# format results in separate *.o files into one results.txt file

for f in ls *.o; do
    # for some reason "ls" is included in this; ignore it
    if [ $f != "ls" ]; then
        cat $f >> results.txt
    fi
done
