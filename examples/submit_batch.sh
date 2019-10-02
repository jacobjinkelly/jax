#!/bin/bash

# activate environment
. ../../jax_nodes.env

curr_time=$(date "+%F-%H-%M-%S")
scripts=$(pwd)/$curr_time/scripts
results=$(pwd)/$curr_time
mkdir -p $scripts
mkdir -p $results

lam_file="lams.txt"

while IFS= read -r line
do
    ./submit.sh $scripts $results r0 $line
    ./submit.sh $scripts $results r1 $line
done < "$lam_file"

./submit.sh $scripts $results none 0
