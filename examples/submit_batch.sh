#!/bin/bash

# activate environment
. ../../jax_nodes.env

curr_time=$(date "+%F-%H-%M-%S")
scripts=$(pwd)/$curr_time/scripts
results=$(pwd)/$curr_time
mkdir -p $scripts
mkdir -p $results

r0_lam_file="r0_lams.txt"

while IFS= read -r line
do
    ./submit.sh $scripts $results r0 $line
done < "$r0_lam_file"

r1_lam_file="r1_lams.txt"

while IFS= read -r line
do
    ./submit.sh $scripts $results r1 $line
done < "$r1_lam_file"

./submit.sh $scripts $results none 0
