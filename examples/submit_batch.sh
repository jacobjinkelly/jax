#!/bin/bash

file=$1
node=$2
mem=$3

# activate environment
. ../../jax_nodes.env

curr_time=$(date "+%F-%H-%M-%S")
scripts=$(pwd)/$curr_time/scripts
results=$(pwd)/$curr_time
mkdir -p $scripts
mkdir -p $results

args="$file $node $mem $scripts $results"

r0_lam_file="r0_lams.txt"

while IFS= read -r line
do
    ./submit.sh $args r0 $line
done < "$r0_lam_file"

r1_lam_file="r1_lams.txt"

while IFS= read -r line
do
    ./submit.sh $args $r1 $line
done < "$r1_lam_file"

./submit.sh $args none 0
