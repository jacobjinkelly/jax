#!/bin/bash

# activate environment
. ../../jax_nodes.env

curr_time=$(date "+%F-%H-%M-%S")
scripts=$(pwd)/$curr_time/scripts
results=$(pwd)/$curr_time
mkdir -p $scripts
mkdir -p $results

./submit.sh $scripts $results r0 1.596
./submit.sh $scripts $results r1 76.880
