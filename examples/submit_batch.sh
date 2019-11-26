#!/bin/bash

file=$1
node=$2
mem=$3

# activate environment
. ../../jax_nodes.env

# set up directory to store results
curr_time=$(date "+%F-%H-%M-%S")
scripts=$(pwd)/$curr_time/scripts
results=$(pwd)/$curr_time
mkdir -p $scripts
mkdir -p $results

args="$file $node $mem $scripts $results"

r0_lam_file="r0_lams.txt"
r1_lam_file="r1_lams.txt"

# save config
cp $file $results
cp $r0_lam_file $results
cp $r1_lam_file $results
echo $args > $results/args.txt

while IFS= read -r line
do
    ./submit.sh $args r0 $line
done < "$r0_lam_file"


while IFS= read -r line
do
    ./submit.sh $args $r1 $line
done < "$r1_lam_file"

./submit.sh $args none 0
