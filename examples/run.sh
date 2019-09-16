#!/bin/bash

curr_time=$(date "+%F-%H-%M-%S")

dir=$curr_time
mkdir $dir

regs=(none state dynamics)
lams=(1 100 1000 10000)

for reg in ${regs[@]}; do
    for lam in ${lams[@]}; do
        python neural_odes.py --reg $reg --lam $lam > $dir/${reg}_lam${lam}.txt
    done
done
