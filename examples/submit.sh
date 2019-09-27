#!/bin/bash

# activate environment
. ../../jax_nodes.env

curr_time=$(date "+%F-%H-%M-%S")
scripts=$(pwd)/$curr_time/scripts
results=$(pwd)/$curr_time
mkdir -p $scripts
mkdir -p $results

command="sbatch -p cpu --mem=4G"

regs=(none r0 r1)
lams=(100)

for reg in ${regs[@]}; do
    for lam in ${lams[@]}; do
        # create shell script for job
        params="reg_${reg}_lam_${lam}"
        script="${scripts}/${params}.sh"
        
        # write shell script
        echo "#!/bin/bash" > ${script}
        args="--reg=$reg --lam=$lam --dirname=$results"
        out="> ${results}/$params.o 2> ${results}/$params.e"
        echo "python neural_odes_demo2.py $args $out" >> $script

        # submit job
        $command $script
    done
done
