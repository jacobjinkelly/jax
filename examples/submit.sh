#!/bin/bash

scripts=$1
results=$2
reg=$3
lam=$4

command="sbatch -p cpu --mem=4G"

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

