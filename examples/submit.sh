#!/bin/bash

file=$1
node=$2
mem=$3
scripts=$4
results=$5
reg=$6
lam=$7

if [ $node = "gpu" ]; then
    command="sbatch -p p100 --mem=$mem --gres=gpu:1"
else
    command="sbatch -p cpu --mem=$mem"
fi

# create shell script for job
params="reg_${reg}_lam_${lam}"
script="${scripts}/${params}.sh"

# write shell script
echo "#!/bin/bash" > ${script}
args="--reg=$reg --lam=$lam --dirname=$results"
out="> ${results}/$params.o 2> ${results}/$params.e"
echo "python $file $args $out" >> $script

# submit job
$command $script

