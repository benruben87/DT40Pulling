#!/bin/bash

for i in {1..40}
do
	starta=`date`
	echo "Running Replica $i $starta"
	cp  -r input Finish-Sub.slurm Native-Finish.py michromanalysislib.py $i
	cd $i
	pwd
	sbatch Finish-Sub.slurm
	end=`date`
	echo "Submitted Replica $i $end"
	cd ../
	pwd
done
