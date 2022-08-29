#!/bin/bash

for i in {1..20}
do
	mkdir -p $i
	starta=`date`
	echo "Running Replica $i $starta"
	rm $i/Open15m-Finish.py $i/michromanalysislib.py
	cp Open15m-Finish.py michromanalysislib.py $i
	cd $i
	pwd
	
	fin=$(sbatch Finish-Sub.slurm | sed 's/Submitted batch job //')
	end=`date`
	echo "Submitted Replica $i $end"
	cd ../
	pwd
done
