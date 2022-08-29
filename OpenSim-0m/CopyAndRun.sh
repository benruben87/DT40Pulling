#!/bin/bash

for i in {1..20}
do
	mkdir -p $i
	starta=`date`
	echo "Running Replica $i $starta"
	cp -r input sub.slurm Finish-Sub.slurm  openmichrolib.py michromanalysislib.py Open0m-Finish.py OrtdStructures0m/Ortd_0m_$i.pdb $i
	sed 's/REPNUM/'${i}'/g' Open0m.py > $i/Open0m.py
	cd $i
	pwd
	
	jrun=$(sbatch sub.slurm | sed 's/Submitted batch job //')
	fin=$(sbatch --dependency=afterany:$jrun Finish-Sub.slurm | sed 's/Submitted batch job //')
	end=`date`
	echo "Submitted Replica $i $end"
	cd ../
	pwd
done
