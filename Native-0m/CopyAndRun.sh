#!/bin/bash

for i in {1..40}
do
	mkdir -p $i
	starta=`date`
	echo "Running Replica $i $starta"
	cp -r input sub.slurm Finish-Sub.slurm openmichrolib.py michromanalysislib.py Native-0m.py Native-Finish.py  $i
	cp OrtdStructures0m/Ortd_0m_"$i".pdb $i/ortd.pdb
	cd $i
	pwd
	jid1=$(sbatch sub.slurm | sed 's/Submitted batch job //')
	sbatch --dependency=afterany:$jid1 Finish-Sub.slurm	
	end=`date`
	echo "Submitted Replica $i $end"
	cd ../
	pwd
done
