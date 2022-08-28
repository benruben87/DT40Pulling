#!/bin/bash
# 1: Number of Force Values (Not Including 0), 2- ForceJump

N=$1
J=$2

for i in {1..40}
do
	mkdir -p $i
	cp michromanalysislib.py $i
	starta=`date`
	echo "Running Replica $i $starta"
	
	for (( fnum=0 ; fnum<=$N ; fnum++ ))
	do
		sed 's/FORCENUM/'${fnum}'/g;s/FJUMP/'$J'/g;s/REPNUM/'$i'/g;' CF-Finish-0m.py > $i/CF_"${fnum}"_Finish_0m.py
                sed 's/FORCENUM/'${fnum}'/g;s/FJUMP/'$J'/g;s/REPNUM/'$i'/g;' CF-Finish-Sub.slurm > $i/CF_"${fnum}"_Finish_0m_Sub.slurm
		cd $i
        	pwd
		sbatch CF_${fnum}_Finish_0m_Sub.slurm
		cd ..
	done
	end=`date`
	echo "Submitted Replica $i $end"
	pwd
done
