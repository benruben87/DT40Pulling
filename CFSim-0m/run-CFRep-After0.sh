#!/bin/bash
# 1: Number of Force Values (Not Including 0), 2- ForceJump

N=$1
J=$2

for i in {1..40}
do
	mkdir -p $i
	starta=`date`
	echo "Running Replica $i $starta"
	#cp -r input  openmichrolib.py NoForce_0m.py michromanalysislib.py  $i
	#cp OrtdStructures0m/Ortd_0m_"$i".pdb $i/ortd.pdb
	
        #sed 's/FJUMP/'$J'/g;s/REPNUM/'$i'/g;' NoForceSub.slurm > $i/NoForceSub.slurm

	#cd $i
	#jid1=$(sbatch NoForceSub.slurm | sed 's/Submitted batch job //')
	#cd ..

	for (( fnum=1 ; fnum<=$N ; fnum++ ))
	do
		sed 's/FORCENUM/'${fnum}'/g;s/FJUMP/'$J'/g;s/REPNUM/'$i'/g;' CF-0m.py > $i/CF_"${fnum}"_0m.py
        	sed 's/FORCENUM/'${fnum}'/g;s/FJUMP/'$J'/g;s/REPNUM/'$i'/g;' CFSub.slurm > $i/CF_"${fnum}"_0m_Sub.slurm

		sed 's/FORCENUM/'${fnum}'/g;s/FJUMP/'$J'/g;s/REPNUM/'$i'/g;' CF-Finish-0m.py > $i/CF_"${fnum}"_Finish_0m.py
                sed 's/FORCENUM/'${fnum}'/g;s/FJUMP/'$J'/g;s/REPNUM/'$i'/g;' CF-Finish-Sub.slurm > $i/CF_"${fnum}"_Finish_0m_Sub.slurm
		cd $i
        	pwd
		jid2=$(sbatch CF_${fnum}_0m_Sub.slurm | sed 's/Submitted batch job //')
		sbatch --dependency=afterok:$jid2 CF_${fnum}_Finish_0m_Sub.slurm
		cd ..
	done
	end=`date`
	echo "Submitted Replica $i $end"
	pwd
done
