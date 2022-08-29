#!/bin/bash

for i in {1..3}
do
	mkdir -p $i
	starta=`date`
	echo "Running Replica $i $starta"
	cp -r input  openmichrolib.py Open15m-Finish.py OrtdStructures15m/Ortd_15m_$i.pdb $i
    sed 's/REPNUM/'${i}'/g' Open15m.py > $i/Open15m.py
	cd $i
	pwd
	python Open15m.py
    	python Open15m-Finish.py
	end=`date`
	echo "Finished Replica $i $end"
	cd ../
	pwd
done
