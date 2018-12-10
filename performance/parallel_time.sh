#!/bin/bash
# Do for 1,2,3,4 processors

outputfile="small2x2.csv"
echo "mpirun -n 1" >> $outputfile
mpirun -n 1 python final_output_2.py >> $outputfile
echo "mpirun -n 2" >> $outputfile
mpirun -n 2 python final_output_2.py >> $outputfile
echo "mpirun -n 3" >> $outputfile
mpirun -n 3 python final_output_2.py >> $outputfile
echo "mpirun -n 4" >> $outputfile
mpirun -n 4 python final_output_2.py >> $outputfile

