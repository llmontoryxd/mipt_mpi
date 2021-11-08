#!/bin/bash

#PBS -l walltime=00:10:00,nodes=7:ppn=4
#PBS -N lgca
#PBS -q batch

cd $PBS_O_WORKDIR
rm -f lgca.txt
for (( i = 1; i <= 28; i++ ))
do
mpirun --hostfile $PBS_NODEFILE -np $i ./main 0 0 1
done
rm -f lgca.o*
rm -f lgca.e*
