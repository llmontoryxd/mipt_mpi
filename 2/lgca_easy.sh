#!/bin/bash

#PBS -l walltime=00:10:00,nodes=7:ppn=4
#PBS -N lgca
#PBS -q batch

cd $PBS_O_WORKDIR
rm -f lgca.txt
mpirun --hostfile $PBS_NODEFILE -np 4 ./main 0 1 0
rm -f lgca.o*
rm -f lgca.e*
