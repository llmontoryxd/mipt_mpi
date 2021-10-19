#!/bin/bash

#PBS -l walltime=00:10:00,nodes=1:ppn=2
#PBS -N latency
#PBS -q batch

cd $PBS_O_WORKDIR
rm -f latency.txt
for (( i = 0; i <= 70; i++))
do
mpirun --hostfile $PBS_NODEFILE -np 2 ./latency $i
done
rm -f latency.o*
rm -f latency.e*
