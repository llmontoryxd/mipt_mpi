#!/bin/bash

#PBS -l walltime=00:10:00,nodes=2:ppn=1
#PBS -N latency_diff_nodes
#PBS -q batch

cd $PBS_O_WORKDIR
rm -f latency_diff_nodes.txt
for (( i = 1; i <= 70; i++))
do
mpirun --hostfile $PBS_NODEFILE -np 2 -pernode ./latency_diff_nodes $i
done
rm -f latency_diff_nodes.o*
rm -f latency_diff_nodes.e*
