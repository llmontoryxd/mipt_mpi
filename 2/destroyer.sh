#!/bin/bash

for (( i = 1; i <= 10; i++ )) 
do
qsub lgca.sh
sleep 240
done
