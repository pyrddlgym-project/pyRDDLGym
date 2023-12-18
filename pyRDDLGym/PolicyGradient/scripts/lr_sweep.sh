#! /bin/bash

CONFIG=$1
SAVETO=$2
DIM=$3
INST=$4
NITERS=$5

#for LR in 0.01 0.03 0.1 0.3 0.7 1.1 1.3 1.7 2.1 2.3 2.7 3.1 3.3 3.7 4.1 4.3 4.7
for LR in 0.01 0.03 0.05 0.1 0.3 0.5 0.7 0.9 1.1 1.3
do
    python3 ../main.py $CONFIG --save-to $SAVETO --learning-rate $LR -d $DIM -i $INST --num-iters $NITERS --verbose 0 &
done
