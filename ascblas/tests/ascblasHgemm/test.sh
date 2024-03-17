#!/bin/bash
# make
# ./run.sh 0 0 512 256 1024 512 1024 512
# ./run.sh 0 0 22 33 44 22 44 22
# ./run.sh 0 0 23 23 23 23 23 23
# ./run.sh 0 0 222 333 444 222 444 222
# ./run.sh 0 0 233 233 233 233 233 233
# ./run.sh 0 0 3333 7777 4444 3333 4444 7777
# ./run.sh 0 0 3333 7777 4444 7777 7777 7777
# ./run.sh 0 0 3333 7777 4451 3333 4451 7777
# ./run.sh 0 0 32 32 32 32 32 32
# ./run.sh 0 0 8192 8192 8192 8192 8192 8192 prof
# ./run.sh 0 0 10000 10000 10000 10000 10000 10000 prof

rm -rf result.csv

for trans in 0
do
    for M in 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384
    do
        N=$(expr 17408 - $M)
        ./run.sh 0 0 $M 1 $N $M $N $M prof
    done
done

for trans in 1
do
    for M in 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384
    do
        N=$(expr 17408 - $M)
        ./run.sh 0 0 1 $N $M 1 $M 1 prof
    done
done

for trans in 0
do
    for MN in 1024 2048 4096 8192 16384 32768
    do
        ./run.sh 0 0 $MN 1 $MN $MN $MN $MN prof
    done
done

for trans in 1
do
    for MN in 1024 2048 4096 8192 16384 32768
    do
        ./run.sh 0 0 1 $MN $MN 1 $MN 1 prof
    done
done
