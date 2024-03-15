rm -rf result.csv

# test M=N
# for trans in 0 1
# do
#     for MN in 1024 2048 4096 8192 16384 32768
#     do
#         for iter in {1..10}
#         do
#             ./run.sh $trans $MN $MN $MN 1.0 0.0 1 1 prof
#         done
#     done
# done

for trans in 0 1
do
    for M in 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384
    do
        N=$(expr 17408 - $M)
        for iter in {1..10}
        do
            ./run.sh $trans $M $N $M 1.0 0.0 1 1 prof
        done
    done
done
