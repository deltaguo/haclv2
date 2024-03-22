rm -rf prof/*
rm -rf result.csv
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# for trans in 0 1
# #for trans in 1
# do
#     for M in 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384
# #    for M in 10240 11264 #12288 13312 14336 15360 16384
#     do
#         N=$(expr 17408 - $M)
#         python hgemv.py --trans $trans -M $M -N $N
#         python3 prof.py --filepath `find ./prof/profiler -name "aicore_intermediate_0_detail.csv"` --trans $trans -M $M -N $N
#     done
# done


for trans in 0 1
do
    for M in 1024 2048 4096 8192 16384 32768
    do
        python hgemv.py --trans $trans -M $M -N $M
        python prof.py --filepath `find ./prof/profiler -name "aicore_intermediate_0_detail.csv"` --trans $trans -M $M -N $M
    done
done

# python hgemv.py --trans 0 -M 4096 -N 4096
# python prof.py --filepath `find ./prof/profiler -name "aicore_intermediate_0_detail.csv"` --trans 1 -M 4096 -N 4096