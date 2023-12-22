# Usage: bash run_sim.sh <Batch> <M> <K> <N>

# source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Modify op_test_frame and simulator path
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/op_test_frame:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH

mkdir -p model

python3 run_sim.py 1 512 512 512 0 1