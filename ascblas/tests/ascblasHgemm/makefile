CANN_DIR=$(ASCEND_HOME_PATH)
OP_NAME=ascblasHgemm
include_dir=-I../ -I../../include
all: $(OP_NAME)_aic.o $(OP_NAME)_aiv.o $(OP_NAME).o $(OP_NAME)

# 编译 cce 文件
# cce 文件需要AIC和AIV分别编译，最后再结合到一起
$(OP_NAME)_aic.o: $(OP_NAME).cce
	ccec -std=c++17 -c -x cce -O2 $(OP_NAME).cce --cce-aicore-arch=dav-c220-cube --cce-aicore-only \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw -I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface -I${CANN_DIR}/compiler/tikcpp/tikcfw/impl ${include_dir} \
	-o build/$(OP_NAME)_aic.o

$(OP_NAME)_aiv.o: $(OP_NAME).cce
	ccec -std=c++17 -c -x cce -O2 $(OP_NAME).cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw -I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface -I${CANN_DIR}/compiler/tikcpp/tikcfw/impl ${include_dir} \
	-o build/$(OP_NAME)_aiv.o

$(OP_NAME).o: $(OP_NAME).cce
	ld.lld -m aicorelinux -Ttext=0 -z separate-code build/$(OP_NAME)_aic.o build/$(OP_NAME)_aiv.o \
    --allow-multiple-definition -static -o build/$(OP_NAME).o

$(OP_NAME): main.cpp
	g++ -O2 -fpic ../../src/handle.cc main.cpp -I$(CANN_DIR)/include  ${include_dir} -L$(CANN_DIR)/lib64 -o build/$(OP_NAME) -lascendcl -lruntime -lstdc++

# 用CAmodel的编译选项
ca: $(OP_NAME)_aic_ca.o $(OP_NAME)_aiv_ca.o $(OP_NAME)_ca.o

$(OP_NAME)_aic_ca.o: $(OP_NAME).cce
	ccec -std=c++17 -c -O2 ${OP_NAME}.cce ${include_dir} -DCAMODEL -o build/${OP_NAME}_aic_ca.o \
	--cce-aicore-arch=dav-c220-cube --cce-aicore-only \
	-mllvm -cce-aicore-record-overflow=false \
	-mllvm -cce-aicore-addr-transform \
	-mllvm --cce-aicore-jump-expand=true

$(OP_NAME)_aiv_ca.o: $(OP_NAME).cce
	ccec -std=c++17 -c -O2 ${OP_NAME}.cce ${include_dir} -DCAMODEL -o build/${OP_NAME}_aiv_ca.o \
	--cce-aicore-arch=dav-c220-vec --cce-aicore-only \
	-mllvm -cce-aicore-record-overflow=false \
	-mllvm -cce-aicore-addr-transform \
	-mllvm --cce-aicore-jump-expand=true
$(OP_NAME)_ca.o: $(OP_NAME).cce
	ld.lld -Ttext=0 build/${OP_NAME}_aic_ca.o build/${OP_NAME}_aiv_ca.o -static -o build/${OP_NAME}_ca.o

clean:
	rm -rf build/*