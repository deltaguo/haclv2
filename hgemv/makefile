CANN_DIR=$(ASCEND_HOME_PATH)
OP_NAME = hgemv

all: $(OP_NAME).o main.o $(OP_NAME)

$(OP_NAME).o: $(OP_NAME).cce $(OP_NAME)_utils.h
	ccec -std=c++17 -c -x cce -O2 $(OP_NAME).cce --cce-aicore-arch=dav-c220 \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw -I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface -I${CANN_DIR}/compiler/tikcpp/tikcfw/impl \
	-o build/$(OP_NAME).o
main.o: main.cpp
	g++ -O2 -c main.cpp -fPIC -I$(CANN_DIR)/include -L$(CANN_DIR)/lib64 -o build/main.o -lascendcl
	
$(OP_NAME): $(OP_NAME).cce main.cpp
	ccec -o build/$(OP_NAME) build/main.o build/$(OP_NAME).o --cce-fatobj-link -L$(CANN_DIR)/lib64 -lascendcl -lruntime -lstdc++

ca:	$(OP_NAME).cce
	ccec -std=c++17 -c -O2 ${OP_NAME}.cce -DCAMODEL -o build/${OP_NAME}_ca.o \
	--cce-aicore-arch=dav-c220 --cce-aicore-only \
	-mllvm -cce-aicore-record-overflow=false \
	-mllvm -cce-aicore-addr-transform \
	-mllvm --cce-aicore-jump-expand=true


clean:
	rm -rf build/*
	rm -rf data/*