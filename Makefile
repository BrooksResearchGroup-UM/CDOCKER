HOME=/home/xqding
OpenMM_INSTALL_DIR=$(HOME)/apps/openmmDev
CUFFTDIR="-I/export/apps/CUDA/7.5"

LIB_DIR=-L$(OpenMM_INSTALL_DIR)/lib -L$(CUFFTDIR)/lib -L$(CUFFTDIR)/lib64
INCLUDE_DIR=-I$(OpenMM_INSTALL_DIR)/include -I$(CUFFTDIR)/include -I./include
LIBS= -lOpenMM -lcufft -lm
CFLAGS=-g 
CC = nvcc

BUILD = ./build
SOURCE = ./src

programs = $(BUILD)/main $(BUILD)/cpuNaive
objects = $(BUILD)/ReadCrd.o $(BUILD)/ReadGrids.o $(BUILD)/ReadQuaternions.o $(BUILD)/Rotate.o $(BUILD)/QuaternionMultiply.o $(BUILD)/FillLigandGrid.o $(BUILD)/GetMinCoors.o $(BUILD)/GetNonbondedParameters.o $(BUILD)/GetIdxOfAtomsForVdwRadius.o


all: $(programs) copy

$(BUILD)/%.o: $(SOURCE)/%.cpp
	$(CC) $(CFLAGS) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%.o: $(SOURCE)/%.cu
	$(CC) $(CFLAGS) $(INCLUDE_DIR) -c $< -o $@

$(programs): %: %.o $(objects)
	$(CC) $(LIB_DIR) $(LIBS) $(objects) $< -o $@

.PHONY: clean copy
clean:
	rm -rf $(objects) $(programs) $(BUILD)/main.o *~
copy:
	cp $(programs) ./test/
