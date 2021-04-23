HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

SOURCES = cuApplyLayerNorm.cpp cuComputeGradInput.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: test


all: build/cuApplyLayerNorm build/cuComputeGradInput

CXXFLAGS =-g
CXX=$(HIPCC)


build/cuApplyLayerNorm : cuApplyLayerNorm.o
	mkdir -p build
	$(HIPCC) $^ -o $@

build/cuComputeGradInput : cuComputeGradInput.o
	mkdir -p build
	$(HIPCC) $^ -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(OBJECTS)
	rm -f build/cuApplyLayerNorm build/cuComputeGradInput

