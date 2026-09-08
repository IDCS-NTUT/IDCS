DEEPSTREAM_ROOT ?= /opt/nvidia/deepstream/deepstream
CUDA_ROOT ?= /usr/local/cuda
CXX ?= g++
TARGET := libnvdsinfer_yolo26_parser.so
SOURCE := nvdsinfer_yolo26_parser.cpp
CXXFLAGS := -O3 -Wall -Wextra -std=c++17 -fPIC \
	-I$(DEEPSTREAM_ROOT)/sources/includes -I$(CUDA_ROOT)/include
LDFLAGS := -shared

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
