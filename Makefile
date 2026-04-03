CXX = nvcc

CXXFLAGS = --std c++17 -Wno-deprecated-gpu-targets
CXXFLAGS += $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
LDFLAGS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)
LDFLAGS += -lnppif -lnppc -lnppig -lnppist -lnppi -lnppisu -lcuda

TARGET = batch_image_filter.exe
SRC = batch_image_filter.cu

.PHONY: all clean build run setup

all: clean build

build: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) processing_log.csv
	rm -rf output/*

run: build
	./$(TARGET) -i sample_images -o output -w 5 -h 5

run-large:
	./$(TARGET) -i sample_images -o output -w 11 -h 11

setup:
	bash generate_test_images.sh
	mkdir -p output
