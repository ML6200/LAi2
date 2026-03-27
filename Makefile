# LAi - Lightweight AI Assistant
# Pure C++ LLM inference with optional Metal GPU acceleration

CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS :=

# Detect platform and architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Platform-specific flags
ifeq ($(UNAME_S),Darwin)
    # macOS - enable Metal GPU backend
    CXXFLAGS += -DLAI_METAL=1
    LDFLAGS += -framework Metal -framework Foundation
    ifeq ($(UNAME_M),arm64)
        # Apple Silicon - NEON is implicit
        ARCH_FLAGS := -mcpu=apple-m1
    else
        # Intel Mac
        ARCH_FLAGS := -march=native -mavx2 -mfma
    endif
else ifeq ($(UNAME_S),Linux)
    ifeq ($(UNAME_M),aarch64)
        # ARM64 Linux
        ARCH_FLAGS := -march=armv8-a+simd
    else
        # x86-64 Linux
        ARCH_FLAGS := -march=native -mavx2 -mfma
    endif
else
    # Windows/MinGW or other
    ARCH_FLAGS := -march=native
endif

# Build configurations
DEBUG_FLAGS := -g -O0 -DDEBUG -fsanitize=address,undefined
RELEASE_FLAGS := -O3 -DNDEBUG -flto -ffast-math $(ARCH_FLAGS)

# Directories
SRC_DIR := src
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp) \
        $(wildcard $(SRC_DIR)/core/*.cpp) \
        $(wildcard $(SRC_DIR)/model/*.cpp) \
        $(wildcard $(SRC_DIR)/tokenizer/*.cpp) \
        $(wildcard $(SRC_DIR)/inference/*.cpp) \
        $(wildcard $(SRC_DIR)/cli/*.cpp) \
        $(wildcard $(SRC_DIR)/backend/*.cpp)

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
DEBUG_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/debug/%.o,$(SRCS))

# Metal sources (macOS only)
ifeq ($(UNAME_S),Darwin)
    METAL_SRCS := $(wildcard $(SRC_DIR)/backend/*.mm)
    METAL_OBJS := $(patsubst $(SRC_DIR)/%.mm,$(OBJ_DIR)/%.o,$(METAL_SRCS))
    METAL_DEBUG_OBJS := $(patsubst $(SRC_DIR)/%.mm,$(OBJ_DIR)/debug/%.o,$(METAL_SRCS))
endif

# Headers
HDRS := $(wildcard $(SRC_DIR)/*.h) \
        $(wildcard $(SRC_DIR)/core/*.h) \
        $(wildcard $(SRC_DIR)/model/*.h) \
        $(wildcard $(SRC_DIR)/tokenizer/*.h) \
        $(wildcard $(SRC_DIR)/inference/*.h) \
        $(wildcard $(SRC_DIR)/cli/*.h) \
        $(wildcard $(SRC_DIR)/backend/*.h)

# Target
TARGET := lai
DEBUG_TARGET := lai_debug

# Include paths
INCLUDES := -I$(SRC_DIR)

.PHONY: all release debug clean test bench valgrind help

all: release

# Release build
release: CXXFLAGS += $(RELEASE_FLAGS)
release: $(TARGET)

$(TARGET): $(OBJS) $(METAL_OBJS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(OBJS) $(METAL_OBJS) -o $@ -lpthread $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Objective-C++ compilation (Metal backend)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.mm $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(DEBUG_TARGET)

$(DEBUG_TARGET): $(DEBUG_OBJS) $(METAL_DEBUG_OBJS)
	@echo "Linking $(DEBUG_TARGET)..."
	$(CXX) $(CXXFLAGS) $(DEBUG_OBJS) $(METAL_DEBUG_OBJS) -o $@ -lpthread $(LDFLAGS)

$(OBJ_DIR)/debug/%.o: $(SRC_DIR)/%.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/debug/%.o: $(SRC_DIR)/%.mm $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(DEBUG_TARGET)

# Run tests
test: debug
	./$(DEBUG_TARGET) --test

# Memory check with Valgrind
valgrind: debug
	valgrind --leak-check=full --show-leak-kinds=all ./$(DEBUG_TARGET) --test

# Benchmark
bench: release
	./$(TARGET) --bench

# Help
help:
	@echo "LAi - Lightweight AI Assistant"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build release version"
	@echo "  make release  - Build optimized release version"
	@echo "  make debug    - Build debug version with sanitizers"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test     - Run tests"
	@echo "  make valgrind - Run with Valgrind memory checker"
	@echo "  make bench    - Run benchmarks"
	@echo ""
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CXX)"
	@echo "Arch flags: $(ARCH_FLAGS)"
