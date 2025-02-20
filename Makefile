# Makefile to use c++ 20, removes our errors, just follow these instructions in shell:
# step1: make
# step2: ./CPlusInfer
# step3: make clean

# Compiler
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -Iheaders

# Source files
SRCS = main.cpp Tensor.cpp
OBJS = $(SRCS:.cpp=.o)

# Executable name
TARGET = CplusInfer

# Default rule
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compilation rule
%.o: %.cpp headers
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)