# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

# Source files for main program
# SRCS = atomic_operations.cpp attention.cpp embedding_layer.cpp feedforward_layer.cpp \
#        layer_normalization.cpp linear_layer.cpp multi_head_attention.cpp positional_encoding.cpp \
#        transformer_block.cpp transformer_model.cpp main.cpp

SRCS = atomic_operations.cpp attention.cpp embedding_layer.cpp feedforward_layer.cpp \
       layer_normalization.cpp linear_layer.cpp multi_head_attention.cpp positional_encoding.cpp \
       transformer_block.cpp main.cpp

# Object files for main program
OBJS = $(SRCS:.cpp=.o)

# Executable name for main program
EXEC = transformer_program

# Test source files
# TEST_SRCS = test_atomic_operations.cpp test_activation_functions.cpp test_linear_layer.cpp \
#             test_positional_encoding.cpp test_attention.cpp test_feedforward_layer.cpp \
#             test_layer_normalization.cpp test_embedding_layer.cpp test_multi_head_attention.cpp \
#             test_transformer_block.cpp test_transformer_model.cpp

TEST_SRCS = test_atomic_operations.cpp test_activation_functions.cpp test_linear_layer.cpp \
            test_positional_encoding.cpp test_attention.cpp test_feedforward_layer.cpp \
            test_layer_normalization.cpp test_embedding_layer.cpp test_multi_head_attention.cpp \
            test_transformer_block.cpp

# Test executables
# TESTS = test_atomic_operations test_activation_functions test_linear_layer \
#         test_positional_encoding test_attention test_feedforward_layer \
#         test_layer_normalization test_embedding_layer test_multi_head_attention \
#         test_transformer_block test_transformer_model

TESTS = test_atomic_operations test_activation_functions test_linear_layer \
        test_positional_encoding test_attention test_feedforward_layer \
        test_layer_normalization test_embedding_layer test_multi_head_attention \
        test_transformer_block

# Default target
all: $(EXEC) $(TESTS)

# Build main executable
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile object files for main program
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test executables
test_atomic_operations: test_atomic_operations.cpp atomic_operations.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

test_activation_functions: test_activation_functions.cpp atomic_operations.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

test_linear_layer: test_linear_layer.cpp linear_layer.cpp atomic_operations.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

test_positional_encoding: test_positional_encoding.cpp positional_encoding.cpp atomic_operations.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

test_attention: test_attention.cpp attention.cpp atomic_operations.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

test_feedforward_layer: test_feedforward_layer.cpp feedforward_layer.cpp linear_layer.cpp atomic_operations.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

test_layer_normalization: test_layer_normalization.cpp layer_normalization.cpp atomic_operations.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

test_embedding_layer: test_embedding_layer.cpp embedding_layer.cpp atomic_operations.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

test_multi_head_attention: test_multi_head_attention.cpp multi_head_attention.cpp attention.cpp atomic_operations.cpp \
                           linear_layer.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

test_transformer_block: test_transformer_block.cpp transformer_block.cpp multi_head_attention.cpp feedforward_layer.cpp \
                        attention.cpp linear_layer.cpp layer_normalization.cpp atomic_operations.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

# test_transformer_model: test_transformer_model.cpp transformer_model.cpp transformer_block.cpp multi_head_attention.cpp \
#                         feedforward_layer.cpp attention.cpp linear_layer.cpp positional_encoding.cpp \
#                         embedding_layer.cpp layer_normalization.cpp atomic_operations.cpp
# 	$(CXX) $(CXXFLAGS) -o $@ $^

# Target to build all tests
tests: $(TESTS)

# Clean target
clean:
	rm -f $(EXEC) $(OBJS) $(TESTS)
