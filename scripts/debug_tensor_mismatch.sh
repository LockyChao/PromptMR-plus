#!/bin/bash

# Debug script for PyTorch tensor dimension mismatches
# Usage: ./debug_tensor_mismatch.sh [error_log_file]

set -e

error_file=""
if [ "$#" -eq 1 ]; then
    error_file="$1"
    if [ ! -f "$error_file" ]; then
        echo "Error: File $error_file does not exist."
        exit 1
    fi
fi

extract_tensor_info() {
    local log_content="$1"
    
    # Extract expected and actual shapes
    expected_shape=$(echo "$log_content" | grep -o "Expected.*size" | sed 's/Expected.*size \([^,]*\).*/\1/')
    actual_shape=$(echo "$log_content" | grep -o "but got.*size" | sed 's/but got.*size \([^,]*\).*/\1/')
    
    echo "Expected shape: $expected_shape"
    echo "Actual shape: $actual_shape"
    
    # Extract operation that caused the mismatch
    operation=$(echo "$log_content" | grep -A 1 "RuntimeError:" | grep -v "RuntimeError:" | head -1 | tr -d ' ')
    if [ ! -z "$operation" ]; then
        echo "Operation: $operation"
    fi
    
    # Suggest possible solutions
    suggest_solutions "$expected_shape" "$actual_shape"
}

suggest_solutions() {
    local expected="$1"
    local actual="$2"
    
    echo -e "\n=== POSSIBLE SOLUTIONS ==="
    
    # Convert bracket notation to individual dimensions
    expected_dims=$(echo "$expected" | tr -d '[]' | tr ',' ' ')
    actual_dims=$(echo "$actual" | tr -d '[]' | tr ',' ' ')
    
    # Compare dimensions
    if [ ! -z "$expected_dims" ] && [ ! -z "$actual_dims" ]; then
        echo "1. Check if you need to reshape your tensor:"
        echo "   tensor = tensor.reshape($expected)"
        
        echo "2. Check if you need to transpose dimensions:"
        echo "   tensor = tensor.permute(...)"
        
        if [[ "$expected" == *"1"* ]] && [[ "$actual" != *"1"* ]]; then
            echo "3. You might need to unsqueeze a dimension:"
            echo "   tensor = tensor.unsqueeze(dim)"
        fi
        
        if [[ "$expected" != *"1"* ]] && [[ "$actual" == *"1"* ]]; then
            echo "4. You might need to squeeze a dimension:"
            echo "   tensor = tensor.squeeze(dim)"
        fi
    fi
    
    echo -e "\n5. Inspect tensor shapes before the operation:"
    echo "   print(f\"Shape before operation: {tensor.shape}\")"
}

if [ -z "$error_file" ]; then
    # Read from stdin if no file is provided
    echo "Paste your error log (Ctrl+D when finished):"
    log_content=$(cat)
    extract_tensor_info "$log_content"
else
    # Read from the provided file
    log_content=$(cat "$error_file")
    extract_tensor_info "$log_content"
fi

echo -e "\n=== DEBUGGING TIPS ==="
echo "1. Use tensor.shape to check dimensions at critical points"
echo "2. Consider using torch.Size() to explicitly set dimensions"
echo "3. For matrix operations, ensure matrix dimensions are compatible"
echo "4. Check batch dimensions in your data loader"
echo "5. Run with smaller batch size to simplify debugging"
