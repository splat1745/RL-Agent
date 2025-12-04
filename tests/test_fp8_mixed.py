
import torch
import torch.nn as nn

def test_fp8_mixed_input():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_fp8 = torch.float8_e4m3fn
    dtype_fp16 = torch.float16

    print(f"Device: {device}")

    try:
        # Create Linear in FP8
        lin = nn.Linear(32, 32).to(device=device, dtype=dtype_fp16)
        lin.to(dtype=dtype_fp8)
        print("Linear layer converted to FP8.")

        # Create Input in FP16
        input_fp16 = torch.randn(1, 32, device=device, dtype=dtype_fp16)
        print("Input tensor created in FP16.")

        # Try Forward
        print("Attempting forward pass with FP16 input -> FP8 Linear...")
        out = lin(input_fp16)
        print("Forward pass successful!")
        print(f"Output dtype: {out.dtype}")

    except Exception as e:
        print(f"Forward pass failed: {e}")

    print("\n--- Test 2: Explicit Cast Input ---")
    try:
        input_fp8 = input_fp16.to(dtype=dtype_fp8)
        print("Input cast to FP8.")
        out = lin(input_fp8)
        print("Forward pass with FP8 input successful!")
        print(f"Output dtype: {out.dtype}")
    except Exception as e:
        print(f"Forward pass with FP8 input failed: {e}")

if __name__ == "__main__":
    test_fp8_mixed_input()
