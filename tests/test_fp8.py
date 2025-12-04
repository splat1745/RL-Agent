
import torch
import torch.nn as nn

def test_fp8():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float8_e4m3fn

    print("\n--- Test 1: Tensor Cast ---")
    try:
        t = torch.randn(10, 10, device=device, dtype=torch.float16)
        t_fp8 = t.to(dtype=dtype)
        print("Tensor cast to FP8 successful.")
        print(f"Tensor dtype: {t_fp8.dtype}")
    except Exception as e:
        print(f"Tensor cast failed: {e}")

    print("\n--- Test 2: Linear Layer Cast ---")
    try:
        lin = nn.Linear(32, 32).to(device=device, dtype=torch.float16)
        lin.to(dtype=dtype)
        print("Linear layer cast to FP8 successful.")
        print(f"Weight dtype: {lin.weight.dtype}")
    except Exception as e:
        print(f"Linear layer cast failed: {e}")

    print("\n--- Test 3: Linear Layer Forward ---")
    try:
        lin = nn.Linear(32, 32).to(device=device, dtype=torch.float16)
        lin.to(dtype=dtype)
        input_t = torch.randn(1, 32, device=device, dtype=dtype)
        out = lin(input_t)
        print("Linear layer forward pass successful.")
        print(f"Output dtype: {out.dtype}")
    except Exception as e:
        print(f"Linear layer forward failed: {e}")

if __name__ == "__main__":
    test_fp8()
