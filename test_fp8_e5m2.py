
import torch
import torch.nn as nn

def test_fp8_e5m2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float8_e5m2

    print(f"Testing {dtype} on {device}")

    try:
        lin = nn.Linear(32, 32).to(device=device, dtype=torch.float16)
        lin.to(dtype=dtype)
        
        input_t = torch.randn(1, 32, device=device, dtype=torch.float16).to(dtype=dtype)
        
        out = lin(input_t)
        print("Forward pass successful!")
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test_fp8_e5m2()
