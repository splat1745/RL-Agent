
try:
    from rfdetr import RFDETRSmall
    import torch.nn as nn
    
    model = RFDETRSmall(num_classes=2)
    
    print("\nIs model.model.model an nn.Module?", isinstance(model.model.model, nn.Module))
    
except Exception as e:
    print(f"Error: {e}")
