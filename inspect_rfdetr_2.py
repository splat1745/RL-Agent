
try:
    from rfdetr import RFDETRSmall
    import torch.nn as nn
    import inspect
    
    model = RFDETRSmall(num_classes=2)
    
    print("\nIs model.model an nn.Module?", isinstance(model.model, nn.Module))
    
    if hasattr(model.model, 'to'):
        print("model.model has 'to' method")
        
    print("\nPredict signature:")
    print(inspect.signature(model.predict))
    
except Exception as e:
    print(f"Error: {e}")
