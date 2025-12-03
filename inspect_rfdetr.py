
try:
    from rfdetr import RFDETRSmall
    import inspect
    
    print("Attributes of RFDETRSmall:")
    print(dir(RFDETRSmall))
    
    print("\nConstructor signature:")
    print(inspect.signature(RFDETRSmall.__init__))
    
    model = RFDETRSmall(num_classes=2)
    print("\nAttributes of instance:")
    print(dir(model))
    
    if hasattr(model, 'model'):
        print("\nmodel.model type:", type(model.model))
        
except ImportError:
    print("rfdetr not found")
except Exception as e:
    print(f"Error: {e}")
