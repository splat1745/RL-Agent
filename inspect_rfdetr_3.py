
try:
    from rfdetr import RFDETRSmall
    import inspect
    
    model = RFDETRSmall(num_classes=2)
    
    print("\nAttributes of model.model:")
    print(dir(model.model))
    
    print("\nHelp for optimize_for_inference:")
    print(help(model.optimize_for_inference))
    
except Exception as e:
    print(f"Error: {e}")
