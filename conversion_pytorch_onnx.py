import torch


weights_path = 'trained_model/weights/best.pt'
model = torch.hub.load('prunedyolov5', 'custom', path=weights_path, source='local')


model.eval()

dummy_input = torch.randn(1, 3, 160, 160)


torch.onnx.export(
    model,                              # Model to export
    dummy_input,                        # Dummy input tensor
    "best.onnx",                        # Name of the output ONNX file
    opset_version=12,                   # ONNX opset version (make sure it matches your needs)
    input_names=['images'],             # Name of the input tensor(s)
    output_names=['output'],            # Name of the output tensor(s)
    dynamic_axes={                      # Dynamic axes for variable-length inputs
        'images': {0: 'batch_size'},    # Allow variable batch size
        'output': {0: 'batch_size'}
    }
)
print("Model has been converted to ONNX format.")
