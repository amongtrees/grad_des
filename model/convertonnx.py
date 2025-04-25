import torch.onnx
from load_model import load_model
#Function to Convert to ONNX
def Convert_ONNX(model):

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 29, requires_grad=True).to('cuda')

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "is_non_model.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=14,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')

if __name__ == '__main__':
    model = load_model(r"E:\Graduate_Design\project\model\is_non_model.pth")
    Convert_ONNX(model)