import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU information:")
    print(torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Make sure your GPU and drivers are correctly installed.")
