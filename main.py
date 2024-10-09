import torch

if __name__ == "__main__":
    tensor = torch.tensor([1, 2, 3])
    print(tensor)
    print(tensor.size())

    # test CUDA
    if torch.cuda.is_available():
        print("CUDA is available")
        tensor = tensor.to("cuda")
    else:
        print("CUDA is not available")
    
    print(tensor.device)