import pickle
import torch

if __name__ == "__main__":
    filepath = "/home/OpenPCDet/debug/spconv.pkl"
    # with open(filepath, "r") as f:
    a = torch.load(filepath)
    print(a)
