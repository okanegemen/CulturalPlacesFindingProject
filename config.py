import torch


device = "mps" if torch.backends.mps.is_available() else "cpu"
deviceWin = "cuda:0" if torch.cuda.is_available() else "cpu"