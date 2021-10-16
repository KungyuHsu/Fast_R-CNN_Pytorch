import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
torch.set_num_threads(8)