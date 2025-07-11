import torch

n_adj_slc=5
n_coil=10
rand_input = torch.randn(1, n_adj_slc*n_coil,5, 5, 2)
rand_mask = torch.randn(1, n_adj_slc, 5, 5, 1).bool()
rand_a = torch.where(rand_mask, rand_input, zero)
print(rand_a)
