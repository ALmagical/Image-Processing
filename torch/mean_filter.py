import torch


def mean_filter(input:torch.Tensor, ksize, padding=True):
    # The dimension of input are 4
    assert input.dim() == 4

    # B : batch numbers
    # C : channels
    # H, W : the height and width of input
    B, C, H, W = input.shape
        
    k_h, k_w = ksize
    # Split the input
    assert k_h == k_w
    if padding:
        pad = (k_h - 1) // 2
        input = torch.nn.functional.pad(input, [pad,pad,pad,pad], mode='constant', value=0)
        out_h = H
        out_w = W
    
    patches = input.unfold(dimension=2, size=k_h, step=1)
    patches = patches.unfold(dimension=3, size=k_h, step=1)

    # Mean
    out = patches.reshape(B, C, out_h, out_w, -1).mean(dim=-1)

    return out