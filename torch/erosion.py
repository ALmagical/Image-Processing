import torch
import cv2


def erosion(input:torch.Tensor, ksize, kernel=None , padding=True):
    # The dimension of input are 4
    assert input.dim() == 4

    # B : batch numbers
    # C : channels
    # H, W : the height and width of input
    B, C, H, W = input.shape
    
    # Get filter kernel
    if kernel == None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize)
    
    kernel = torch.tensor(kernel, dtype=torch.bool)
    k_h, k_w = kernel.shape

    # Split the input
    assert k_h == k_w
    if padding:
        pad = (k_h - 1) // 2
        input = torch.nn.functional.pad(input, [pad,pad,pad,pad], mode='constant', value=0)
        out_h = H
        out_w = W
    
    patches = input.unfold(dimension=2, size=k_h, step=1)
    patches = patches.unfold(dimension=3, size=k_h, step=1)

    # Erosion
    out, _ = patches[:,:,:,:,kernel].reshape(B, C, out_h, out_w, -1).min(dim=-1)

    return out