import torch
import differentiable_color_conversion.color as color
import PIL.Image as Im

img = Im.open('plane.jpg').convert('rgb')
img = torch.Tensor(img)