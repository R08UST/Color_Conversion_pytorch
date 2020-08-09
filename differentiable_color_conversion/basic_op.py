import torch
import numpy as np

#rgb, lab, xyz, ycbcr https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
# srgb https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz  
# http://www.brucelindbloom.com/index.html?Math.html  

# all input tensor should be B, W, H, C


def rgb2xyz(var, device = 'cuda'):
    #input (min, max) = (0, 1)
    #output (min, max) = (0, 1)
    transform = torch.FloatTensor([[0.412453, 0.357580, 0.180423], 
                            [0.212671, 0.715160, 0.072169], 
                            [ 0.019334,  0.119193,  0.950227]]).to(device)
    xyz = torch.matmul(var, transform.t())
    return xyz

def xyz2lab(pix):
    #input (min, max) = (0, 1)
    #output (min, max) = (-100, 100)
    x = pix[:, :, :, 0]/0.950456 # x = xr/whitepoint
    y = pix[:, :, :, 1]  #y's whitepoint  = 1
    z = pix[:, :, :, 2]/1.088754 # z = zr/whitepoint
    fx = torch.where(x > 0.008856, torch.pow(x, 1/3), 7.787*x + (16/116))
    fy = torch.where(y > 0.008856, torch.pow(y, 1/3), 7.787*y + (16/116))
    fz = torch.where(z > 0.008856, torch.pow(z, 1/3), 7.787*z + (16/116))
    l = 116*fy - 16
    
    a = 500*(fx - fy)#+128
    b = 200*(fy - fz)#+128
    lab = torch.stack([l, a, b], 3)
    return lab



def lab2xyz(imgs):
    #input (min , max) = (0, 100)
    #output (min, max) = (0, 1)
    l = (imgs[:, :, :, 0])
    a = imgs[:, :, :, 1]
    b = imgs[:, :, :, 2]
    fy = (l + 16)/116
    fx = a/500 + fy
    fz = fy - b/200
    xr = torch.where(torch.pow(fx, 3)>0.008856, torch.pow(fx, 3), (116*fx-16)/903.3)
    yr = torch.where(torch.pow(fy, 3)>0.008856, torch.pow(fy, 3), (116*fy-16)/903.3)
    zr = torch.where(torch.pow(fz, 3)>0.008856, torch.pow(fz, 3), (116*fz-16)/903.3)
    x = 0.950456*xr # x = xr*whitepoint
    y = yr #y's whitepoint  = 1 
    z = 1.088754*zr # z = zr*whitepoint
    xyz = torch.stack([x, y, z], 3)
    return xyz

def xyz2rgb(imgs, device = 'cuda'):
    #input (min , max) = (0, 1)
    #output (min, max) = (0, 1)
    transform = torch.FloatTensor([[ 3.240479, -1.53715, -0.498535], 
                            [-0.969256, 1.875991, 0.041556], 
                            [ 0.055648, -0.204043, 1.057311]]).to(device)
    rgb = torch.matmul(imgs, transform.t())
    return rgb

def ycrcb2rgb(imgs):
    #input (min, max) = (16, 240)  Y∈[16,235] cr∈[16,240] cb∈[ 16,240 ]
    #output (min, max) = (0, 255)
    y = imgs[:, :, :, 0]
    cr = imgs[:, :, :, 1]
    cb = imgs[:, :, :, 2]
    r = y+1.403*(cr - 128)
    g = y-0.714*(cr - 128) - 0.344*(cb - 128)
    b = y+1.773*(cb - 128)
    rgb = torch.stack([r, g, b], 3)
    return rgb

def rgb2ycrcb(imgs):
    #input (min, max) = (0, 255)
    #output (min, max) = (16, 240)
    r = imgs[:, :, :, 0]
    g = imgs[:, :, :, 1]
    b = imgs[:, :, :, 2]
    y = 0.299*r + 0.587*g + 0.114*b
    cr = (r - y)*0.713 + 128
    cb = (b - y)*0.564 + 128
    ycrcb = torch.stack([y, cb, cr], 3)
    return ycrcb    

def rgb2srgb(imgs):

    return torch.where(imgs <= 0.04045, imgs/12.92, torch.pow((imgs + 0.055)/1.055, 2.4))

def srgb2rgb(imgs):
    return torch.where(imgs <= 0.0031308, imgs*12.92, 1.055*torch.pow((imgs), 1/2.4) - 0.055)

def rgb2gray(imgs, device = "cuda"):
    #input (min, max) = (0, 255)
    #output (min, max) = (0, 255)
    transform = torch.FloatTensor([0.299], [0.587], [0.114]).to(device)
    gray = torch.matmul(imgs, transform)
    return gray


def rgb2hsv(imgs, device = "cuda"):
    #input (min, max) = (0, 255)
    #output (min, max) = (0, 360)
    r = imgs[:, :, :, 0]
    g = imgs[:, :, :, 1]
    b = imgs[:, :, :, 2]
    v = torch.max(imgs)
    inds = torch.argmax(torch.max(torch.max(imgs, dim = 1).values, dim = 1).values, dim = 1) # from bhwc 2 bwc 2 bc
    s = torch.where(v == 0, 0, torch.div(v - torch.min(imgs), v))
    h = torch.div(60*(g - b), v - torch.min(imgs))
    inds = inds*120 
    h = torch.add(h, inds)
    '''
    for ind in inds:
        if ind == 0:
            h_ = torch.div(60*(g - b), v - torch.min(imgs)) # if v == r 
        if ind == 1:
            h_ = 120 + torch.div(60*(g - b), v - torch.min(imgs)) # if v == g 
        if ind == 2:
            h_ = 240 +torch.div(60*(g - b), v - torch.min(imgs)) # if v == b
    '''
    hsv = torch.stack([h, s, v], 3)
    return hsv

def hsv2rgb(imgs, device = "cuda"):
    #ref: https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    #input (min, max) = (0, 360)
    #output (min, max) = (0, 255)
    h = imgs[:, :, :, 0]
    assert (h<0).any(), "h is less than 0"
    s = imgs[:, :, :, 1]
    v = imgs[:, :, :, 2]
    c = torch.matmul(v, s)
    x = torch.matmul(c, (1 - torch.abs((
                        (h/60)%2) - 1)))
    m = v - c

    def interval_index(arr):
        ind60 = h[torch.where(h<60)] = 0
        ind120 = h[torch.where(h >= 60 & h < 120)] = 1
        ind180 = h[torch.where(h >= 120 & h < 180)] = 2
        ind240 = h[torch.where(h >= 180 & h < 240)] = 3
        ind300 = h[torch.where(h >= 240 & h < 300)] = 4
        ind360 = h[torch.where(h >= 300 & h < 360)] = 5
        return arr
    h = interval_index(h)
    rgb = torch.stack([c, x, 0], 3) + m #waiting for permute
    order = np.array([[0, 3, 1], [2, 0, 1], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]])
    rgb = rgb[:, :, :, order[h]]
    return rgb


