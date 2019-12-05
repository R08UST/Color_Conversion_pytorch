# Differentiable Color Conversion for Pytorch

## Requirement
1. pytorch
2. numpy 

## Usage
1. import color.py
2. embededing the conversion class in to you code. 


    class ColorModel(nn.Module):
        '''
        input should be B, C, H, W and the value should between [0, 1]
        output is B, C, H, W and the RGB is between [0, 1]
        '''


        def __init__(self):
            super(ColorModel, self).__init__()
            self.A = Autoencoder()
            self.rgb = lab2rgb()
            self.lab = rgb2lab()
            

        def forward(self,x):# input rgb [0,1]

            lab_x = self.lab(x)
            a = self.A(lab_x)

## TODO
1. often-used color space conversion 
2. usage demo
3. Chainer Version
4. Tensorflow Version
