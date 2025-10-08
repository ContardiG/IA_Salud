import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG3DViewSelector(nn.Module):
    def __init__(self, num_classes=5, example_input=(10,1,16,128,128)):
        
        # example_input : [batch_size,channel,frame,height,width]
        
        super(VGG3DViewSelector, self).__init__()
        
        self.ch_in = example_input[1]
        
        # Bloque convolucional
        self.block = nn.Sequential(nn.Conv3d(self.ch_in, 64, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(64, 64, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=2, stride=2),
                                   
                                   nn.Conv3d(64, 128, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(128, 128, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=2, stride=2),
                                   
                                   nn.Conv3d(128, 256, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(256, 256, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=2, stride=2))
        
        # Cálculo de las dimensiones de salida
        with torch.no_grad():
            dummy = torch.zeros(example_input)  # [batch, C, T, H, W]
            out = self.conv_block(dummy)
            flatten_dim = out.view(out.size(0), -1).shape[1]

        # Clasificador
        self.classifier = nn.Sequential(nn.Linear(flatten_dim, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(512, num_classes),
                                        nn.Softmax(dim=1))

    def forward(self, x):
        # x: (batch, 1, frames, H, W)
        x = nn.conv_block(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x