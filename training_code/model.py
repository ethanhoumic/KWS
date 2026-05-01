import torch
import torch.nn as nn

# ====================================================================================
#                               model structure                      
# ====================================================================================

    
class CNNTradFpool3(nn.Module):
    """
    CNN architecture: cnn-trad-fpool3
    Reference: Sainath & Parada, "CNNs for Small-footprint KWS", Interspeech 2015
    Table 1 architecture (adapted for Google Speech Commands):

    Input: (B, 1, 101, 40)  -> (batch, channel, time, freq)
    """
    def __init__(self, num_classes=12, dropout_rate = 0.3):
        super(CNNTradFpool3, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(67, 8),
            stride=(1, 1),
            padding=0,
            bias=True
        )
        
        self.sigmoid1 = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(
            in_channels=64, 
            out_channels= 64, 
            kernel_size=(10, 4),
            stride=(1, 1),
            padding=0,
            bias=True
        )
        
        self.sigmoid2 = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self._conv_out_dim = self._get_conv_out_dim()
        
        self.linear = nn.Linear(self._conv_out_dim, 32, bias=True)
        
        self.dnn = nn.Linear(32, 128, bias=True)
        self.sigmoid3 = nn.Sigmoid()
        
        self.classifier = nn.Linear(128, num_classes, bias=True)
        
        self._initialize_weight()
    
    def _get_conv_out_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 101, 40)
            conv1 = self.conv1(dummy)
            sigmoid1 = self.sigmoid1(conv1)
            conv2 = self.conv2(sigmoid1)
            sigmoid2 = self.sigmoid2(conv2)
            return sigmoid2.view(1, -1).shape[1]
    
    def _initialize_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        if x.dim() == 3 :
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.sigmoid1(x)
        
        x = self.conv2(x)
        x = self.sigmoid2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.linear(x)
        x = self.dropout(x)
        x = self.dnn(x)
        x = self.sigmoid3(x)
        x = self.dropout(x)
        
        logits = self.classifier(x)
        
        return logits