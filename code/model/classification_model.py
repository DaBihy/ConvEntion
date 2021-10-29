import torch
import torch.nn as nn
from .bert import SBERT

class SBERTClassification(nn.Module):
    """
    Downstream task:  Time Series Classification
    """

    def __init__(self, sbert: SBERT, num_classes):
        super().__init__()
        self.sbert = sbert
        self.classification = MulticlassClassification(self.sbert.hidden, num_classes)

    def forward(self, x, mask, band, ebv):
        x = self.sbert(x, mask, band)
        return self.classification(x,ebv)


class MulticlassClassification(nn.Module):

    def __init__(self, hidden, num_classes):
        super().__init__()
        self.pooling = nn.MaxPool3d((1, 1, 33))
        self.linear1 = nn.Linear(hidden[1]*hidden[2]*hidden[3], 256)
        #self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.ReLU(inplace=True)
 

    def forward(self, x, ebv):
 
        x = self.pooling(x).squeeze()
        x = self.dropout(torch.flatten(x ,start_dim=1))
        # ebv = torch.unsqueeze(ebv, 1)
        # print(f'ebv shape is {ebv.shape}')
        # print(f'x shape is {x.shape}')
        # x= torch.cat((x,ebv),1)
        #x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.dropout(self.activation(self.linear1(x)))
        #x = self.linear3(self.activation(x))
        x = self.linear3(x)

        return x