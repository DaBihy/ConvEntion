import torch
import torch.nn as nn
from .conv_bert import SBERT

class ConvBERTClassification(nn.Module):
    """
    Downstream task:  Image Time Series Classification
    """

    def __init__(self, sbert: SBERT, num_classes):
        super().__init__()
        self.sbert = sbert
        self.classification = MulticlassClassification(self.sbert.hidden, num_classes)

    def forward(self, x, mask, band ):
        x = self.sbert(x, mask, band)
        return self.classification(x)


class MulticlassClassification(nn.Module):

    def __init__(self, hidden, num_classes):
        super().__init__()
        self.pooling = nn.MaxPool3d((1, 1, 33))
        self.linear1 = nn.Linear(hidden[1]*hidden[2]*hidden[3], 256)
        self.linearCon = nn.Linear(256, 128)

        self.linear3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.activation = nn.ReLU(inplace=True)
 

    def forward(self, x):
 
        x = self.pooling(x).squeeze()
        x = self.dropout(torch.flatten(x ,start_dim=1))
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.linear3(x)

        return x