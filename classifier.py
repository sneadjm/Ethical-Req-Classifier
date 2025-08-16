from sentence_transformers import SentenceTransformer
import torch.nn as nn

class ReduceDimBlock(nn.Module):
    def __init__(self, hidden_size, embed_size=768):
        super(ReduceDimBlock, self).__init__()
        self.fc_red = nn.Linear(embed_size, hidden_size)
        self.bn_red = nn.BatchNorm1d(num_features=hidden_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, embeds):
        x = self.fc_red(embeds)
        x = self.bn_red(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class SBERTClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(SBERTClassifier, self).__init__()
        self.embedder =  SentenceTransformer("all-mpnet-base-v2")
        self.reducer = ReduceDimBlock(hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.head = nn.Sequential(self.reducer, self.classifier, self.sigmoid)

        if num_classes == 1:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def prep_for_tuning(self):
        for param in self.embedder.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def encode(self, x):
        embeds = self.embedder(x)
        return embeds

    def forward(self, embeds):
        x = self.head(embeds)
        return x
