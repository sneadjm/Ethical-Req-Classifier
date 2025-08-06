from sentence_transformers import SentenceTransformer
import torch.nn as nn

class SBERTClassifier(nn.Module):
    def __init__(self, num_classes, embed_size=768):
        super(SBERTClassifier, self).__init__()
        self.embedder =  SentenceTransformer("all-mpnet-base-v2")
        self.head = nn.Linear(embed_size, num_classes)
        self.sigmoid = nn.Sigmoid()

        if num_classes == 1:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.head(x)
        x = self.sigmoid(x)
        return x
