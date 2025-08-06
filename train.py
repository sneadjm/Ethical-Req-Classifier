import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import os

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, precision_score, recall_score

from dataset import load_training_dataset
from classifier import SBERTClassifier


def train(n_epochs, model, save_pth, data_pth='data_config/data.yaml', n_patience=10, batch_size=32):
    valid_losses = []
    train_losses = []
    # making initial loss infinite to guarantee first epoch improvement
    valid_loss_min = np.inf

    criterion = model.loss

    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.001)

    global final_val_dl
    train_dl, val_dl, final_val_dl = load_training_dataset(data_pth=data_pth, batch_size=batch_size, embedder=model.embedder)

    n_no_improve = 0
    for epoch in range(n_epochs):
        # monitor training loss
        train_loss, val_loss = 0.0, 0.0

        model.train()  # prep model for training

        for data, target in train_dl:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target.float().unsqueeze(1))
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()

        model.eval()  # prep model for evaluation
        for data, target in val_dl:
            output = model(data)
            loss = criterion(output, target.float().unsqueeze(1))
            # update running validation loss
            val_loss += loss.item()

        # calculate average loss over an epoch
        train_loss = train_loss / len(train_dl)
        val_loss = val_loss / len(val_dl)
        valid_losses.append(val_loss)
        train_losses.append(train_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            val_loss
        ))

        # will only save model if val_loss has decreased to save storage and prevent overtraining
        if val_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                val_loss))
            torch.save(model.head.state_dict(), save_pth)
            valid_loss_min = val_loss
            n_no_improve = 0
        else:
            n_no_improve +=1
            if n_no_improve == n_patience:
                print(f'Processed {n_patience} epochs without improvement. Implementing early stopping')
                break

def val(model, wts_list):
    for wts in wts_list:
        model.head.load_state_dict(torch.load(os.path.join('weights', wts)), strict=True)
        model.eval()

        for data, target in final_val_dl:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # _, predicted = torch.max(output, 1)
            predicted = torch.max(output, 1).values

            # Update the running total of correct predictions and samples
            total_correct = (predicted.round() == target).sum().item()
            total_samples = target.size(0)
            # output = model(data).to(torch.float32)
            # target = target.to(torch.float32)
            # print(target, output)
            # loss = criterion(output, target)
            accuracy = total_correct / total_samples
            precision = precision_score(target.detach().numpy(), predicted.detach().numpy())
            recall = recall_score(target.detach().numpy(), predicted.detach().numpy())
            print()
            print('-'*50)
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            RocCurveDisplay.from_predictions(target.detach().numpy(), predicted.detach().numpy())
            plt.title('Small Commonsense Model Tested on Justice Dataset')
            plt.show()

if __name__ == "__main__":
    model = SBERTClassifier(num_classes=1)
    pre_trained = False
    if pre_trained:
        model.load_state_dict(torch.load('weights/cm.pt'))

    n_epochs = 100
    save_pth = 'weights/just_and_cm.pt'
    train(n_epochs=n_epochs, model=model, save_pth=save_pth)

    wts_list = ['just.pt', 'cm.pt', 'just_and_cm.pt']
    val(model, wts_list=wts_list)