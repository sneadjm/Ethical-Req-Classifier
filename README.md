# Ethical Requirements Classifier


### File Overview:
- train.py: The main script for loading a classification model and its dataset, and performing training/validation runs using them. See below for specifics on how to use.
- classifier.py: Contains the object that represents the classification model and explicitly lists out layers as part of a subclass of nn.Module.
- weights: Directory where weights to use can be uploaded and where weights will be saved following training.
- plots: Directory where plots displaying the ROC curves following a validation run of a model will be saved.
- dataset.py: Helpful processing script that reads the data as it appears in a CSV and converts it to PyTorch-based dataloaders. This will need to be reconfigured if using a different training dataset.


#### train.py

How to run:
- Run a `pip install -r requirements.txt` in a virtual environment to ensure all dependencies are met
- Edit the file paths in data_config/data.yaml to match 
- Specify whether this is a tuning of a pre-trained model (`pretrained=True`). If so, add the path to the pretrained weights.
- If only validating a model, set `do_train=False`. Be sure to update the weights in `wt_list` with the models you want evaluated. If you want to train and validate, set `do_train=True`, and it will validate the models listed following the training. 
- Provide a number of epochs for the model to undergo with `n_epochs`. The training logic will train until it hits this number or if there are `n_patience` number of epochs without improvement. The early stopping criteria is defaulted to 10, but can be overridden as a parameter in the `train()` function.
- Add a path for the newly trained weights to be added to. 
- The `val()` function will do an overall validation of the model following training (notably, not the same process of validation that occurs during training) and will produce ROC curve plots and print out basic performance metrics. If this is not desired, simply comment out the last line that calls `val()`.
- Run the script by either using the run functionality in an IDE of your choice or by `python train.py` in the command line with the virtual environment running.

This project represents the code forming the basis of a graduate research project into ethical requirements identification led by Dr. Krishnendu Ghosh.

### References
This repository uses datasets from:
- PURE Dataset: A. Ferrari, G. O. Spagnolo, and S. Gnesi, “Pure: A dataset of public requirements documents,” in 2017 IEEE 25th International Requirements
Engineering Conference (RE). IEEE, 2017, pp. 502–505.
- ETHICS Benchmark: D. Hendrycks, C. Burns, S. Basart, A. Critch, J. Li, D. Song, and
J. Steinhardt, “Aligning AI with shared human values,” Proceedings of
the International Conference on Learning Representations (ICLR), 2021.
- E.U. Guidelines for Trustworthy AI: https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai
- IEEE Code of Ethics: https://www.ieee.org/content/dam/ieee-org/ieee/web/org/about/corporate/ieee-code-of-ethics.pdf

The model card from the all-mpnet-v2 mdoel used to generate the embeddings can be found here in the HuggingFace library: https://huggingface.co/sentence-transformers/all-mpnet-base-v2