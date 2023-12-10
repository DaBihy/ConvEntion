# ConvEntion For Astronomical Classification

This repository contains the implementation of ConvEntion, a transformer-based model, trained for the classification of astronomical objects. The model was primarily trained on data from the Sloan Digital Sky Survey (SDSS) and augmented with data from the ZTF (Zwicky Transient Facility) dataset. Our approach is detailed in the paper titled "Astronomical image time series classification using CONVolutional attENTION (ConvEntion)", available [here](https://arxiv.org/abs/2304.01236).

## Structure

The repository is structured as follows:

- `model/`: Contains the implementation of the ConvEntion model along with various attention mechanisms.
- `trainer/`: Includes the training logic and loss functions used for training the model.
- `dataset/`: Holds the dataset wrappers and specific implementations for processing the SDSS and ZTF datasets.
- `utils/`: Contains utility functions and layers used across the model.

## Dependencies

- PyTorch
- NumPy
- scikit-learn

## Configuration

The model training can be configured via the `config.yaml` file, where you can set various parameters like batch size, learning rate, and paths to the datasets.

## Usage

To train the ConvEntion model on your dataset, follow these steps:

1. Set up your Python environment with the necessary dependencies.
2. Update `config.yaml` with the appropriate paths and parameters for your training data.
3. Run the training script with `python training.py`.

## Scripts

- `training.py`: Main entry point for training the model.
- `trainer/trainer.py`: The trainer class responsible for the training loop and validation.
- `model/conv_bert.py`: ConvEntion model implementation.
- `dataset/train_dataset_ztf.py`: Dataset class implementation for loading and preprocessing the ZTF data.

## Model Training

The training process involves initializing the ConvEntion model, creating data loaders for the training and validation sets, and iteratively training the model on the SDSS and ZTF datasets. The `KFold` cross-validation is used to ensure the model's generalizability.

## Contributing

Contributions to this project are welcome. Please open an issue or a pull request with your suggested changes.


## Acknowledgments

- Thanks to the SDSS and ZTF collaborations for providing the datasets.
- This work utilizes resources from the Weights & Biases platform for tracking and visualizing the training runs.
- The model implementation in this repository is based on findings from our paper: "Astronomical image time series classification using CONVolutional attENTION (ConvEntion)". For details, please refer to our [paper](https://arxiv.org/abs/2304.01236).
