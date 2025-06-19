# Quantum Machine Learning for malware classification

In this research, we developed and evaluated Quantum Machine Learning (QML) models for malware classification, specifically focusing on the Quantum Multilayer Perceptron (QMLP) and the Quantum Convolutional Neural Network (QCNN).

## Datasets Used

For this research the datasets used were:

- API Graph
- AZ-Class-Task
- AZ-Domain
- Ember-Class-Task
- Ember Domain

## Reproducibility

### System requirements:

The following system requirements are necessary for the code to run:

- CUDA Version: 12.8
- Python: 3.10.18
- RAM: ≥32gb recommended

### Enviroment setup

Create and activate conda enviroment

```
conda create -f qmlenv.yml
conda activate qmlenv
```

### Running the code

The models used can be accessed in the `Models` folder. Make sure to have your conda enviroment activated before running the code. The code is designed in a way that it wil also compute the metrics once the training session is completed.

### Pretrained models

The pretrained models that we used for evaluations of this research are under the saved models folder on their respective results section.
