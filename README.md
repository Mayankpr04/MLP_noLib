# Custom MLP implementation
This project is a from-scratch implementation of a Multi-Layer Perceptron (MLP) in Python, built without using any machine learning libraries.
It is designed to predict the behavior of a 2-input XOR gate and a 2-bit binary adder.

The implementation includes the full training pipeline: forward propagation, backpropagation, and gradient descent.

## Getting started 
Clone the repository 
```
git clone https://github.com/Mayankpr04/MLP_noLib.git
cd MLP_noLib
```
## Datasets
This project includes two datasets: xor_dataset.csv and adder_dataset.csv
The datasets are generated with noise, which encourages generalization and helps prevent overfitting to perfectly clean truth tables.

## Implements
The neural network is implemented entirely from scratch, including
* Sigmoid activation function
* Euclidean loss
* Backpropagation with gradient descent
* Matrix multiplication
* Weight and bias updates
* Gradient computation

## Customization
You can experiment with arbitrary network depths and layer sizes to observe how architecture choices affect:
- Convergence speed
- Final accuracy
- Stability during training

Several architectures and hyperparameter settings are already tested in main.py

## How to run
Navigate to the project directory 
```
cd ~/MLP_noLib
python3 main.py
```
You can also setup the run script by doing:
```
chmod +x run
./run
```




