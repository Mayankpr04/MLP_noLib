import random

def sigmoid(x):
    e = 2.718281828459045
    return 1 / (1 + e**-x) # Sigmoid function implementation

def sigmoid_derivative(x):
    return x * (1 - x) # Derivative of the sigmoid function

def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h # Derivative (to test sigmoid_derivative)

def euclidean_loss(y_true, y_pred):
    loss = (y_true - y_pred) * (y_true - y_pred)
    return loss # Euclidean loss implementation

def transpose(matrix):
    matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return matrix # Transpose of a matrix

def matrix_multiplication(A,B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    if cols_A != rows_B:
        raise ValueError("Cannot multiply the two matrices. Incorrect dimensions.")
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result # Matrix multiplication

seed = 1234567
def random_implementation():
    global seed
    seed = (1103515245 * seed + 12345) % (2**31)
    return (seed / (2**31)) # Custom random number generator (well, pseudo)

class MultiLayerPerceptron:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.number_of_layers = len(layer_sizes)-1
        self.hidden_layers = self.number_of_layers - 1
        self.weights = []
        self.biases = []
        for i in range(self.number_of_layers):
            #weight_matrix = [[2*random_implementation()-1 for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])]
            #bias_vector = [2*random_implementation()-1 for _ in range(layer_sizes[i+1])]
            weight_matrix = [[random.uniform(-1, 1) for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])]
            bias_vector = [random.uniform(-1, 1) for _ in range(layer_sizes[i+1])]
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def forward_propagation(self, input):
        X = [[x] for x in input]
        self.Weighted_X = []
        self.A = [X]
        for i in range(self.number_of_layers):
            weight_matrix = self.weights[i]
            bias_vector = self.biases[i]
            Weighted_X = matrix_multiplication(weight_matrix, X)
            for j in range(len(Weighted_X)):
                Weighted_X[j][0] += bias_vector[j]
            self.Weighted_X.append(Weighted_X) # Store Weighted_X for backpropagation
            A =[[sigmoid(Weighted_X[0])]for Weighted_X in Weighted_X] 
            self.A.append(A) # Store activations for backpropagation
            X = A
        return self.A[-1]
    
    def back_propagation(self, y_true):
        y_true = [[y] for y in y_true]
        deltas = [None]*self.number_of_layers
        gradients_w = [None]*self.number_of_layers
        gradients_b = [None]*self.number_of_layers
        output_layer = self.number_of_layers - 1
        deltas[output_layer] = []
        for o in range(len(self.A[-1])):
            loss = euclidean_loss(y_true[o][0], self.A[-1][o][0])
            error = 2 * (self.A[-1][o][0] - y_true[o][0]) # Derivative of Euclidean loss
            delta = error * sigmoid_derivative(self.A[-1][o][0])
            deltas[output_layer].append([delta]) # o for output layer
        for l in reversed(range(output_layer)):
            W_next_T = transpose(self.weights[l+1])
            delta_next = deltas[l+1]
            delta_raw = matrix_multiplication(W_next_T, delta_next)
            deltas[l] = []
            for i in range(len(delta_raw)):
                delta = delta_raw[i][0] * sigmoid_derivative(self.A[l+1][i][0])
                deltas[l].append([delta])
        for l in range(self.number_of_layers):
            A_prev_T = transpose(self.A[l])
            gradients_w[l] = matrix_multiplication(deltas[l], A_prev_T)
            gradients_b[l] = [d[0] for d in deltas[l]]
        return gradients_w, gradients_b # Return gradients for weights and biases
    
    def update_weights(self, gradients_w, gradients_b):
        for l in range(self.number_of_layers):
            for i in range(len(self.weights[l])):
                for j in range(len(self.weights[l][0])):
                    self.weights[l][i][j] -= self.learning_rate * gradients_w[l][i][j]
            for i in range(len(self.biases[l])):
                self.biases[l][i] -= self.learning_rate * gradients_b[l][i]
    
    def train(self, X_train, Y_train, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X_train, Y_train):
                y_pred = self.forward_propagation(x)
                for o in range(len(y)):
                    total_loss += euclidean_loss(y[o], y_pred[o][0])
                gradients_w, gradients_b = self.back_propagation(y)
                self.update_weights(gradients_w, gradients_b)
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss = {total_loss/len(Y_train)}")

def load_csv(filename, input_dimension, output_dimension):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()[1:] # Skip header - contains names
        for line in lines:
            row = [float(x) for x in line.strip().split(',')] 
            data.append(row)
    X = [row[:input_dimension] for row in data]
    Y = [row[input_dimension:input_dimension+output_dimension] for row in data]
    return X, Y

def train_test_split(X, Y, split_ratio=0.8):
    split_index = int(len(X) * split_ratio)
    X_train = X[:split_index]
    Y_train = Y[:split_index]
    X_test = X[split_index:]
    Y_test = Y[split_index:]
    return X_train, Y_train, X_test, Y_test

def compute_accuracy(X, Y, mlp):
    correct = 0
    total = len(X)
    for x, y_true in zip(X, Y):
        y_pred = mlp.forward_propagation(x) 
        y_pred_binary = [round(p[0]) for p in y_pred] # round predictions to 0 or 1
        if y_pred_binary == y_true:
            correct += 1
    return correct / total

def test_random_case(X, Y, mlp):
    idx = int(random_implementation() * len(X))
    x = X[idx]
    y_true = Y[idx]
    y_pred = mlp.forward_propagation(x)
    y_pred_binary = [round(p[0]) for p in y_pred]
    print("--- Random Test Case ---")
    print(f"Input: {x}")
    print(f"True Output: {y_true}")
    print(f"Model Prediction: {[p[0] for p in y_pred]}")
    print(f"Model Rounded Prediction: {y_pred_binary}")

def mlp_run(dataset, input_dimension, output_dimension, layer_sizes, learning_rate=0.1, epochs=100, split_ratio=0.8):
    print("\n------ MLP Run ------")
    print(f"Dataset: {dataset}, Input Dim: {input_dimension}, Output Dim: {output_dimension}")
    print(f"Layer Sizes: {layer_sizes}, Learning Rate: {learning_rate}, Epochs: {epochs}, split_ratio: {split_ratio}")
    X, Y = load_csv(dataset, input_dimension, output_dimension)
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, split_ratio=0.8)
    mlp = MultiLayerPerceptron(layer_sizes, learning_rate)
    mlp.train(X_train, Y_train, epochs)
    total_loss = 0
    for x, y in zip(X_test, Y_test):
        y_pred = mlp.forward_propagation(x)
        for o in range(len(y)):
            total_loss += euclidean_loss(y[o], y_pred[o][0])
    average_loss = total_loss / len(Y_test)
    accuracy = compute_accuracy(X_test, Y_test, mlp)
    print(f"Test Loss: {average_loss}, Test Accuracy: {accuracy*100:.2f}%\n")
    test_random_case(X_test, Y_test, mlp)
    return mlp

if __name__ == '__main__':
    print("\n------- XOR Dataset -------")
    mlp_run("xor_dataset.csv", input_dimension=2, output_dimension=1,
        layer_sizes=[2,4,1], learning_rate=0.3, epochs=200, split_ratio=0.2)

    mlp_run("xor_dataset.csv", input_dimension=2, output_dimension=1,
            layer_sizes=[2,3,2,1], learning_rate=0.1, epochs=50, split_ratio=0.7)

    mlp_run("xor_dataset.csv", input_dimension=2, output_dimension=1,
            layer_sizes=[2,4,4,1], learning_rate=0.05, epochs=300, split_ratio=0.5)
    
    mlp_run("xor_dataset.csv", input_dimension=2, output_dimension=1,
            layer_sizes=[2,2,1], learning_rate=0.05, epochs=500, split_ratio=0.8)

    print("\n----- Adder dataset experiments ------")
    mlp_run("adder_dataset.csv", input_dimension=5, output_dimension=3,
            layer_sizes=[5,6,3], learning_rate=0.5, epochs=200, split_ratio=0.3)

    mlp_run("adder_dataset.csv", input_dimension=5, output_dimension=3,
            layer_sizes=[5,3,5,3], learning_rate=0.2, epochs=400, split_ratio=0.6)

    mlp_run("adder_dataset.csv", input_dimension=5, output_dimension=3,
            layer_sizes=[5,8,4,3], learning_rate=0.1, epochs=300, split_ratio=0.8)
    
    mlp_run("adder_dataset.csv", input_dimension=5, output_dimension=3,
            layer_sizes=[5,5,4,3], learning_rate=0.05, epochs=500, split_ratio=0.7)
