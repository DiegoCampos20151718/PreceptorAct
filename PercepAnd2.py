import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def activation(self, x):
        return 1 if x >= 0 else -1
    
    def predict(self, x):
        x = np.insert(x, 0, 1)  # Insert bias term (θ)
        summation = np.dot(x, self.weights)
        return self.activation(summation)
    
    def train(self, X, y):
        for epoch in range(self.epochs):
            print(f"Época {epoch + 1}/{self.epochs}")
            for i in range(len(X)):
                x_with_bias = np.insert(X[i], 0, 1)  # Insert bias term (θ)
                summation = np.dot(x_with_bias, self.weights)
                prediction = self.activation(summation)
                error = y[i] - prediction
                self.weights += self.learning_rate * error * x_with_bias
                print(f"  Input: {X[i]}, Summation: {summation:.5f}, Prediction: {prediction}, Error: {error}, Weights: {self.weights}")
            print()
        print("Training complete")
        print(f"Weights after training: {self.weights}")

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_input_hidden = np.random.rand(input_size + 1, hidden_size)  # +1 for bias
        self.weights_hidden_output = np.random.rand(hidden_size + 1, output_size)  # +1 for bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def activation(self, x):
        return 1 if x >= 0.5 else -1

    def predict(self, x):
        if len(x.shape) == 1:  # Single sample
            x = np.insert(x, 0, 1)  # Insert bias term
            hidden_input = np.dot(x, self.weights_input_hidden)
            hidden_output = self.sigmoid(hidden_input)
            hidden_output = np.insert(hidden_output, 0, 1)  # Insert bias term
            final_input = np.dot(hidden_output, self.weights_hidden_output)
            final_output = self.sigmoid(final_input)
        else:  # Batch of samples
            final_output = []
            for sample in x:
                sample = np.insert(sample, 0, 1)  # Insert bias term
                hidden_input = np.dot(sample, self.weights_input_hidden)
                hidden_output = self.sigmoid(hidden_input)
                hidden_output = np.insert(hidden_output, 0, 1)  # Insert bias term
                final_input = np.dot(hidden_output, self.weights_hidden_output)
                final_output.append(self.sigmoid(final_input))
            final_output = np.array(final_output)
        return np.array([self.activation(o) for o in final_output])

    def train(self, X, y):
        y = (y + 1) / 2  # Convert -1 to 0 and 1 to 1
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x = np.insert(X[i], 0, 1)  # Insert bias term
                hidden_input = np.dot(x, self.weights_input_hidden)
                hidden_output = self.sigmoid(hidden_input)
                hidden_output = np.insert(hidden_output, 0, 1)  # Insert bias term
                final_input = np.dot(hidden_output, self.weights_hidden_output)
                final_output = self.sigmoid(final_input)

                output_error = y[i] - final_output
                output_delta = output_error * self.sigmoid_derivative(final_output)

                hidden_error = output_delta.dot(self.weights_hidden_output.T)
                hidden_delta = hidden_error[1:] * self.sigmoid_derivative(hidden_output[1:])

                self.weights_hidden_output += self.learning_rate * np.outer(hidden_output, output_delta)
                self.weights_input_hidden += self.learning_rate * np.outer(x, hidden_delta)

            if epoch % 1000 == 0:
                predictions = self.predict(X)
                predictions = (predictions + 1) / 2  # Convert -1 to 0 and 1 to 1 for loss calculation
                loss = np.mean(np.square(y - predictions))
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.5f}")

# Datos de entrada y salida para las compuertas lógicas con valores -1 y 1
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

# Compuerta AND
y_and = np.array([1, -1, -1, -1])
perceptron_and = Perceptron(input_size=2, learning_rate=0.4, epochs=10)
perceptron_and.train(X, y_and)
print("Predicciones AND:")
for i in range(len(X)):
    prediction = perceptron_and.predict(X[i])
    print(f"Input: {X[i]}, Predicción: {prediction}")

# Compuerta OR
y_or = np.array([1, 1, 1, -1])
perceptron_or = Perceptron(input_size=2, learning_rate=0.4, epochs=10)
perceptron_or.train(X, y_or)
print("Predicciones OR:")
for i in range(len(X)):
    prediction = perceptron_or.predict(X[i])
    print(f"Input: {X[i]}, Predicción: {prediction}")

# Compuerta XOR con una red neuronal
# Datos de entrada y salida para la compuerta XOR con valores -1 y 1
X_xor = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y_xor = np.array([[-1], [1], [1], [-1]])

# Crear y entrenar la red neuronal
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1, epochs=10000)
nn.train(X_xor, y_xor)

# Probar la red neuronal entrenada
print("Predicciones XOR:")
for i in range(len(X_xor)):
    prediction = nn.predict(X_xor[i])
    print(f"Input: {X_xor[i]}, Predicción: {prediction}")
