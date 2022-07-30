from cmath import tan
import numpy as np
 
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
 
def sigmoid_derivada(x):
    return sigmoid(x)*(1.0-sigmoid(x))
 
def tanh(x):
    return np.tanh(x)
 
def tanh_derivada(x):
    return 1.0 - x**2
 
class NeuralNetwork:
 
    def __init__(self, layers, activation ='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivada
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivada
 
        self.weights = []
        self.deltas = []

        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
    
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)
 
    def fit(self, X, y, learning_rate=0.2, epochs=100000):
      
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
 
            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = tanh(dot_value)
                    a.append(activation)
    
            error = y[i] - a[-1]
            deltas = [error * sigmoid_derivada(a[-1])]
            
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*sigmoid_derivada(a[l]))
            self.deltas.append(deltas)
          
            deltas.reverse()
 
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
 
            if k % 10000 == 0: print('epochs:', k)
 
    def predict(self, x): 
        ones = np.atleast_2d(np.ones(x.shape[0]))
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = tanh(np.dot(a, self.weights[l]))
        return a
 
    def print_weights(self):
        print("LISTADO PESOS DE CONEXIONES")
        for i in range(len(self.weights)):
            print(self.weights[i])
 
    def get_deltas(self):
        return self.deltas

nn = NeuralNetwork([2, 3, 4,], activation = 'tahn')
X = np.array([
    [0 ,0], # SIN OBSTACULOS
    [0, 1], # SIN OBSTACULOS
    [0, -1], # SIN OBSTACULOS
    [0.5, 1], # OBSTACULO A LA DRCH
    [0.5, -1], # OBSTACULO A LA IZQ
    [1, 1], # OBSTACULO DEMASIADO CERCA A LA DRCH
    [1, -1]  # OBSTACULO DEMASIADO CERCA A LA IZQ
    ])          

y = np.array([
    [1,0,0,1], # AVANZAR
    [1,0,0,1], # AVANZAR
    [1,0,0,1], # AVANZAR
    [0,1,0,1], # GIRAR A LA DERECHA
    [1,0,1,0], # GIRAR A LA IZQUIERDA
    [1,0,0,1], # AVANZAR
    [0,1,1,0], # RETROCEDER
    [0,1,1,0], # RETROCEDER
    [0,1,1,0]  # RETROCEDER
])

nn.fit(X, y, learning_rate=0.03,epochs=40001)

def Predict(x):
    return (int(abs(round(x))))

index=0
for e in X:
    prediccion = nn.predict(e)
    print("X:",e,"esperado:",y[index],"Obtenido:", Predict(prediccion[0]),Predict(prediccion[1]),Predict(prediccion[2]),Predict(prediccion[3]))
    index=index+1
