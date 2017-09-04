import numpy as np
import matplotlib.pyplot as plt

# input
x = np.array(np.arange(-10, 10, 1.0))
print(x)

# input nNW
X = np.array(np.arange(-1, 1, 1.0).reshape(1,2))
W1 = np.array([np.arange(1, 4, 1.0),[2,3,4]])
W2 = np.array(np.random.rand(3,2))
B1 = np.array([1,1,1])
B2 = np.array([2,2])
#print (X.shape)
#print (W1.shape)
#print (X)
#print (W1)
#print (W2)
#print (B1)
#print (B2)
              
y = x > 0
#print(y)
y = y.astype(np.int)
#print(y)
    
### step
def stepFunc(x):
    return np.array(x > 0, dtype=np.int)

### sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x) )

### softmax
def softmax(x):
    return 1 / (1 + np.exp(-x) )

### reLu
def reLu(x):
    return np.maximum(0,x)

### softMax
def softMax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp
    return y

### do
y1 = stepFunc(x)
y2 = sigmoid(x)
y3 = reLu(x)
y4 = softMax(x)
print ("softmax",y4)
print ("sum",np.sum(y4))

### neuralNW 2x3x2
A1 = np.dot(X , W1) + B1
Z1= sigmoid(A1)
#print("Z1",Z1)
A2 = np.dot(Z1 , W2) + B2
Z2= sigmoid(A2)
#print("Z2",Z2)
#print (Z2.shape)
### graph

plt.plot(x, y1,label="step")
#plt.plot(x, y2,label="sigmoid")
#plt.plot(x, y3,label="ReLU")
plt.plot(x,y4,label="softmax")
plt.xlabel("x")
plt.title("title")
plt.legend()
plt.ylim(-0.1, 1.1)

plt.show()


