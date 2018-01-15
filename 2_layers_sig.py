import pandas as pd
import numpy as np

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    ### YOUR CODE HERE
    if x.ndim == 1:
        maxi= np.amax(x)
        sub = maxi * np.ones(x.size)
        x = x-sub
        x = np.exp(x)
        tot = np.sum(x)
        x = x / tot
    
    
    else:
        maxi= np.amax(x, axis=1)
        
        x = x - maxi[:,None]
        x = np.exp(x)
        tot = np.sum(x, axis = 1)
        x = x / tot[:,None]
	
    
    return x



def sigmoid(x):
    x = 1 /(1 + np.exp(-x))
    return x

def sigmoid_grad(f):
    
    f = f - (f ** 2)
    return f


def cost_grad(W1, W2, b1, b2, X_train, y):
    (m, n) = X_train.shape
    l1 = X_train.dot(W1) + b1
    h = sigmoid(l1)
    l2 = h.dot(W2) + b2
    y_hat = softmax(l2)

    J = -np.sum(y * np.log(y_hat)) / m # cross entropy
    
    ### backward propagation
    dl2 = y_hat - y
    dW2 = np.dot(h.T, dl2)
    db2 = np.sum(dl2, axis=0)

    dh = np.dot(dl2, W2.T)

    dl1 = dh * sigmoid_grad(h)
    dW1 = np.dot(X_train.T, dl1)
    db1 = np.sum(dl1, axis=0)

    gradW2 = dW2/m
    gradb2 = db2/m
    gradW1 = dW1/m
    gradb1 = db1/m   
    return J, gradW1, gradW2, gradb1, gradb2

def predict(X_train, y, W1, W2, b1 ,b2):
    (m, n) = X_train.shape
    l1 = X_train.dot(W1) + b1
    h = sigmoid(l1)
    l2 = h.dot(W2) + b2
    y_hat = softmax(l2)
    y = y.argmax(1)
    preds = y_hat.argmax(1)
    accuracy = (np.sum(preds == y)/m)
    return accuracy

def give_acc(X_train, y_train, X_test, y_test, w1, w2, b1, b2):
    acc = predict(X_train, y_train, w1, w2, b1, b2)
    t_acc = predict(X_test, y_test, w1, w2, b1, b2)
    return acc, t_acc

def train_nn(para):
    file_ = pd.read_csv('train.csv')
    labels = pd.DataFrame(file_["label"])
    datu = pd.DataFrame(file_.drop("label", axis=1))

    X = datu.values
    (m, n) = X.shape

    X_train = X[0:round(m * (0.9)), :]
    X_test = X[0:(m - round(m * (0.9))), :]

    hidden_neurons = 500
    output_neurons = 10

    y = np.zeros((m, output_neurons))
    i = 0
    for i in range(m):
        y[i, labels.iloc[i]] = 1

    y_train = y[0:round(m * (0.9)), :]
    y_test = y[0:(m - round(m * (0.9))), :]

    w1 = np.random.uniform(high=((6 / (hidden_neurons + n)) ** (1 / 2)), low=-((6 / (hidden_neurons + n)) ** (1 / 2)),
                           size=(n, hidden_neurons))
    b1 = np.zeros((1, hidden_neurons))
    w2 = np.random.uniform(high=((6 / (hidden_neurons + output_neurons)) ** (1 / 2)),
                           low=-((6 / (hidden_neurons + output_neurons)) ** (1 / 2)),
                           size=(hidden_neurons, output_neurons))
    b2 = np.zeros((1, output_neurons))

    """
    lra = 0.001
    batch_size = 100
    i=0
    while i < (len(y_train)-batch_size):
        J, gradW1, gradW2, gradb1, gradb2 = cost_grad(w1, w2, b1, b2, X_train[i:i+batch_size], y_train[i:i+batch_size])
        w2 = w2 - (lra * gradW2)
        b2 = b2 - (lra * gradb2)
        w1 = w1 - (lra * gradW1)
        b1 = b1 - (lra * gradb1)
        print("cost:", J)
        i += batch_size
    """

    # lra = 0.9(1-i/1000)
    # batch_size = 100
    # i=0


    arre = np.zeros((40, 3))
    for i in range(2000):
        J, gradW1, gradW2, gradb1, gradb2 = cost_grad(w1, w2, b1, b2, X_train, y_train)
        lra = 0.09 * (1/(1+np.exp((i-1000)/para)))
        w2 = w2 - (lra * gradW2)
        b2 = b2 - (lra * gradb2)
        w1 = w1 - (lra * gradW1)
        b1 = b1 - (lra * gradb1)
        if (i % 50) == 0:
            acc, t_acc = give_acc(X_train, y_train, X_test, y_test, w1, w2, b1, b2)
            print("cost:", J)
            arre[int(i / 50), 0] = i
            arre[int(i / 50), 1] = acc
            arre[int(i / 50), 2] = t_acc
        i += 1
    pd.DataFrame(np.reshape(w1, (-1))).to_csv("w1_2ly_sig" + str(para) + ".csv", index=False)
    pd.DataFrame(np.reshape(w2, (-1))).to_csv("w2_2ly_sig" + str(para) + ".csv", index=False)
    pd.DataFrame(np.reshape(b1, (-1))).to_csv("b1_2ly_sig" + str(para) + ".csv", index=False)
    pd.DataFrame(np.reshape(b2, (-1))).to_csv("b2_2ly_sig" + str(para) + ".csv", index=False)
    arre = pd.DataFrame({"epoch": arre[:, 0], "train": arre[:, 1], "test": arre[:, 2]})
    arre.to_csv("result_sig"+str(para)+".csv", index=False)

paras = [1, 2, 5, 10]


for i in paras:
    train_nn(i)


 
