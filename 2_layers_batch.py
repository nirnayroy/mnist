import pandas as pd
import numpy as np
import timeit

def softmax(x):

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

def train_nn(batch_size):
    file_ = pd.read_csv('train.csv')
    labels = pd.DataFrame(file_["label"])
    datu = pd.DataFrame(file_.drop("label", axis=1))

    X = datu.values
    (m, n) = X.shape

    X_train = X[0:round(m * (0.9)), :]
    X_test = X[0:(m - round(m * (0.9))), :]

    (m_train, n_train) = X_train.shape

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
    #batch_size = 100
    i=0
    start = timeit.default_timer()
    while i<2000:
        idx = np.random.randint(m_train, size=batch_size)
        J, gradW1, gradW2, gradb1, gradb2 = cost_grad(w1, w2, b1, b2, X_train[idx,:], y_train[idx,:])
        lra = 0.9*(1 - (i / 2000))
        w2 = w2 - (lra * gradW2)
        b2 = b2 - (lra * gradb2)
        w1 = w1 - (lra * gradW1)
        b1 = b1 - (lra * gradb1)
        if (i % 50) == 0:

            print("cost:", J)
        i += 1
    end = timeit.default_timer()
    time_taken = end - start
    _, t_acc = give_acc(X_train, y_train, X_test, y_test, w1, w2, b1, b2)
    pd.DataFrame(np.reshape(w1, (-1))).to_csv("w1_2ly"+str(hidden_neurons)+".csv", index=False)
    pd.DataFrame(np.reshape(w2, (-1))).to_csv("w2_2ly"+str(hidden_neurons)+".csv", index=False)
    pd.DataFrame(np.reshape(b1, (-1))).to_csv("b1_2ly"+str(hidden_neurons)+".csv", index=False)
    pd.DataFrame(np.reshape(b2, (-1))).to_csv("b2_2ly"+str(hidden_neurons)+".csv", index=False)
    return t_acc, time_taken

paras = [50, 100, 500, 1000, 2000]

outc = np.zeros(5)
outc_time = np.zeros(5)

for q in range(5):
    accu, time = train_nn(paras[q])
    outc[q] = accu
    outc_time[q] = time
    q += 1
pd.DataFrame(outc).to_csv("2_ly_sgd.csv", index=False)
pd.DataFrame(outc_time).to_csv("2_ly_sgd_time.csv", index=False)
 
