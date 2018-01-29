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


def cost_grad(W1, W2, W3, b1, b2, b3, X_train, y):
    (m, n) = X_train.shape
    l1 = X_train.dot(W1) + b1
    h = sigmoid(l1)
    l2 = h.dot(W2) + b2
    h2 = sigmoid(l2)
    l3 = h2.dot(w3) + b3
    y_hat = softmax(l3)

    J = -np.sum(y * np.log(y_hat)) / m # cross entropy
    
    ### backward propagation
    dl3 = y_hat - y
    dW3 = np.dot(h2.T, dl3)
    db3 = np.sum(dl3, axis=0)

    dh2 = np.dot(dl3, W3.T)

    dl2 = dh2 * sigmoid_grad(h2)
    dW2 = np.dot(h.T, dl2)
    db2 = np.sum(dl2, axis=0)

    dh = np.dot(dl2, W2.T)

    dl1 = dh * sigmoid_grad(h)
    dW1 = np.dot(X_train.T, dl1)
    db1 = np.sum(dl1, axis=0)

    gradW3 = dW3/m
    gradb3 = db3/m
    gradW2 = dW2/m
    gradb2 = db2/m
    gradW1 = dW1/m
    gradb1 = db1/m   
    return J, gradW1, gradW2, gradb1, gradb2, gradW3, gradb3

def predict(X_train, y, W1, W2, W3, b1 ,b2, b3):
    (m, n) = X_train.shape
    l1 = X_train.dot(W1) + b1
    h = sigmoid(l1)
    l2 = h.dot(W2) + b2
    h2 = sigmoid(l2)
    l3 = h2.dot(w3) + b3
    y_hat = softmax(l3)
    y = y.argmax(1)
    preds = y_hat.argmax(1)
    accuracy = (np.sum(preds == y)/m)
    return accuracy

def give_acc(X_train, y_train, X_test, y_test, w1, w2, w3, b1, b2, b3):
    acc = predict(X_train, y_train, w1, w2, w3, b1, b2, b3)
    t_acc = predict(X_test, y_test, w1, w2, w3, b1, b2, b3)
    return acc, t_acc

def train_nn(hidden_neurons1, hidden_neurons2):
    file_ = pd.read_csv('train.csv')
    labels = pd.DataFrame(file_["label"])
    datu = pd.DataFrame(file_.drop("label", axis=1))

    X = datu.values
    (m, n) = X.shape

    X_train = X[0:round(m * (0.9)), :]
    X_test = X[0:(m - round(m * (0.9))), :]

    output_neurons = 10

    y = np.zeros((m, output_neurons))
    i = 0
    for i in range(m):
        y[i, labels.iloc[i]] = 1

    y_train = y[0:round(m * (0.9)), :]
    y_test = y[0:(m - round(m * (0.9))), :]

    w1 = np.random.uniform(high=((6 / (hidden_neurons1 + n)) ** (1 / 2)), low=-((6 / (hidden_neurons1 + n)) ** (1 / 2)),
                           size=(n, hidden_neurons1))
    b1 = np.zeros((1, hidden_neurons1))
    w2 = np.random.uniform(high=((6 / (hidden_neurons1 + hidden_neurons2)) ** (1 / 2)),
                           low=-((6 / (hidden_neurons1 + hidden_neurons2)) ** (1 / 2)),
                           size=(hidden_neurons1, hidden_neurons2))
    b2 = np.zeros((1, hidden_neurons2))
    w3 = np.random.uniform(high=((6 / (hidden_neurons2 + output_neurons)) ** (1 / 2)),
                           low=-((6 / (hidden_neurons2 + output_neurons)) ** (1 / 2)),
                           size=(hidden_neurons2, output_neurons))
    b3 = np.zeros((1, output_neurons))
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
    start = timeit.default_timer()
    for i in range(2000):
        J, gradW1, gradW2, gradb1, gradb2, gradW3, gradb3 = cost_grad(w1, w2, w3, b1, b2, b3, X_train, y_train)
        lra = 0.09 * ((2000 - i) / 2000)
        w3 = w3 - (lra * gradW3)
        b3 = b3 - (lra * gradb3)
        w2 = w2 - (lra * gradW2)
        b2 = b2 - (lra * gradb2)
        w1 = w1 - (lra * gradW1)
        b1 = b1 - (lra * gradb1)
        if (i % 50) == 0:

            print("cost:", J)
        i += 1
    end = time.process_time()
    time_taken = end - start
    _, t_acc = give_acc(X_train, y_train, X_test, y_test, w1, w2, w3, b1, b2, b3)
    pd.DataFrame(np.reshape(w1, (-1))).to_csv("w1_3ly"+str(hidden_neurons1)+"_"+str(hidden_neurons2)+".csv", index=False)
    pd.DataFrame(np.reshape(w2, (-1))).to_csv("w2_3ly"+str(hidden_neurons1)+"_"+str(hidden_neurons2)+".csv", index=False)
    pd.DataFrame(np.reshape(w3, (-1))).to_csv("w3_3ly"+str(hidden_neurons1)+"_"+str(hidden_neurons2)+".csv", index=False)
    pd.DataFrame(np.reshape(b1, (-1))).to_csv("b1_3ly"+str(hidden_neurons1)+"_"+str(hidden_neurons2)+".csv", index=False)
    pd.DataFrame(np.reshape(b2, (-1))).to_csv("b2_3ly"+str(hidden_neurons1)+"_"+str(hidden_neurons2)+".csv", index=False)
    pd.DataFrame(np.reshape(b3, (-1))).to_csv("b3_3ly"+str(hidden_neurons1)+"_"+str(hidden_neurons2)+".csv", index=False)
    return t_acc, time_taken



vals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
outc = np.zeros((10,10))
outc_time = np.zeros((10,10))
for k in vals:
    for j in vals:
        accu, time = train_nn(k,j)
        outc[((0.01*k)-1),((0.01*j)-1)] = accu
        outc_time[((0.01 * k) - 1), ((0.01 * j) - 1)] = time

pd.DataFrame(outc).to_csv("3_ly_best.csv", index=False)
pd.DataFrame(outc_time).to_csv("3_ly_best_time.csv", index=False)
 
