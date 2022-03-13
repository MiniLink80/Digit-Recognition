import numpy as np
import cnn

TRAINING_SAMPLES = 100

trainImages = cnn.decode_image_file('data/train-images-idx3-ubyte.gz')/255.0
trainLabels = cnn.decode_label_file('data/train-labels-idx1-ubyte.gz')/9.0
testImages = cnn.decode_image_file('data/t10k-images-idx3-ubyte.gz')/255.0
testLabels = cnn.decode_label_file('data/t10k-labels-idx1-ubyte.gz')/9.0

trainLabels = np.reshape(trainLabels[:TRAINING_SAMPLES], (TRAINING_SAMPLES,1))
testLabels = np.reshape(testLabels, (len(testLabels),1))
trainImages = trainImages[:TRAINING_SAMPLES]

def sigmoid(a):
    return 1/(1+np.exp(-a))
def relu(a):
    return np.maximum(0,a)
def dsigmoid(a):
    s = sigmoid(a)
    return np.multiply(s, 1-s)
def drelu(a):
    return np.where(a < 0, 0, 1)
def sigmoidBack(dyhat, actCache):
    return np.multiply(dyhat, dsigmoid(actCache))
def reluBack(dyhat, actCache):
    return np.multiply(dyhat, drelu(actCache))

def initializeParameters(dims):
    params = {}
    np.random.seed(31)
    for i in range(1, len(dims)):
        params["w"+str(i)] = np.random.randn(dims[i], dims[i-1]) * 0.01
        params["b"+str(i)] = np.zeros((dims[i],1))
    return params


def linearForward(Aprev, w, b, act):
    if act == "relu":
        z, linCache = np.dot(w,Aprev)+b, (Aprev, w, b)
        A, actCache = relu(z), z
    else:
        z, linCache = np.dot(w,Aprev)+b, (Aprev, w, b)
        A, actCache = sigmoid(z), z
        
    return A, (linCache, actCache)

def modelForward(X, params):
    caches = []
    A = X
    L = len(params)//2

    for i in range(1, L):
        #relu
        Aprev = A
        A, cache = linearForward(Aprev, params["w"+str(i)], params["b"+str(i)], "relu")
        caches.append(cache)
    
    AL, cache = linearForward(A, params["w"+str(L)], params["b"+str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches

def getCost(yhat, y):
    m = y.shape[1]
    cost = -np.sum(np.dot(y.T,np.log(yhat).T)+np.dot(1-y.T, np.log(1-yhat).T))/m
    return np.squeeze(cost)

def linearBackward(dyhat, cache, act):
    linCache, actCache = cache
    dz = []
    if act == "relu":
        dz = reluBack(dyhat, actCache)
    else:
        dz = sigmoidBack(dyhat, actCache)
    
    aprev, w, b = linCache
    m = aprev.shape[1]
    dw = np.dot(dz, aprev.T)/m
    db = np.sum(dz, axis=1, keepdims=True)/m
    daprev = np.dot(w.T, dz)
    return daprev, dw, db

def modelBackward(yhat, y, caches):
    grads = {}
    L = len(caches)
    m = y.shape[1]
    print(y, yhat)
    y = y.reshape(yhat.shape)
    dyhat = -(np.divide(y, yhat) - np.divide(1-y, 1-yhat))
    
    curCache = caches[L-1]
    daPrev, dw, db = linearBackward(dyhat, curCache, "sigmoid")
    grads["da" + str(L-1)] = daPrev
    grads["dw" + str(L)] = dw
    grads["db" + str(L)] = db

    for i in reversed(range(L-1)):
        curCache = caches[i]
        daPrev, dw, db = linearBackward(grads["da" + str(i+1)], curCache, "relu")
        grads["da" + str(i)] = daPrev
        grads["dw" + str(i+1)] = dw
        grads["db" + str(i+1)] = db
    
    return grads

def updateParams(params, grads, learnRate):
    parameters = params.copy()
    L = len(parameters)//2
    for i in range(L):
        parameters["w" + str(i+1)] = parameters["w" + str(i+1)] - learnRate*grads["dw" + str(i+1)]
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - learnRate*grads["db" + str(i+1)]
    return parameters

def model(trainX, trainY, testX, testY, iter, learnRate, layerDims, reuseParams = False):
    params = {}
    costs = []
    if not reuseParams:
        params = initializeParameters(layerDims)
    
    for i in range(iter):
        yhat, caches = modelForward(trainX, params)
        cost = getCost(yhat, trainY)
        grads = modelBackward(yhat, trainY, caches)
        params = updateParams(params, grads, learnRate)
        if i % 8 == 0 or i == iter-1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
    return params

def test(trainX, trainY, testX, testY, params):
    ytrain, cache = modelForward(trainX, params)
    ytest, cache = modelForward(testX, params)
    print(ytrain)

    ytrain = np.round(ytrain*9)
    ytest = np.round(ytest*9)
    print("train accuracy: {} %".format(100*np.sum(trainY == ytrain)/TRAINING_SAMPLES))
    print("test accuracy: {} %".format(100*np.sum(testY == ytest)/len(testY)))

params = model(trainImages.T, trainLabels, testImages.T, testLabels, 25, 0.1, [28*28, 20,20, 1])
test(trainImages.T, trainLabels, testImages.T, testLabels, params)
#print(params)