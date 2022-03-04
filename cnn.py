import numpy as np
import matplotlib.pyplot as plt
import gzip
import copy

IMG_DIM = 28
TRAINING_SAMPLES = 60000
NUM_PX = IMG_DIM * IMG_DIM

def decode_image_file(fname):
    result = []
    n_bytes_per_img = IMG_DIM*IMG_DIM

    with gzip.open(fname, 'rb') as f:
        bytes_ = f.read()
        data = bytes_[16:]

        if len(data) % n_bytes_per_img != 0:
            raise Exception('Something wrong with the file')

        result = np.frombuffer(data, dtype=np.uint8).reshape(
            len(bytes_)//n_bytes_per_img, n_bytes_per_img)

    return result

def decode_label_file(fname):
    result = []

    with gzip.open(fname, 'rb') as f:
        bytes_ = f.read()
        data = bytes_[8:]

        result = np.frombuffer(data, dtype=np.uint8)

    return result

def sigmoid(x):
    return 1/(1 + np.exp(-x))

    """
    w=  [[w1],
         [w2],
         [w3],
         ...
         [wn]]
    """
    
def propagate(w,b,x,y):
    m = x.shape[1]
    a = sigmoid(np.dot(x,w) + b)
    
    cost = -(1/m)*(np.dot(y.T,np.log(a))+np.dot((1-y).T,np.log(1-a)))
    cost = np.squeeze(cost)
    
    dw = np.dot(x.T,(a-y))/m
    db = np.sum(a-y)/m
    grads = {"dw":dw, "db":db}
    
    return grads, cost

def optimize(w,b,x,y,iterations, learningRate, print_cost=False):
    costs = []
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    for i in range(iterations):
        grads, cost = propagate(w,b,x,y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w -= learningRate*dw
        b -= learningRate*db

        if (i % 50 == 0):
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f" %(i, cost))


    params = {"w":w, "b":b}
    grads = {"dw":dw, "db":db}
    return params, grads, costs
    
def predict(w,b,x):
    a = np.round(sigmoid(np.dot(x,w) + b)*9)
    return a
    
def run(w,b,trainImages,trainLabels,testImages,testLabels,iterations, learningRate, reuseData = True):
    if reuseData:
        with open('data/params.txt', 'r') as f:
            lines = f.readlines()
            for i in range(NUM_PX):
                w[i][0] = float(lines[i])
            b = float(lines[NUM_PX])
    
    params, grads, costs = optimize(w,b,trainImages,trainLabels,iterations, learningRate, True)
    w = params["w"]
    b = params["b"]

    with open("data/params.txt", "w") as f:
        for i in range(NUM_PX):
            f.write(str(w[i][0])+"\n")
        f.write(str(b))

    trainPrediction = predict(w,b,trainImages)
    testPrediction = predict(w,b,testImages)

    trainLabels *= 9
    testLabels *= 9

    print("train accuracy: {} %".format(100*np.sum(trainLabels == trainPrediction)/TRAINING_SAMPLES))
    print("test accuracy: {} %".format(100*np.sum(testLabels == testPrediction)/len(testLabels)))
    return params

    
trainImages = decode_image_file('data/train-images-idx3-ubyte.gz')/255.0
trainLabels = decode_label_file('data/train-labels-idx1-ubyte.gz')/9.0
testImages = decode_image_file('data/t10k-images-idx3-ubyte.gz')/255.0
testLabels = decode_label_file('data/t10k-labels-idx1-ubyte.gz')/9.0

trainImages = trainImages[:TRAINING_SAMPLES]
trainLabels = np.reshape(trainLabels[:TRAINING_SAMPLES], (TRAINING_SAMPLES,1))
testImages = testImages
testLabels = np.reshape(testLabels, (len(testLabels),1))

w = np.zeros((NUM_PX,1))
b = 10.0

params = run(w,b,trainImages,trainLabels,testImages,testLabels,100, 0.003, True)
