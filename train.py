import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def reshape_into_sliding_windows(X, windowSize, advanceSamples=1):
    # determine number of sliding windows that fit within dataset
    nWindows = int(np.floor((X.shape[0] - windowSize) / (advanceSamples * 1.0)))

    # pre-allocate matrix which holds sliding windows
    outputMatrix = np.zeros((nWindows, windowSize))

    # populate each sliding window
    for iWindow in range(nWindows):
        startIndex = iWindow * advanceSamples
        endIndex = startIndex + windowSize

        outputMatrix[iWindow, :] = X[startIndex:endIndex, 0]

    return outputMatrix

trainingData = np.load('data/2018-01-01__2019-01-01__NConservatory_npWeekdayAllOrderedSensorsTimeRef.npy')

import torch, torch.nn as nn, time
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

dataLoaderTrain = DataLoader( trainingData.astype('float32'),
                                 batch_size = 16,
                                 shuffle = True )

dataLoaderTest = DataLoader( trainingData.astype('float32'),
                                 batch_size = 1,
                                 shuffle = False )


inputDimensionality = trainingData.shape[1]
hidden_size = inputDimensionality // 2
num_layers = 256
dropout = 0.2
model = nn.Sequential (
    nn.LSTM(input_size=inputDimensionality, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, dropout=dropout),
#     nn.LSTM(input_size=inputDimensionality//2, hidden_size=hidden_size, num_layers=num_layers)
#     nn.LSTM(input_size=inputDimensionality//4, hidden_size=hidden_size, num_layers=num_layers)
#     nn.LSTM(input_size=inputDimensionality//10, hidden_size=hidden_size, num_layers=num_layers)
#     nn.LSTM(input_size=inputDimensionality//4, hidden_size=hidden_size, num_layers=num_layers)
    nn.LSTM(input_size=hidden_size, hidden_size=inputDimensionality, num_layers=num_layers)
#     nn.Linear(inputDimensionality, inputDimensionality//2), nn.Sigmoid(),
#     nn.Linear(inputDimensionality//2, inputDimensionality//4), nn.Sigmoid(),
#     nn.Linear(inputDimensionality//4, inputDimensionality//10), nn.Sigmoid(),
#     nn.Linear(inputDimensionality//10, inputDimensionality//4), nn.Sigmoid(),
#     nn.Linear(inputDimensionality//4, inputDimensionality//2), nn.Sigmoid(),
#     nn.Linear(inputDimensionality//2, inputDimensionality)
)

def train_model(model, dataLoader, targeDevice, nEpochs=10):
    # --model = model.to( targetDevice )

    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    lossHistory = []

    # training loop
    for iEpoch in range(nEpochs):
        cumulativeLoss = 0
        for i, iInputBatch in enumerate(dataLoader):
            # move batch data to target training device [ cpu or gpu ]
            # --iInputBatch = iInputBatch.to( targetDevice )

            # zero/reset the parameter gradient buffers to avoid accumulation [ usually accumulation is necessary for temporally unrolled networks ]
            optimizer.zero_grad()

            # generate predictions/reconstructions
            predictions = model.forward(iInputBatch)

            # compute error
            loss = lossFunction(predictions, iInputBatch)
            cumulativeLoss += loss.item()  # gets scaler value held in the loss tensor

            # compute gradients by propagating the error backward through the model/graph
            loss.backward()

            # apply gradients to update model parameters
            optimizer.step()

        print('epoch {} of {} -- avg batch loss: {}'.format(iEpoch, nEpochs, cumulativeLoss))

        lossHistory += [cumulativeLoss]
    return model, lossHistory


targetDeviceCPU = torch.device('cpu')
targetDeviceGPU = torch.device('cuda:0')
targetDevice = targetDeviceGPU
model, lossHistory = train_model(model, dataLoaderTrain, targetDevice, nEpochs=5)
