###################################################################################################
#
# RecoilElectrons.py
#
# Copyright (C) by Andreas Zoglauer & contributors
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################



###################################################################################################


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers.merge import concatenate
from pointnet.utils.data_prep_util import *

import numpy as np

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

import random
import pickle

import signal
import sys
import time
import math
import csv
import os
import argparse
from datetime import datetime
from functools import reduce

print("\nCompton Track Identification")
print("============================\n")



# Step 1: Input parameters
###################################################################################################


# Default parameters

# X, Y, Z bins
XBins = 170
YBins = 170
ZBins = 170

# File names
FileName = "data\\RecoilElectrons.100k.data"

# Depends on GPU memory and layout
BatchSize = 8

# Split between training and testing data
TestingTrainingSplit = 0.1

# Maximum number of events to use
MaxEvents = 1000000

# The network Layout
Layout = "original"

# Dimensions of the tracker
XMin = -43
XMax = 43

YMin = -43
YMax = 43

ZMin = 13
ZMax = 45

# Output data space size (3 location + 3 direction)
OutputDataSpaceSize = 6

OutputDirectory = "Results"


parser = argparse.ArgumentParser(description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='data\\RecoilElectrons.100k.data', help='File name with training/testing data')
parser.add_argument('-m', '--maxevents', default=MaxEvents, help='Maximum number of events to use')
parser.add_argument('-s', '--testingtrainingsplit', default=TestingTrainingSplit, help='Testing-training split')
parser.add_argument('-b', '--batchsize', default=BatchSize, help='Batch size')
parser.add_argument('-c', '--cpuonly', default=False, action="store_true", help='Limit to CPU')
parser.add_argument('-l', '--layout', default=Layout, help='One of: default, andreas')

args = parser.parse_args()

if args.filename != "":
  FileName = args.filename
if not os.path.exists(FileName):
  print("Error: The training data file does not exist: {}".format(FileName))
  sys.exit(0)

if int(args.maxevents) > 1000:
  MaxEvents = int(args.maxevents)
else:
  print("Warning: You cannot use less then 1000 events")
  MaxEvents = 1000

if int(args.batchsize) >= 1:
  BatchSize = int(args.batchsize)
else:
  print("Warning: Minimum batch size is 1")
  BatchSize = 1

if float(args.testingtrainingsplit) >= 0.05:
   TestingTrainingSplit = float(args.testingtrainingsplit)
else:
  print("Warning: Minimum testing-training split is 0.05 (5%)")
  TestingTrainingSplit = 0.05

CPUOnly = False
if args.cpuonly == True:
  CPUOnly = True
  os.environ["CUDA_VISIBLE_DEVICES"]="-1"

Layout = args.layout
if not Layout == "original" and not Layout == "andreas":
  print("Error: The neural network layout must be one of [original, andreas], and not: {}".format(Layout))
  sys.exit(0)

#if os.path.exists(OutputDirectory):
#  Now = datetime.now()
#  OutputDirectory += Now.strftime("_%Y%m%d_%H%M%S")
#os.makedirs(OutputDirectory)



###################################################################################################
# Step 2: Global functions
###################################################################################################


# Take care of Ctrl-C
Interrupted = False
NInterrupts = 0
def signal_handler(signal, frame):
  global Interrupted
  Interrupted = True
  global NInterrupts
  NInterrupts += 1
  if NInterrupts >= 2:
    print("Aborting!")
    sys.exit(0)
  print("You pressed Ctrl+C - waiting for graceful abort, or press Ctrl-C again, for quick exit.")
signal.signal(signal.SIGINT, signal_handler)


# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
from EventData import EventData


###################################################################################################
# Step 3: Read the data
###################################################################################################


print("\n\nStarted reading data sets")

with open(FileName, "rb") as FileHandle:
   DataSets = pickle.load(FileHandle)

if len(DataSets) > MaxEvents:
  DataSets = DataSets[:MaxEvents]

NumberOfDataSets = len(DataSets)


print("Info: Parsed {} events".format(NumberOfDataSets))


###################################################################################################
# Step 4: Split the data into training, test & verification data sets
###################################################################################################


# Split the data sets in training and testing data sets

# The number of available batches in the inoput data
NBatches = int(len(DataSets) / BatchSize)
if NBatches < 2:
  print("Not enough data!")
  quit()

# Split the batches in training and testing according to TestingTrainingSplit
NTestingBatches = int(NBatches*TestingTrainingSplit)
if NTestingBatches == 0:
  NTestingBatches = 1
NTrainingBatches = NBatches - NTestingBatches

# Now split the actual data:
TrainingDataSets = []
for i in range(0, NTrainingBatches * BatchSize):
  TrainingDataSets.append(DataSets[i])


TestingDataSets = []
for i in range(0,NTestingBatches*BatchSize):
   TestingDataSets.append(DataSets[NTrainingBatches * BatchSize + i])


NumberOfTrainingEvents = len(TrainingDataSets)
NumberOfTestingEvents = len(TestingDataSets)


# Train the network
BestLoss = sys.float_info.max
IterationOutputInterval = 10
CheckPointNum = 0


BestLocation = 100000
BestAngle = 180
BestStartCorrect = 0


TimeConverting = 0.0
TimeTraining = 0.0
TimeTesting = 0.0

Iteration = 0
MaxIterations = 50000
TimesNoImprovement = 0
MaxTimesNoImprovement = 50

num_data = 10

point_cloud = list()
labels = list()
# Loop over all training data sets and add them to the tensor
for g in range(num_data):
  point_cloud.append(list())
  Event = TrainingDataSets[g]

  # Set all the hit locations and energies
  for h in range(0, len(Event.X)):
    XBin = int( (Event.X[h] - XMin) / ((XMax - XMin) / XBins) )
    YBin = int( (Event.Y[h] - YMin) / ((YMax - YMin) / YBins) )
    ZBin = int( (Event.Z[h] - ZMin) / ((ZMax - ZMin) / ZBins) )
    if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
      point_cloud[-1].append([XBin, YBin, ZBin, Event.E[h]])

  labels.append(Event.TrackRealStartX)


labels = np.array(labels)
point_cloud = np.array(point_cloud)

save_h5('test', point_cloud, labels, data_dtype='float32')

print(point_cloud)



sys.exit(0)
