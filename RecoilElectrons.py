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
XBins = 120
YBins = 120
ZBins = 120

# File names
FileName = "data\\RecoilElectrons.100k.data"

# Depends on GPU memory and layout
BatchSize = 16

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

print("Info: Number of training data sets: {}   Number of testing data sets: {} (vs. input: {} and split ratio: {})".format(NumberOfTrainingEvents, NumberOfTestingEvents, len(DataSets), TestingTrainingSplit))




###################################################################################################
# Step 5: Setting up the neural network
###################################################################################################


print("Info: Setting up neural network...")


if 1 == 1:
  print("Info: Using \"original\" neural network layout")

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
    except RuntimeError as e:
      print(e)

  inputs_xy = keras.Input(shape=(172,172,1))
  inputs_yz = keras.Input(shape=(172,172,1))
  inputs_zx = keras.Input(shape=(172,172,1))


  c1_xy = layers.Conv2D(512, (3, 3), activation='relu', padding="same")
  c1norm_xy = layers.BatchNormalization()
  c1b_xy = layers.MaxPooling2D((2, 2))
  c2_xy = layers.Conv2D(512, (3, 3), activation='relu', padding="same")
  c2norm_xy = layers.BatchNormalization()
  c2b_xy = layers.MaxPooling2D((2, 2))
  c3_xy = layers.Conv2D(256, (3, 3), activation='relu', padding="same")
  c3norm_xy = layers.BatchNormalization()
  c3b_xy = layers.MaxPooling2D((2, 2))
  c4_xy = layers.Conv2D(128, (3, 3), activation='relu', padding="same")


  c1_yz = layers.Conv2D(512, (3, 3), activation='relu', padding="same")
  c1norm_yz = layers.BatchNormalization()
  c1b_yz = layers.MaxPooling2D((2, 2))
  c2_yz = layers.Conv2D(512, (3, 3), activation='relu', padding="same")
  c2norm_yz = layers.BatchNormalization()
  c2b_yz = layers.MaxPooling2D((2, 2))
  c3_yz = layers.Conv2D(256, (3, 3), activation='relu', padding="same")
  c3norm_yz = layers.BatchNormalization()
  c3b_yz = layers.MaxPooling2D((2, 2))
  c4_yz = layers.Conv2D(128, (3, 3), activation='relu', padding="same")

  c1_zx = layers.Conv2D(512, (3, 3), activation='relu', padding="same")
  c1norm_zx = layers.BatchNormalization()
  c1b_zx = layers.MaxPooling2D((2, 2))
  c2_zx = layers.Conv2D(512, (3, 3), activation='relu', padding="same")
  c2norm_zx = layers.BatchNormalization()
  c2b_zx = layers.MaxPooling2D((2, 2))
  c3_zx = layers.Conv2D(256, (3, 3), activation='relu', padding="same")
  c3norm_zx = layers.BatchNormalization()
  c3b_zx = layers.MaxPooling2D((2, 2))
  c4_zx = layers.Conv2D(128, (3, 3), activation='relu', padding="same")


  d1 = layers.Dense(512, activation='relu', kernel_regularizer='l1')
  d2 = layers.Dense(256, activation='relu', kernel_regularizer='l1')
  d3 = layers.Dense(128, activation='relu', kernel_regularizer='l1')
  d4 = layers.Dense(6)

  drop = layers.Dropout(0.0)

  _xy = inputs_xy
  _xy = c1b_xy(drop(c1_xy(_xy)))
  _xy = c2b_xy(drop(c2_xy(_xy)))
  _xy = c3b_xy(drop(c3_xy(_xy)))
  _xy = drop(c4_xy(_xy))
  _xy = layers.Flatten()(_xy)

  _yz = inputs_yz
  _yz = c1b_yz(drop(c1_yz(_yz)))
  _yz = c2b_yz(drop(c2_yz(_yz)))
  _yz = c3b_yz(drop(c3_yz(_yz)))
  _yz = drop(c4_yz(_yz))
  _yz = layers.Flatten()(_yz)

  _zx = inputs_zx
  _zx = c1b_zx(drop(c1_zx(_zx)))
  _zx = c2b_zx(drop(c2_zx(_zx)))
  _zx = c3b_zx(drop(c3_zx(_zx)))
  _zx = drop(c4_zx(_zx))
  _zx = layers.Flatten()(_zx)

  merge = concatenate([_xy, _yz, _zx])

  merge = d1(merge)
  merge = d2(merge)
  merge = d3(merge)
  merge = d4(merge)



  Model = keras.Model(inputs=[inputs_xy, inputs_yz, inputs_zx], outputs=merge)

elif Layout == "andreas":
  print("Info: Using \"andreas\" neural network layout")

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
    except RuntimeError as e:
      print(e)

  Model = models.Sequential()
  Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(XBins, YBins, ZBins, 1), padding="SAME"))
  Model.add(layers.BatchNormalization())
  Model.add(layers.MaxPooling3D((2, 2, 2)))
  Model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding="SAME"))
  #Model.add(layers.BatchNormalization())
  Model.add(layers.MaxPooling3D((2, 2, 2)))
  Model.add(layers.Conv3D(512, (3, 3, 3), activation='relu', padding="SAME"))
  #Model.add(layers.BatchNormalization())
  Model.add(layers.MaxPooling3D((2, 2, 2)))
  Model.add(layers.Conv3D(1024, (3, 3, 3), activation='relu', padding="SAME"))
  #Model.add(layers.BatchNormalization())

  Model.add(layers.Flatten())
  Model.add(layers.Dense(12, activation='relu'))
  Model.add(layers.Dense(OutputDataSpaceSize))
else:
  print("Error: Unknown model: {}".format(Layout))
  sys.exit(0)


Model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1), loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])

Model.summary()



###################################################################################################
# Step 6: Training and evaluating the network
###################################################################################################

print("Info: Training and evaluating the network")

# Train the network
BestLoss = sys.float_info.max
IterationOutputInterval = 10
CheckPointNum = 0


BestLocation = 100000
BestAngle = 180
BestStartCorrect = 0

###################################################################################################

def CheckPerformance():
  global BestLocation
  global BestAngle
  global BestStartCorrect

  Improvement = False

  TotalEvents = 0
  SumDistDiff = 0
  SumAngleDiff = 0
  SumStartCorrect = 0

  # Step run all the testing batches, and detrmine the percentage of correct identifications
  # Step 1: Loop over all Testing batches
  for Batch in range(0, NTestingBatches):

    # Step 1.1: Convert the data set into the input and output tensor
    xyInput = np.zeros(shape=(BatchSize, XBins, YBins, 1))
    yzInput = np.zeros(shape=(BatchSize, XBins, YBins, 1))
    zxInput = np.zeros(shape=(BatchSize, XBins, YBins, 1))
    OutputTensor = np.zeros(shape=(BatchSize, 6))
    # Loop over all testing  data sets and add them to the tensor
    for e in range(0, BatchSize):
      Event = TestingDataSets[e + Batch*BatchSize]
      # Set the layer in which the event happened

      # Set all the hit locations and energies
      for h in range(0, len(Event.X)):
        XBin = int((Event.X[h] - XMin) / ((XMax - XMin) / XBins) )
        YBin = int((Event.Y[h] - YMin) / ((YMax - YMin) / YBins) )
        ZBin = int((Event.Z[h] - ZMin) / ((ZMax - ZMin) / ZBins) )
        #print("hit z bin: {} {}".format(Event.Z[h], ZBin))
        if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
          xyInput[e][XBin][YBin][0] = Event.E[h]
          yzInput[e][YBin][ZBin][0] = Event.E[h]
          zxInput[e][ZBin][XBin][0] = Event.E[h]

      OutputTensor[e][0] = Event.TrackRealStartX
      OutputTensor[e][1] = Event.TrackRealStartY
      OutputTensor[e][2] = Event.TrackRealStartZ
      OutputTensor[e][3] = Event.TrackRealDirectionX
      OutputTensor[e][4] = Event.TrackRealDirectionY
      OutputTensor[e][5] = Event.TrackRealDirectionZ


    print(Model.predict([xyInput[:1], yzInput[:1], zxInput[:1]]), OutputTensor[:1])
    # Step 2: Run it
    # Result = Session.run(Output, feed_dict={X: InputTensor})
    Result = Model.predict([xyInput, yzInput, zxInput])

    #print(Result[e])
    #print(OutputTensor[e])

    for e in range(0, BatchSize):
      Event = TestingDataSets[e + Batch*BatchSize]

      oPos = np.array([ Event.TrackRealStartX, Event.TrackRealStartY, Event.TrackRealStartZ])
      rPos = np.array([Result[e][0], Result[e][1], Result[e][2]])

      oDir = np.array([ Event.TrackRealDirectionX, Event.TrackRealDirectionY, Event.TrackRealDirectionZ])
      rDir = np.array([ Result[e][3], Result[e][4],Result[e][5]])

      # Distance difference location:
      DistDiff = np.linalg.norm(oPos - rPos)

      # Angle difference direction
      Norm = np.linalg.norm(oDir)
      if Norm == 0:
        print("Warning: original direction is zero: {} {} {}".format(Event.DirectionStartX, Event.DirectionStartY, Event.DirectionStartZ))
        continue
      oDir /= Norm
      Norm = np.linalg.norm(rDir)
      if Norm == 0:
        print("Warning: estimated direction is zero: {} {} {}".format(Result[e][3], Result[e][4], Result[e][5]))
        continue
      rDir /= Norm
      AngleDiff = np.arccos(np.clip(np.dot(oDir, rDir), -1.0, 1.0)) * 180/math.pi

      if math.isnan(AngleDiff):
        continue


      # Found closest start
      MinDist = 1000000
      MinPos = 0
      for h in range(0, len(Event.X)):
        hPos = np.array([ Event.X[h], Event.Y[h], Event.Z[h]])
        if np.linalg.norm(hPos - rPos) < MinDist:
          MinDist = np.linalg.norm(hPos - rPos)
          MinPos = h

      # The first one is always the correct start:
      if MinPos == 0:
        SumStartCorrect += 1

      SumDistDiff += DistDiff
      SumAngleDiff += AngleDiff
      TotalEvents += 1


      # Some debugging
      if Batch == 0 and e < 50:
        EventID = e + Batch*BatchSize + NTrainingBatches*BatchSize
        print("\nEvent {}:".format(EventID))
        DataSets[EventID].print()

        print("Positions: {} vs {} -> {} cm difference".format(oPos, rPos, DistDiff))
        print("Directions: {} vs {} -> {} degree difference".format(oDir, rDir, AngleDiff))

  if TotalEvents > 0:
    if SumDistDiff / TotalEvents < BestLocation and SumAngleDiff / TotalEvents < BestAngle:
      BestLocation = SumDistDiff / TotalEvents
      BestAngle = SumAngleDiff / TotalEvents
      Improvement = True
    if SumStartCorrect / TotalEvents > BestStartCorrect:
      BestStartCorrect = SumStartCorrect / TotalEvents

    print("\n\n--> Status: distance difference = {:-6.2f} cm (best: {:-6.2f} cm), angle difference = {:-6.2f} deg (best: {:-6.2f} deg), start correct = {:5.2f}% (best: {:5.2f}%)\n\n".format(SumDistDiff / TotalEvents, BestLocation, SumAngleDiff / TotalEvents, BestAngle, 100 * SumStartCorrect / TotalEvents, 100.0 * BestStartCorrect))

  return Improvement


###################################################################################################



# Main training and evaluation loop

TimeConverting = 0.0
TimeTraining = 0.0
TimeTesting = 0.0

Iteration = 0
MaxIterations = 50000
TimesNoImprovement = 0
MaxTimesNoImprovement = 50
while Iteration < MaxIterations:
  Iteration += 1
  print("\n\nStarting iteration {}".format(Iteration))

  # Step 1: Loop over all training batches
  for Batch in range(0, NTrainingBatches):
    print("Batch {} / {}".format(Batch+1, NTrainingBatches))

    # Step 1.1: Convert the data set into the input and output tensor
    TimerConverting = time.time()

    InputTensor = np.zeros(shape=(BatchSize, 3, XBins, YBins, 1))
    OutputTensor = np.zeros(shape=(BatchSize, 6))
    xyInput = np.zeros(shape=(BatchSize, XBins, YBins, 1))
    yzInput = np.zeros(shape=(BatchSize, XBins, YBins, 1))
    zxInput = np.zeros(shape=(BatchSize, XBins, YBins, 1))
    # Loop over all training data sets and add them to the tensor
    for g in range(0, BatchSize):
      Event = TrainingDataSets[g + Batch*BatchSize]

      # Set all the hit locations and energies
      for h in range(0, len(Event.X)):
        XBin = int( (Event.X[h] - XMin) / ((XMax - XMin) / XBins) )
        YBin = int( (Event.Y[h] - YMin) / ((YMax - YMin) / YBins) )
        ZBin = int( (Event.Z[h] - ZMin) / ((ZMax - ZMin) / ZBins) )
        if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
          xyInput[g][XBin][YBin][0] = Event.E[h]
          yzInput[g][YBin][ZBin][0] = Event.E[h]
          zxInput[g][ZBin][XBin][0] = Event.E[h]

      OutputTensor[g][0] = Event.TrackRealStartX
      OutputTensor[g][1] = Event.TrackRealStartY
      OutputTensor[g][2] = Event.TrackRealStartZ
      OutputTensor[g][3] = Event.TrackRealDirectionX
      OutputTensor[g][4] = Event.TrackRealDirectionY
      OutputTensor[g][5] = Event.TrackRealDirectionZ

    TimeConverting += time.time() - TimerConverting


    # Step 1.2: Perform the actual training
    TimerTraining = time.time()
    History = Model.fit([xyInput, yzInput, zxInput], OutputTensor)
    print(Model.predict([xyInput[0:1], yzInput[:1], zxInput[:1]]), OutputTensor[:1])
    Loss = History.history['loss'][-1]
    TimeTraining += time.time() - TimerTraining

    if Interrupted == True: break

  # End for all batches

  # Step 2: Check current performance
  TimerTesting = time.time()
  print("\nCurrent loss: {}".format(Loss))
  Improvement = CheckPerformance()

  if Improvement == True:
    TimesNoImprovement = 0

    print("\nFound new best model and performance!")
  else:
    TimesNoImprovement += 1

  TimeTesting += time.time() - TimerTesting

  # Exit strategy
  if TimesNoImprovement == MaxTimesNoImprovement:
    print("\nNo improvement for {} iterations. Quitting!".format(MaxTimesNoImprovement))
    break;

  print("\n\nTotal time converting per Iteration: {} sec".format(TimeConverting/Iteration))
  print("Total time training per Iteration:   {} sec".format(TimeTraining/Iteration))
  print("Total time testing per Iteration:    {} sec".format(TimeTesting/Iteration))

  # Take care of Ctrl-C
  if Interrupted == True: break


# End: for all iterations


#input("Press [enter] to EXIT")
sys.exit(0)
