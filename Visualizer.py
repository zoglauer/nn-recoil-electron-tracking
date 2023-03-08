import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import EventData
import argparse

MARGIN = .1

def save_pred_projections(
        data: EventData, 
        pred_vec=[0, 0, 0, 0, 0, 0], 
        save_file=None,
        show_vectors=True,
        show_track=False,
        pred_edge_list=None
    ):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    x, y, z, dx, dy, dz = pred_vec

    axs[0, 0].set_title('XY Projection')
    axs[0, 1].set_title('XZ Projection')
    axs[1, 0].set_title('YZ Projection')
    axs[1, 1].remove()
    axs[1, 1] = fig.add_subplot(2, 2, 4, projection='3d')
    axs[1, 1].set_title('3d Projection')

    for i in range(len(data.E)): 
        axs[0, 0].scatter(data.X[i],data.Y[i],color='b') 
        axs[0, 0].text(data.X[i],data.Y[i],  '%d' % data.E[i], size=9, zorder=1, color='k') 
        axs[0, 1].scatter(data.X[i],data.Z[i],color='b') 
        axs[0, 1].text(data.X[i],data.Z[i],  '%d' % data.E[i], size=9, zorder=1, color='k') 
        axs[1, 0].scatter(data.Y[i],data.Z[i],color='b') 
        axs[1, 0].text(data.Y[i],data.Z[i],  '%d' % data.E[i], size=9, zorder=1, color='k') 

    axs[0, 0].set_xlim([min(data.X) - MARGIN, max(data.X) + MARGIN])
    axs[0, 0].set_ylim([min(data.Y) - MARGIN, max(data.Y + MARGIN)])

    axs[0, 1].set_xlim([min(data.X) - MARGIN, max(data.X) + MARGIN])
    axs[0, 1].set_ylim([min(data.Z) - MARGIN, max(data.Z) + MARGIN])

    axs[1, 0].set_xlim([min(data.Y) - MARGIN, max(data.Y) + MARGIN])
    axs[1, 0].set_ylim([min(data.Z) - MARGIN, max(data.Z) + MARGIN])

    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].set_zlabel('z')

    for i in range(len(data.E)): #plot each point + it's index as text above
        axs[1, 1].scatter(data.X[i],data.Y[i],data.Z[i],color='b') 
        axs[1, 1].text(data.X[i],data.Y[i],data.Z[i],  '%d' % data.E[i], size=10, zorder=1, color='k') 

    if show_vectors:
        axs[0, 0].quiver(
            data.TrackRealStartX, 
            data.TrackRealStartY, 
            data.TrackRealDirectionX, 
            data.TrackRealDirectionY,
            color='red', angles='xy', scale_units='xy', scale=1.)
        axs[0, 0].quiver(
            data.TrackMeasuredStartX, 
            data.TrackMeasuredStartY, 
            data.TrackMeasuredDirectionX, 
            data.TrackMeasuredDirectionY,
            color='blue', angles='xy', scale_units='xy', scale=1.)
        axs[0, 0].quiver(x,  y,  dx,  dy, color='g', angles='xy', scale_units='xy', scale=1.)
        axs[0, 0].set(xlabel='x', ylabel='y')
    
        axs[0, 1].quiver(data.TrackRealStartX, 
            data.TrackRealStartZ, 
            data.TrackRealDirectionX, 
            data.TrackRealDirectionZ,
            color='red', angles='xy', scale_units='xy', scale=1.)
        axs[0, 1].set(xlabel='x', ylabel='z')
        axs[0, 1].quiver(
            data.TrackMeasuredStartX, 
            data.TrackMeasuredStartZ, 
            data.TrackMeasuredDirectionX, 
            data.TrackMeasuredDirectionZ,
            color='blue', angles='xy', scale_units='xy', scale=1.)
        axs[0, 1].quiver(x,  z,  dx,  dz, color='g', angles='xy', scale_units='xy', scale=1.)

        axs[1, 0].quiver(data.TrackRealStartY, 
            data.TrackRealStartZ, 
            data.TrackRealDirectionY, 
            data.TrackRealDirectionZ,
            color='red', angles='xy', scale_units='xy', scale=1.)
        axs[1, 0].set(xlabel='y', ylabel='z')
        axs[1, 0].quiver(
            data.TrackMeasuredStartY, 
            data.TrackMeasuredStartZ, 
            data.TrackMeasuredDirectionY, 
            data.TrackMeasuredDirectionZ,
            color='blue', angles='xy', scale_units='xy', scale=1.)
        axs[1, 0].quiver(y,  z,  dy,  dz, color='g', angles='xy', scale_units='xy', scale=1.)

        axs[1, 1].quiver(data.TrackRealStartX, 
            data.TrackRealStartY, 
            data.TrackRealStartZ, 
            data.TrackRealDirectionX, 
            data.TrackRealDirectionY,
            data.TrackRealDirectionZ, 
            color='red',
            label='Real',
            arrow_length_ratio=.1)

        axs[1, 1].quiver(data.TrackMeasuredStartX, 
            data.TrackMeasuredStartY, 
            data.TrackMeasuredStartZ, 
            data.TrackMeasuredDirectionX, 
            data.TrackMeasuredDirectionY,
            data.TrackMeasuredDirectionZ, 
            color='b', 
            label='Measured',
            arrow_length_ratio=.1)

        axs[1, 1].quiver(pred_vec[0],pred_vec[1],pred_vec[2], pred_vec[3],pred_vec[4],pred_vec[5], color='g', label='Predicted', arrow_length_ratio=.1)
        
    if show_track:
        for i in range(len(data.E)-1):
            dx, dy, dz = data.X[i+1] - data.X[i], data.Y[i+1] - data.Y[i], data.Z[i+1] - data.Z[i]
            axs[0, 0].quiver(
                data.X[i], data.Y[i],
                dx, dy,
                color='purple',
                angles='xy', scale_units='xy', scale=1.
            )
            axs[0, 1].quiver(
                data.X[i], data.Z[i],
                dx, dz,
                color='purple',
                angles='xy', scale_units='xy', scale=1.
            )
            axs[1, 0].quiver(
                data.Y[i], data.Z[i],
                dy, dz,
                color='purple',
                angles='xy', scale_units='xy', scale=1.
            )
            axs[1, 1].quiver(
                data.X[i], data.Y[i], data.Z[i],
                dx, dy, dz,
                color='purple',
                arrow_length_ratio=.1
            )

    if pred_edge_list:
        for i, j in pred_edge_list:
            color = "green" if i+1 == j else "orange"
            dx, dy, dz = data.X[j] - data.X[i], data.Y[j] - data.Y[i], data.Z[j] - data.Z[i]
            axs[0, 0].quiver(
                data.X[i], data.Y[i],
                dx, dy,
                color=color,
                angles='xy', scale_units='xy', scale=1.
            )
            axs[0, 1].quiver(
                data.X[i], data.Z[i],
                dx, dz,
                color=color,
                angles='xy', scale_units='xy', scale=1.
            )
            axs[1, 0].quiver(
                data.Y[i], data.Z[i],
                dy, dz,
                color=color,
                angles='xy', scale_units='xy', scale=1.
            )
            axs[1, 1].quiver(
                data.X[i], data.Y[i], data.Z[i],
                dx, dy, dz,
                color=color,
                arrow_length_ratio=.1
            )

    plt.legend(loc="upper left")

   
    ds_hash = abs(hash(data))
    if not save_file:
        save_file = "pictures"

    plt.savefig(f"{save_file}/event_{ds_hash}.png", pad_inches='.05')
    #plt.show()
    plt.close()

def show_pred(data: EventData, 
              pred_vec=[0, 0, 0, 0, 0, 0], 
              show_vectors=True, 
              show_track=True,
              pred_edge_list=None):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.plot(data.X, data.Y, data.Z, label='Electron Path')

    fig.canvas.mpl_connect('key_press_event', on_press)

    for i in range(len(data.E)): #plot each point + it's index as text above
        ax.scatter(data.X[i],data.Y[i],data.Z[i],color='b') 
        ax.text(data.X[i],data.Y[i],data.Z[i],  '%d' % data.E[i], size=10, zorder=1, color='k') 

    if show_track:
        for i in range(len(data.E)-1):
            dx, dy, dz = data.X[i+1] - data.X[i], data.Y[i+1] - data.Y[i], data.Z[i+1] - data.Z[i]
            ax.quiver(
                data.X[i], data.Y[i], data.Z[i],
                dx, dy, dz,
                color='purple',
                arrow_length_ratio=.1
            )

    if pred_edge_list:
        for i, j in pred_edge_list:
            dx, dy, dz = data.X[j] - data.X[i], data.Y[j] - data.Y[i], data.Z[j] - data.Z[i]
            color = "green" if i+1 == j else "orange"
            ax.quiver(
                data.X[i], data.Y[i], data.Z[i],
                dx, dy, dz,
                color='orange',
                arrow_length_ratio=.1
            )
    
    if show_vectors:
        ax.quiver(data.TrackRealStartX, 
            data.TrackRealStartY, 
            data.TrackRealStartZ, 
            data.TrackRealDirectionX, 
            data.TrackRealDirectionY,
            data.TrackRealDirectionZ, 
            color='red',
            arrow_length_ratio=.1)

        ax.quiver(data.TrackMeasuredStartX, 
            data.TrackMeasuredStartY, 
            data.TrackMeasuredStartZ, 
            data.TrackMeasuredDirectionX, 
            data.TrackMeasuredDirectionY,
            data.TrackMeasuredDirectionZ, 
            color='b', 
            arrow_length_ratio=.1)

        ax.quiver(pred_vec[0],pred_vec[1],pred_vec[2], pred_vec[3],pred_vec[4],pred_vec[5], color='g', arrow_length_ratio=.1)

    plt.show()
    #ds_hash = abs(hash(data))
    #plt.savefig(f"pictures/event_{ds_hash}.pdf")
    return

def on_press(event):
    print('press', event.key)
    if event.key == 'n':
        plt.close()

def view_interactive(args):
    f = open(args.data_file, mode='rb')
    data = pickle.load(f)
    for i in range(len(data)):
        show_pred(data[i])

def view_png_format(args):
    f = open(args.data_file, mode='rb')
    data = pickle.load(f)

    n = args.num_save if args.num_save else len(data)
    n = min(n, len(data))
    for i in range(n):
        save_pred_projections(data[i], save_file=args.save_file, show_track=True)

def test():
    f = open('./data/RecoilElectrons.1k.data', mode='rb')
    dataset = pickle.load(f)

    XBins = 64
    YBins = 64
    ZBins = 64

    XMin = -43
    XMax = 43

    YMin = -43
    YMax = 43

    ZMin = 13
    ZMax = 45

    # Output data space size (3 location + 3 direction)
    OutputDataSpaceSize = 6
    NUM_SAMPLES = 10
    InputTensor = np.zeros(shape=(NUM_SAMPLES, XBins, YBins, ZBins, 1))

    # Loop over all testing  data sets and add them to the tensor
    for e in range(NUM_SAMPLES):
        Event = dataset[e]
      # Set all the hit locations and energies
        for h in range(0, len(Event.X)):
            XBin = int( (Event.X[h] - XMin) / ((XMax - XMin) / XBins) )
            YBin = int( (Event.Y[h] - YMin) / ((YMax - YMin) / YBins) )
            ZBin = int( (Event.Z[h] - ZMin) / ((ZMax - ZMin) / ZBins) )
            #print("hit z bin: {} {}".format(Event.Z[h], ZBin))
            if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
                InputTensor[e][XBin][YBin][ZBin][0] = Event.E[h]


    Model = models.Sequential()
    Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(XBins, YBins, ZBins, 1)))
    Model.add(layers.MaxPooling3D((2, 2, 3)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    Model.add(layers.MaxPooling3D((2, 2, 2)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    Model.add(layers.Flatten())
    Model.add(layers.Dense(64, activation='relu'))
    Model.add(layers.Dense(OutputDataSpaceSize))

    Model.load_weights("results/model.ckpt")

    #visualize(Model, dataset, InputTensor)
    preds = Model.predict(InputTensor)
    for i in range(NUM_SAMPLES):
        #dataset[i].print()
        #print(preds[i])
        #show_pred(dataset[i],preds[i])# 
        save_pred_projections(dataset[i], preds[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualizer')
    parser.add_argument('--interactive', default=False, action='store_true', help='press n to step through data interactively')
    parser.add_argument('--debug', default=False, action='store_true', help='debug')
    parser.add_argument('--data_file', default='./data/RecoilElectrons.1k.data', type=str, help='serialized data location')
    parser.add_argument('--save_file', default=None, type=str, help='file save location')
    parser.add_argument('--num_save', default=None, type=int, help='when not interactive, how many images data instances to save')
    args = parser.parse_args()

    if args.debug:
        test()
    elif args.interactive:
        view_interactive(args)
    else:
        view_png_format(args)
