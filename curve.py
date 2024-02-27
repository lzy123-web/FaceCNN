import h5py
import matplotlib.pyplot as plt
import numpy as np


def opencurve():
    file = h5py.File('./train_fig/VGGWithAttention.h5', 'r')
    train_loss = np.array(file['train_loss'])
    val_loss = np.array(file['val_loss'])

    val_acc = np.array(file['acc'])

    plt.figure(1)
    plt.plot(train_loss, label='train_loss')


    plt.plot(val_loss, label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')

    plt.figure(2)
    plt.plot(val_acc, label='acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    opencurve()
