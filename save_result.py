import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import pandas as pd

def save_history(base_path):
    history_path = os.path.join(base_path, 'history.pkl')

    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    fig, ax = plt.subplots(1,2)

    loss = history['loss']
    acc = history['accuracy']

    x = np.arange(1, len(loss)+1)
    ax[0].plot(x, loss, label='loss')
    ax[1].plot(x, acc, label='acc')

    if 'val_loss' in history.keys() and 'val_accuracy' in history.keys():
        val_loss = history['val_loss']
        val_acc = history['val_accuracy']

        ax[0].plot(x, val_loss, label='val_loss')
        ax[1].plot(x, val_acc, label='val_acc')

    fig.legend()
    saved_path = os.path.join(base_path, 'loss.jpg')

    if not os.path.exists(saved_path):
        fig.savefig(saved_path)
    else:
        print('the path is exists:{}'.format(saved_path))

def save_predict(base_path):
    test_output_path = os.path.join(base_path, 'test_output.pkl')

    with open(test_output_path, 'rb') as f:
        test_output = pickle.load(f)

    for i, y in enumerate(test_output['y']):
        if y > 0.5:
            test_output['y'][i] = int(1)
        elif y < 0.5:
            test_output['y'][i] = 0
        else:
            pass

    for key in test_output.keys():
        test_output[key] = [i[0] for i in test_output[key]]

    saved_path = os.path.join(base_path, 'result.csv')
    if not os.path.exists(saved_path):
        df = pd.DataFrame(test_output)
        df.to_csv(saved_path, index=None)
    else:
        print('the path is exists:{}'.format(saved_path))

def save_history_and_predict(base_path):
    save_history(base_path)
    save_predict(base_path)

def main():
    base_path = 'data/models/jaad/24Jan2023-08h10m10s'
    save_history_and_predict(base_path)

if __name__ == "__main__":
    main()