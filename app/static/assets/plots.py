from pathlib import Path

import keras.utils.vis_utils
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import model_from_json


def plot_results_yolo(file='yolo*.csv', dir=''):
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob('yolo*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for fi, f in enumerate(files):
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)
                ax[i].set_title(s[j], fontsize=12)
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig('results_yolo.png', dpi=200)
    plt.close()


def plot_results_cnn():
    df = pd.read_csv('cnn_model_accuracy.csv')
    acc = [df['accuracy'], df['val_accuracy']]
    loss = [df['loss'], df['val_loss']]
    epoch = range(30)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch, acc[0], label='Training Accuracy')
    plt.plot(epoch, acc[1], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epoch, loss[0], label='Training Loss')
    plt.plot(epoch, loss[1], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('results_cnn.png')
    plt.close()


# from keras.utils import plot_model
def model_summary_cv2():
    json_file = open('../yolov5/roadsigns1.json', 'r')
    rs = json_file.read()
    json_file.close()
    loaded_model = model_from_json(rs)
    loaded_model.load_weights("../yolov5/roadsigns1.h5")
    # loaded_model.summary()
    keras.utils.vis_utils.plot_model(loaded_model, to_file='model.png')

def class_distribution():
    import os

    import pandas as pd
    import plotly.express as px

    c = {}
    pdir = '/Users/anirudhnegi/Downloads/Section 40 - Convolutional Neural Networks (CNN)/dataset/train/'
    for dir in os.listdir(pdir):
        if '.' not in dir:
            c[dir] = len(os.listdir(pdir + dir))
    df = pd.DataFrame(list(c.items()), columns=['Class', 'Count'])
    fig = px.bar(df, x='Class', y='Count')
    fig.write_image('class_dist.png')


plot_results_yolo()
plot_results_cnn()
model_summary_cv2()
class_distribution()
