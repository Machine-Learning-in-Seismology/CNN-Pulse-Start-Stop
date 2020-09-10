import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from MaLeNeuralNetworkFactory import MaLeNeuralNetworkFactory
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use('ggplot')
print("Loading starting and ending points...")
with open("startandend.pkl", "rb") as f:
    se = pickle.load(f, encoding='latin1')

with open("pulseshape_2.pkl", "rb") as f:
    data = pickle.load(f, encoding='latin1')

skiplist = [99, 246, 269, 336, 388, 401]
data = np.delete(data, skiplist, axis=0)
se = np.delete(se, skiplist, axis=0)
se = se * 20
results = pd.DataFrame(
    columns=["method", "fold", "mse", "mae", "r2", "mse_start", "mse_end", "mae_start", "mae_end", "r2_start",
             "r2_end"])
o = np.zeros((se.shape[0], se.shape[1] + 2))
o[:, :-2] = se


def evaluate(p, y):
    mse = mean_squared_error(y, p, multioutput='raw_values')
    mae = mean_absolute_error(y, p, multioutput='raw_values')
    r2 = r2_score(y, p, multioutput='raw_values')
    # cos = cosine_similarity(y, p)
    return mse, mae, r2


kf = KFold(n_splits=5)
fold = 0
timestr = time.strftime("%Y%m%d-%H%M%S")
verbose = False
for train, test in kf.split(data):
    test = test[se[test, :].min(axis=1) > 0]
    x_train = data[train, 0, :]
    x_test = data[test, 0, :]
    y_train = se[train, 0:2]
    y_test = se[test, 0:2]
    baker = se[test, 2:4]
    chang = se[test, 4:6]
    deniz = se[test, 6:8]

    network = MaLeNeuralNetworkFactory.get_network('MaLeConvSeq', x_train.shape, 100, 'adam')
    x_train2, x_test2 = network.input_scale(x_train, x_test)
    e = network.train(x_train2, y_train, 1000)
    score = network.evaluate(x_test2, y_test)
    # print("[CNN] MSE: %.2f MAE: %.2f MAPE: %.2f CP: %.2f" % (score[1], score[2], score[3], score[4]))
    output = network.predict(x_test2)
    o[test, 8] = output[:, 0]
    o[test, 9] = output[:, 1]
    mse, mae, r2 = evaluate(output, y_test)
    print("[CNN] MSE: %.2f MAE: %.2f R2: %.2f" % (np.mean(mse), np.mean(mae), np.mean(r2)))
    results.loc[len(results) + 1] = ["CNN", fold, np.mean(mse), np.mean(mae), np.mean(r2), mse[0], mse[1], mae[0],
                                     mae[1], r2[0], r2[1]]
    mse, mae, r2 = evaluate(baker, y_test)
    print("[BAKER] MSE: %.2f MAE: %.2f R2: %.2f" % (np.mean(mse), np.mean(mae), np.mean(r2)))
    results.loc[len(results) + 1] = ["BAKER", fold, np.mean(mse), np.mean(mae), np.mean(r2), mse[0], mse[1], mae[0],
                                     mae[1], r2[0], r2[1]]
    mse, mae, r2 = evaluate(chang, y_test)
    print("[CHANG] MSE: %.2f MAE: %.2f R2: %.2f" % (np.mean(mse), np.mean(mae), np.mean(r2)))
    results.loc[len(results) + 1] = ["CHANG", fold, np.mean(mse), np.mean(mae), np.mean(r2), mse[0], mse[1], mae[0],
                                     mae[1], r2[0], r2[1]]
    mse, mae, r2 = evaluate(deniz, y_test)
    print("[DENIZ] MSE: %.2f MAE: %.2f R2: %.2f" % (np.mean(mse), np.mean(mae), np.mean(r2)))
    results.loc[len(results) + 1] = ["DENIZ", fold, np.mean(mse), np.mean(mae), np.mean(r2), mse[0], mse[1], mae[0],
                                     mae[1], r2[0], r2[1]]
    results.to_csv("results-{:s}.csv".format(timestr), index=False)
    tmp = results.groupby(["method"]).mean().reset_index()
    tmp = tmp[["method", "mse", "mae", "r2", "r2_start", "r2_end"]]
    print(tmp)
    print("Plotting testing")

    j = 0
    for i in test:
        Path('fig/test/fold-' + str(fold) + '/').mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
        ax1.plot(data[i, 0, :])
        ax1.set_title('CNN')
        ax2.plot(data[i, 0, :])
        ax2.set_title('Real')
        ax3.plot(data[i, 0, :])
        ax3.set_title('Baker')
        ax4.plot(data[i, 0, :])
        ax4.set_title('Chang')
        ax5.plot(data[i, 0, :])
        ax5.set_title('Deniz')
        start = output[j, 0]
        end = output[j, 1]
        ax1.axvline(start, -2, 2, label='start', c="b")
        ax1.axvline(end, -2, 2, label='end', c="g")
        ax2.axvline(se[i, 0], -2, 2, label='real start', c="b")
        ax2.axvline(se[i, 1], -2, 2, label='real end', c="g")
        ax3.axvline(se[i, 2], -2, 2, label='start', c="b")
        ax3.axvline(se[i, 3], -2, 2, label='end', c="g")
        ax4.axvline(se[i, 4], -2, 2, label='start', c="b")
        ax4.axvline(se[i, 5], -2, 2, label='end', c="g")
        ax5.axvline(se[i, 6], -2, 2, label='start', c="b")
        ax5.axvline(se[i, 7], -2, 2, label='end', c="g")
        plt.savefig('fig/test/fold-' + str(fold) + '/pulse_' + str(i) + '.png')
        plt.clf()
        plt.cla()
        plt.close()
        j += 1

    if verbose:
        print("Plotting training")
        output = network.predict(x_train2)

        j = 0
        for i in train:
            Path('fig/train/fold-' + str(fold) + '/').mkdir(parents=True, exist_ok=True)
            fig, (ax1, ax2) = plt.subplots(2)
            ax1.plot(data[i, 0, :])
            ax1.set_title('CNN')
            ax2.plot(data[i, 0, :])
            ax2.set_title('Real')
            start = output[j, 0]
            end = output[j, 1]
            ax1.axvline(start, -2, 2, label='start', c="b")
            ax1.axvline(end, -2, 2, label='end', c="g")
            ax2.axvline(se[i, 0], -2, 2, label='real start', c="b")
            ax2.axvline(se[i, 1], -2, 2, label='real end', c="g")
            plt.savefig('fig/train/fold-' + str(fold) + '/pulse_' + str(i) + '.png')
            plt.clf()
            plt.cla()
            plt.close()
            j += 1
    fold += 1
    with open('out.pkl', 'wb') as handle:
        pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)
