import os
import argparse
import re

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


def load_random(fold):
    rsvs = ['1.0', '0.95', '0.9', '0.85', '0.8', '0.75', '0.7', '0.65', '0.6', '0.55', '0.5', '0.45', '0.4', '0.35', '0.3', '0.25', '0.2']#, '0.15', '0.1', '0.09', '0.08', '0.07', '0.0625', '0.06', '0.0575', '0.055', '0.0525', '0.05', '0.0475', '0.045', '0.0425', '0.04', '0.0375', '0.035', '0.0325', '0.03']
    accs = []
    for para in rsvs:
        fn = os.path.join(fold, para+".out")
        with open(fn, 'r') as ips:
            for line in ips:
                mobj = re.findall("\d+\.\d+", line)
                if len(mobj) == 2:
                    acc = float(mobj[0])
            accs.append(acc)
    return np.asarray([float(elem) for elem in rsvs]), np.asarray(accs)


def load_al(fold):
    rsvs = []
    accs = []
    for fn in os.listdir(fold):
        if fn.endswith(".out"):
            params = re.findall("\d+\.\d+", fn)
            rsv, alpha = float(params[0]), float(params[1])
            if rsv == alpha:
                rsvs.append(rsv)
                with open(os.path.join(fold, fn), 'r') as ips:
                    for line in ips:
                        mobj = re.findall("\d+\.\d+", line)
                        if len(mobj) == 2:
                            acc = float(mobj[0])
                    accs.append(acc)
    results = sorted([(x, y) for x, y in zip(rsvs, accs)], key=lambda tp:tp[0])
    xs = [tp[0] for tp in results]
    ys = [tp[1] for tp in results]
    return np.asarray(xs), np.asarray(ys)


def main():
    parser = argparse.ArgumentParser(description='Plot Results')
    parser.add_argument('--folds', type=str, default='logs_s42')
    parser.add_argument('--legends', type=str, default='Random,EL2N,Mem,Infl-max,DDD,AGE,Ours')
    args = parser.parse_args()
    print(args)

    V = 1112392

    xs, ys, lgs = [], [], []
    for dirname in args.folds.split(','):
        for lg in args.legends.split(','):
            fn = lg.lower()
            #if fn == 'random':
            #    tp = load_random(os.path.join(args.fold, fn))
            #else:
            #    tp = load_al(os.path.join(args.fold, fn))
            tp = load_al(os.path.join(dirname, fn))
            if lg in lgs:
                idx = lgs.index(lg)
                assert len(ys[idx]) == len(tp[1]), "inconsistent len {}".format(len(tp[1]))
                ys[idx] += tp[1]
            else:
                lgs.append(lg)
                xs.append(tp[0])
                ys.append(tp[1])

    num_folds = len(args.folds.split(','))
    min_error_rate = 100
    method_id = -1
    ratio = -1
    for i in range(len(xs)):
        xs[i] = V * xs[i]
        ys[i] = 100.0 - (ys[i] / num_folds)
        idx = np.argmin(ys[i])
        if ys[i][idx] < min_error_rate and i != 0:
            min_error_rate = ys[i][idx]
            method_id = i
            ratio = xs[i][idx]
    print(lgs[method_id], ratio, min_error_rate)

    # first reach
    if args.legends != 'Random':
        std = 0.0139
        num_folds = len(args.folds.split(','))
        min_ratio_idx = 10
        crsp_error_rate = 100
        method_id = -1
        for i in range(1, len(xs)):
            for j in range(len(ys[i])):
                if ys[i][j] < ys[0][-1] + std:
                    break
            if j < min_ratio_idx or (j == min_ratio_idx and ys[i][j] < crsp_error_rate):
                method_id = i
                min_ratio_idx = j
                crsp_error_rate = ys[i][j]
        print(lgs[method_id], min_ratio_idx, crsp_error_rate)

    # tangent
    for i in range(len(xs)):
        if args.legends == 'Random':
            indX = np.log(np.expand_dims(xs[i], axis=1))
            depy = np.log(ys[i])

            reg = LinearRegression().fit(indX, depy)
            print(lgs[i], reg.score(indX, depy), reg.coef_)

    # plt.axvline(np.sqrt(N)/2)
    with sns.color_palette('viridis_r', len(xs)):
        #plt.figure(figsize=(6, 5))
        for i in range(len(xs)):
            plt.scatter(xs[i], ys[i], label=lgs[i])
            rtn = plt.plot(xs[i], ys[i])
            if i == 0:
                first_color = next(iter(rtn)).get_color()

            if args.legends == 'Random':
                #print(xs[i], (ys[i][-1] - 1e-6) + np.exp(reg.predict(indX)), ys[i])
                #print(xs[i], np.exp(reg.predict(indX)), ys[i])
                #plt.plot(xs[i], (ys[i][-1] - 1e-6) + np.exp(reg.predict(indX)), color='black')
                yhat = np.exp(reg.predict(indX))
                plt.plot(xs[i], yhat, color='black')
                plt.text(xs[i][-2], yhat[-2]+0.25, '$y=x^{-0.041}$', bbox=dict(facecolor='red', alpha=0.5))
            else:
                # plot first reach
                if i == method_id:
                    #seg = [xs[i][min_ratio_idx] - 0.025*V, xs[i][min_ratio_idx] + 0.025*V]
                    #plt.plot(seg, [ys[0][-1]+std, ys[0][-1]+std], color=first_color)
                    plt.plot([xs[i][min_ratio_idx] - 0.1*V, xs[0][-1]], [ys[0][-1] + std, ys[0][-1] + std], '--', color=first_color)
                    plt.plot([xs[i][min_ratio_idx] - 0.1*V, xs[0][-1]], [ys[0][-1], ys[0][-1]], '--', color=first_color)
                    #plt.errorbar([xs[i][min_ratio_idx]], [ys[0][-1]], yerr=std)
                    #plt.plot(seg, [ys[0][-1]-std, ys[0][-1]-std], color=first_color)


    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Error rate (%)')
    #plt.xlabel(r'$\alpha$')
    plt.xlabel('#Training examples')
    if args.legends != 'Random':
        plt.ylim(32.75, 35.75)
    plt.legend()
    if args.legends != 'Random':
        plt.yticks([33, 33.5, 34, 34.5, 35, 35.5], [33, 33.5, 34, 34.5, 35, 35.5])
    else:
        plt.yticks([33, 33.5, 34, 34.5, 35, 35.5], [33, 33.5, 34, 34.5, 35, 35.5])
    #plt.xticks([1,2,3],[1,2,3])
    #plt.yticks([2,10,20,50],[2,10,20,50])
    #plt.ylim([2,50])
    plt.grid(True,which='both',alpha=0.2)
    sns.despine()
    plt.savefig("cmp.pdf", transparent='True')


if __name__ == "__main__":
    main()
