import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

def main():
    f=open("ExperimentPassive.txt", "r")
    fl =f.readlines()
    Xs = []
    Xs_means = []
    Xs_stds = []
    # ys = []
    # ks = []
    # gs = []
    cnt = 0
    for x in fl:
        cnt += 1
        a = x.split(",")

        # print(a)
        # print(float(a[0]))
        # xs.append(float(a[0]))
        # ys.append(np.round(float(a[1]), 2))
        Xs.append(float(a[2]))
        # gs.append(np.round(float(a[3]), 2))
        if (cnt % 100 == 0):
            Xs = np.array(Xs)
            Xs_means.append(np.round(Xs.mean(), 2))
            Xs_stds.append(np.round(Xs.std(), 2))
            # print(Xs, len(Xs), np.round(Xs.mean(), 2), np.round(Xs.std(), 2))
            # print("----------------------------------------------------------")
            cnt = 0
            Xs = []

    print(Xs_stds, Xs_means, np.subtract(Xs_means, Xs_stds))
    x_ = np.arange(0, 2.01, 0.01)

    Color = 'g'

    plt.style.use('seaborn')



    plt.figure(figsize=(16, 10), dpi = 100)
    x_Up = np.add(Xs_means, np.divide(Xs_stds, 2))
    x_Down = np.subtract(Xs_means, np.divide(Xs_stds, 2))
    
    
    xnew = np.linspace(x_.min(), x_.max(), 250)
    
    splM = make_interp_spline(x_, Xs_means, k=3)  # type: BSpline
    splU = make_interp_spline(x_, x_Up, k=3)  # type: BSpline
    splD = make_interp_spline(x_, x_Down, k=3)  # type: BSpline
    
    Xs_means_smooth = splM(xnew)
    x_Up_smooth = splU(xnew)
    x_Down_smooth = splD(xnew)
    

    plt.plot(xnew, Xs_means_smooth, color = Color, label = 'Succes rate using entropy-minimizing method')
    plt.plot(xnew, x_Up_smooth, color = Color, alpha = 0.0)
    plt.plot(xnew, x_Down_smooth, color = Color, alpha = 0.0)
    plt.fill_between(xnew, x_Up_smooth, x_Down_smooth, color = Color, alpha = 0.2)
    
    f=open("ExperimentActive.txt", "r")
    fl =f.readlines()
    Xs = []
    Xs_means = []
    Xs_stds = []
    # ys = []
    # ks = []
    # gs = []
    cnt = 0
    for x in fl:
        cnt += 1
        a = x.split(",")

        # print(a)
        # print(float(a[0]))
        # xs.append(float(a[0]))
        # ys.append(np.round(float(a[1]), 2))
        Xs.append(float(a[2]))
        # gs.append(np.round(float(a[3]), 2))
        if (cnt % 100 == 0):
            Xs = np.array(Xs)
            Xs_means.append(np.round(Xs.mean(), 2))
            Xs_stds.append(np.round(Xs.std(), 2))
            # print(Xs, len(Xs), np.round(Xs.mean(), 2), np.round(Xs.std(), 2))
            # print("----------------------------------------------------------")
            cnt = 0
            Xs = []

    print(Xs_stds, Xs_means, np.subtract(Xs_means, Xs_stds))
    x_ = np.arange(0, 2.01, 0.01)

    Color = '#F26437'
    x_Up = np.add(Xs_means, np.divide(Xs_stds, 2))
    x_Down = np.subtract(Xs_means, np.divide(Xs_stds, 2))

    xnew = np.linspace(x_.min(), x_.max(), 500)
    
    splM = make_interp_spline(x_, Xs_means, k=3)  # type: BSpline
    splU = make_interp_spline(x_, x_Up, k=3)  # type: BSpline
    splD = make_interp_spline(x_, x_Down, k=3)  # type: BSpline
    
    Xs_means_smooth = splM(xnew)
    x_Up_smooth = splU(xnew)
    x_Down_smooth = splD(xnew)

    plt.plot(xnew, Xs_means_smooth, color = Color, label = 'Succes rate using proposed method')
    plt.plot(xnew, x_Up_smooth, color = Color, alpha = 0.0)
    plt.plot(xnew, x_Down_smooth, color = Color, alpha = 0.0)
    plt.fill_between(xnew, x_Up_smooth, x_Down_smooth, color = Color, alpha = 0.2)
    # plt.plot(xs, ks)
    # plt.plot(xs, gs)
    plt.axis([0, 2, 0, 1])
    plt.xlabel('Error')
    plt.legend()
    plt.show()

if __name__== "__main__":
  main()
