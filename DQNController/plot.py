import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

PlotIndex = 4 # 2 for Success rate, 3 for Steps taken to succeed, 3 for Steps taken to miss the ball

def main():
    f=open("TrainingCourseOutput2.txt", "r")
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
        if (float(a[PlotIndex]) == -1):
            Xs.append(20)
        else:
            Xs.append(float(a[PlotIndex]))
        # gs.append(np.round(float(a[3]), 2))
        if (cnt % 100 == 0):
            Xs = np.array(Xs)
            Xs_means.append(np.round(Xs.mean(), 2))
            Xs_stds.append(np.round(Xs.std(), 2))
            # print(Xs, len(Xs), np.round(Xs.mean(), 2), np.round(Xs.std(), 2))
            # print("----------------------------------------------------------")
            cnt = 0
            Xs = []

    plt.style.use('seaborn')

    # print(Xs_stds, Xs_means, np.subtract(Xs_means, Xs_stds))
    x_ = np.arange(0, 101)

    if (PlotIndex == 2):
        Color = 'r'
        Label = 'Success rate'
    elif (PlotIndex == 3):
        Color = '#F26437'
        Label = 'Steps taken to succeed'
    elif (PlotIndex == 4):
        Color = 'g'
        Label = 'Steps taken to miss the ball'


    

    Alpha = 0.2
    plt.figure(figsize=(16, 10), dpi = 100)
    Stdss = np.divide(Xs_stds, 2)
    # Stdss = Xs_stds
    x_Up = np.add(Xs_means, Stdss)
    x_Down = np.subtract(Xs_means, Stdss)

    # xnew = np.linspace(x_.min(), x_.max(), 500)
    # splM = make_interp_spline(x_, Xs_means, k=3)  # type: BSpline
    # splU = make_interp_spline(x_, x_Up, k=3)  # type: BSpline
    # splD = make_interp_spline(x_, x_Down, k=3)  # type: BSpline
    
    # Xs_means_smooth = splM(xnew)
    # x_Up_smooth = splU(xnew)
    # x_Down_smooth = splD(xnew)

    # print(Xs_means_smooth)
    plt.plot(x_, Xs_means, color = Color, label = Label)
    plt.plot(x_, x_Up, color = Color, alpha = 0.0)
    plt.plot(x_, x_Down, color = Color, alpha = 0.0)
    plt.fill_between(x_, x_Up, x_Down, color = Color, alpha = 0.2)
    # plt.plot(xs, ks)
    # plt.plot(xs, gs)
    if (PlotIndex == 2):
        plt.axis([0, 100, 0, 1])
    else:
        plt.axis([0, 100, 0, 22])
    
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()

if __name__== "__main__":
  main()
