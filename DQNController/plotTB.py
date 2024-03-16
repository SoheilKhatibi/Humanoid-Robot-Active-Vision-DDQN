import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import pandas as pd

def main():
    data = pd.read_csv("run-DQN_1-tag-episode_reward.csv")
    plt.style.use('seaborn')

    a = np.array(data['Step'])
    b = np.array(data['Value'])
    print(a)
    print(b)
    
    x_ = a

    # if (PlotIndex == 2):
    #     Color = 'r'
    #     Label = 'Success rate'
    # elif (PlotIndex == 3):
    #     Color = '#F26437'
    #     Label = 'Steps taken to succeed'
    # elif (PlotIndex == 4):
    #     Color = 'g'
    #     Label = 'Steps taken to miss the ball'


    

    Alpha = 0.2
    plt.figure(figsize=(16, 10), dpi = 100)
    
    Color = 'g'
    Label = 'episode reward'
    
    # Stdss = np.divide(Xs_stds, 2)
    # # Stdss = Xs_stds
    # x_Up = np.add(Xs_means, Stdss)
    # x_Down = np.subtract(Xs_means, Stdss)

    # # xnew = np.linspace(x_.min(), x_.max(), 500)
    # # splM = make_interp_spline(x_, Xs_means, k=3)  # type: BSpline
    # # splU = make_interp_spline(x_, x_Up, k=3)  # type: BSpline
    # # splD = make_interp_spline(x_, x_Down, k=3)  # type: BSpline
    
    # # Xs_means_smooth = splM(xnew)
    # # x_Up_smooth = splU(xnew)
    # # x_Down_smooth = splD(xnew)

    # # print(Xs_means_smooth)
    plt.plot(x_, b, color = Color, label = Label)
    # plt.plot(x_, x_Up, color = Color, alpha = 0.0)
    # plt.plot(x_, x_Down, color = Color, alpha = 0.0)
    # plt.fill_between(x_, x_Up, x_Down, color = Color, alpha = 0.2)
    # # plt.plot(xs, ks)
    # # plt.plot(xs, gs)
    # if (PlotIndex == 2):
    #     plt.axis([0, 100, 0, 1])
    # else:
    #     plt.axis([0, 100, 0, 22])
    
    # plt.axis([0, 30000, 0, 22])
    plt.xlabel('Time steps')
    plt.legend()
    plt.show()

if __name__== "__main__":
  main()
