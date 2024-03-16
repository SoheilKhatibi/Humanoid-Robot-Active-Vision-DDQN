import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("run-DQN_1-tag-input_info_importance_weights.csv")
print(df)
plt.style.use('seaborn')
TSBOARD_SMOOTHING = 0.9

smooth = df.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

plt.plot(df['Step'], df["Value"], alpha=0.4, color = '#fc724b')
plt.plot(df['Step'], smooth["Value"], color = '#fc724b', label = 'Loss')
# plt.title("Tensorboard Smoothing = {}".format(TSBOARD_SMOOTHING))
# plt.grid(alpha=0.3)
plt.axis([0, 30000, 0, 0.12])
plt.legend()
plt.show()