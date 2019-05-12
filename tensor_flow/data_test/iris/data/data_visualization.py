import seaborn as sns;
sns.set(style="ticks", color_codes=True)

import matplotlib.pyplot as plt
iris = sns.load_dataset("iris")
g = sns.pairplot(iris)

plt.show()
