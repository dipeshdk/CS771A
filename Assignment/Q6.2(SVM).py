# SVM for part-2
import os
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from mlxtend.plotting import plot_decision_regions

filepath = "./binclassv2.txt"

if(os.path.exists(filepath)):
    data = pd.read_csv(filepath, header=None)

X = np.array([[x,y] for x,y in zip(data[0],data[1])])
clf = svm.SVC(kernel='linear')
clf.fit(X, np.array(data[2]))
plot_decision_regions(X, np.array(data[2]), clf=clf, colors='blue,red')
plt.savefig("Q6.2(SVM)_plot.png")
plt.show()