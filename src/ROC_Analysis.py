# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 18:55:32 2019

@author: b8059092
"""

i##### ROC AUC Plot (ensure load_transform.py and CNN.py have been loaded first for new analyses)
# Source: https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Compute y_score (uncomment to use)
#y_score = model.predict(X_test)

# Cache y_score (uncomment to use)
#np.save(os.path.join(project_dir,"cache","y_score_"+test_n+".npy"), y_score)

# Load cached y_score
y_score = np.load(os.path.join(project_dir,"cache","y_score_"+test_n+".npy"))

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
#plt.plot(fpr["micro"], tpr["micro"],
         #label='micro-average (AUC = {0:0.2f})'
               #''.format(roc_auc["micro"]),
         #color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average (AUC = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'orange', 'blue','red','green','yellow','gray'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='{0} (AUC = {1:0.2f})'
             ''.format(class_list[i], roc_auc[i]))

# Removed 50-50 curve as relevance is questionable in multi-class context (MT)
#plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Classifier ROC AUC (One vs. All)')
plt.legend(loc="lower right")
#plt.show()
savefig(os.path.join(project_dir,"graphs",'ROC_'+test_n+'.pdf'))


