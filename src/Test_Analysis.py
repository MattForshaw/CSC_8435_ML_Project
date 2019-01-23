# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 18:22:06 2019

@author: b8059092
"""

##### Analysis (ensure load_transform.py and CNN.py have been loaded first for new analyses)

# Load cached model
from keras.models import load_model
model = load_model(os.path.join(project_dir,"tests","cnn_"+test_n+".hdf5"))

# Load cached metadata df
hist_df = pd.read_pickle(os.path.join(project_dir,"cache","hist_df_"+test_n+".p"))

# Make sliced copies of hist_df for plotting
hist_df_acc = hist_df.drop({'train_loss','val_loss','lr'}, axis=1)
hist_df_loss = hist_df.drop({'train_acc','val_acc','lr'}, axis=1)

# Import ggplot (note - need to manually edit pandas lib import call in ggplot folder for loading to work in py3.6)
from ggplot import *

# Plot Training Accuracy
p1 = ggplot(pd.melt(hist_df_acc, id_vars=['epoch']), aes(x='epoch', y='value', color='variable')) +\
    geom_line() +\
    xlab("Epoch") + ylab("Accuracy (%)") + ggtitle("Training Accuracy")

# Save plot
ggplot.save(p1, filename=os.path.join(project_dir,"graphs",'Acc_'+test_n+'.pdf'))

# Plot Training Loss
p2 = ggplot(pd.melt(hist_df_loss, id_vars=['epoch']), aes(x='epoch', y='value', color='variable')) +\
    geom_line() +\
    xlab("Epoch") + ylab("Loss (%)") + ggtitle("Training Loss")

# Save plot
ggplot.save(p2, filename=os.path.join(project_dir,"graphs",'Loss_'+test_n+'.pdf'))


