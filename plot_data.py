

import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, Y, STD = None, image_loc = 'plot.png', title = '', x_label ='', x_best = None):
   # Set the Y limits
   min_Y = np.min(Y)
   max_Y = np.max(Y)
   diff_Y = max_Y - min_Y
   buffer_Y = diff_Y * 0.5

   plt.ylim(min_Y-buffer_Y, max_Y+buffer_Y)
   #plt.ylim(0.5, 1.01)
   plt.title(title)
   plt.ylabel('Score')
   plt.xlabel(x_label)

   if x_best is not None:
      plt.axvline(x=x_best, color='darkblue', linestyle='--')

   if STD is not None:
      plt.fill_between(X, np.add(Y, np.negative(STD)), np.add(Y, STD), alpha=0.1, color="red")

   plt.grid()
   plt.plot(X, Y, color='red', label='Score')
   plt.legend(loc='best')
   plt.savefig(image_loc)
   plt.close()