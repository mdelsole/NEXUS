import numpy as np


horizontal = 10*[0.0] + 5*[1.0] + 10*[0.0]
vertical   = 5*[0.0, 0.0, 1.0, 0.0, 0.0]
leftdiag   = [1.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 1.0]
rightdiag  = [0.0, 0.0, 0.0, 0.0, 1.0,
               0.0, 0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 1.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0, 0.0,
               1.0, 0.0, 0.0, 0.0, 0.0]

horizontal_2d = np.reshape(horizontal,(-1,int(np.size(horizontal)**(1/2))))
vertical_2d = np.reshape(vertical,(-1,int(np.size(vertical)**(1/2))))
leftdiag_2d = np.reshape(leftdiag,(-1,int(np.size(leftdiag)**(1/2))))
rightdiag_2d = np.reshape(rightdiag,(-1,int(np.size(rightdiag)**(1/2))))


