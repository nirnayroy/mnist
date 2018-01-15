import numpy as np
import cv2



def convo(img, window_size, step_size):
    img = np.array(img)
    conv_feat = np.array([])
    m = 0
    for i in range(32):
        for j in range(32):
            conv_feat[:,m]=np.reshape(img[i:i+window_size,j:j+window_size],(-1))
            m += 1
            j += step_size
        i += step_size
    return conv_feat, m
    
img = cv2.imread("16.png",0)
_, m = convo(img,8,4)
print(m)
