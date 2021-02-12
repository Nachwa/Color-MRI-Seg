import numpy as np

'''
planes:
0 = axial plane (from top of the head)
1 = coronal plane (from the front)
2 = sagittal plane (from the side)
'''


def MeanP(img, plane=0):
#mean intensity projection
    x = np.mean(img,axis=plane)
    return x
    
def MedianP(img, plane=0):
#median intensity projection
    x = np.median(img,axis=plane)

    return x
    
def MidRangeP(img, plane=0):
#mid range intensity projection
    x = (np.min(img,axis=plane)+np.max(img,axis=plane))/2.0

    return x
    
def MIP(img, plane=0):
# maximum intensity projection
    x = np.amax(img,axis=plane)

    return x
    
def StdP(img, plane=0):
# standard deviation intensity projection
    x = np.std(img,axis=plane)

    return x
    
def SsP(img, plane=0):
# sum of slices intensity projection
    x = np.sum(img,axis=plane)

    return x
