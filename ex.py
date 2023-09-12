import numpy as np
def softmax2(Z):
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / np.sum(e_Z)
print(softmax2([1,3,5]))