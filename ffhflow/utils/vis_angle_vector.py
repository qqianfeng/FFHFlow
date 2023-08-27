import numpy as np
import matplotlib.pyplot as plt

angle_vec = np.load('/home/vm/workspace/FFHFlow/angle_vector.npy')

print(angle_vec[:,0].max())
print(angle_vec[:,0].min())

plt.figure()
plt.subplot(311)
plt.hist(angle_vec[:,0])
plt.subplot(312)
plt.hist(angle_vec[:,1])
plt.subplot(313)
plt.hist(angle_vec[:,2])
plt.show()