import numpy as np
from numpy.linalg import inv

#Store the resistance measurements "y"
y = np.array([[1068, 988, 1002, 996]]).T

#declare the standard diviation and variances of the measurements
SD_1 =  20
SD_2 =  2
var_1 = SD_1**2
var_2 = SD_2**2

# Define the R matrix - what does it contain?
R_Mea = np.zeros((4,4))
diagonal = [var_1, var_1, var_2, var_2]
np.fill_diagonal(R_Mea, diagonal)
R_Mea_inv = np.linalg.inv(R_Mea)

# Define the H matrix - what does it contain?
H = np.ones((4,1))
H_T = H.T
print (H_T)

# Now estimate the resistance parameter
R_WLS = np.linalg.inv(H_T @ R_Mea_inv @ H) @ H_T @ R_Mea_inv @ y

print('The slope parameter of the best-fit line (i.e., the resistance) is:')
print(R_WLS)



