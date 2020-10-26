import shared
import numpy as np
import math

def __quaternion_to_rotation_matrix(q):
    a, b, c, d = q
    return np.array([[a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c],
                     [2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b],
                     [2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]])

def __rotate(yprSample, theta, rotation_axis=[0, 0, 1]):
    '''
    With an input matrix, rotate the object counter-clockwisely around the same rotation axis by the same angle at each time stamp

    Parameter:
        yprSample: input yall, pitch, roll matrix with dimension (N,3)
        rotation_axis: the axis around which the input should rotate.
                        It is an unit vector (x, y ,z)
        theta: the angle of rotation in degree

    Return:
        the new matrix after rotation, dimension (N,3)
    '''

    rotation_axis = np.array(rotation_axis)
    q = np.hstack([np.array(math.cos(np.radians(theta/2.0))),
                   rotation_axis * math.sin(np.radians(theta/2.0))])
    
    return np.matmul(__quaternion_to_rotation_matrix(q), yprSample.T).T

def __stretch(yprSample, ky, kp, kr):
    '''
    With an input matrix, stretch the object with constants ky, kp, kr

    Parameter:
        yprSample: input yall, pitch, roll matrix with dimension (N,3)
        k*: stretching constants

    Return:
        the new matrix after stretching, dimension (N,3)
    '''
    return yprSample*np.array([ky, kp, kr])

def __add_noise(yprSample, noise_mean=0.0, noise_std=1.0):
    '''
    With an input matrix, add noise~N(noise_mean,noise_std^2) to each entry

    Parameter:
        yprSample: input yall, pitch, roll matrix with dimension (N,3)
        noise_mean: the mean of the noise normal distribution, defaul 0
        noise_std: the standard deviation of the noise normal distribution, default 1

    Return:
        the new matrix with noise, dimension (N,3)
    '''
    for i in range(yprSample.shape[0]):
        eps = np.random.randn(3)*noise_std + noise_mean
        yprSample[i] += eps
    return yprSample

def augment(yprSample):
    '''
    yprSample shape=(N,3)
    '''

    theta = np.random.randn()*shared.THETA_RANGE
    aug_yprSample = __rotate(yprSample, theta)

    ky, kp, kr = np.random.randn(3)*0.3 + 1
    aug_yprSample = __stretch(aug_yprSample, ky, kp, kr)

    aug_yprSample = __add_noise(aug_yprSample)

    return aug_yprSample