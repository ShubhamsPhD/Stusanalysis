import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from  .molecules import *

def smoothing(y, window_size, order, deriv=0, rate=1):
    """
    Parameters
    ----------
    y : list
        data to be smoothed
    window_size : int
        window size for spline smoothing calc. Must positive and odd
    order : int
        polynomial order. Must be greater than window_size + 2
    deriv : int
        order of derivative. Must be less than or equal to order
    rate : float
        rate coefficient
    Returns
    -------
    y : list
        smoothed list
    """

    order_range = range(order+1)
    half_window = (window_size - 1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window,
                                                           half_window +1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

def calc_height(frame, atomselection, n_frames, window, n_layers, masses):
    atoms = frame.top.select(atomselection) # which atoms to plot
    box_length = np.mean([frame.unitcell_lengths[0,2]])
    hist, edges = np.histogram(frame.xyz[0, atoms, 2].reshape(-1), weights=np.tile(masses.take(atoms), n_frames),
                               range=[-.01,box_length+.01], bins=400)
    bins = (edges[1:]+edges[:-1]) / 2.0
    hist = smoothing(hist, window, order)
    hist = np.array([(hist[i], bins[i]) for i in range(len(hist))])
    hist.sort(reverse=True)
    peaks = []

    for i in range(n_layers):
        peaks.append(hist[0])
        hist = hist[np.abs( hist[:,0] - peaks[i][0] ) > 3.0]

    peaks = np.array(peaks)[:,1].sort()
    height = peaks[1:] - peaks[:-1]
    return height