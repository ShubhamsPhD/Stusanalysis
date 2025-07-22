import mdtraj as md
import numpy as np

__all__ = ["calc_com", "calc_moi", "calc_director"]


def calc_com(coords, masses):
    """ Calculate the center of mass of a list of coordinates
    com = mean(masses * coords)

    Parameters:
    -----------
    coords : list
        list of coordinates
    masses : list
        list of masses corresponding to coordinates in coords.
        Must have the same length as coords

    Returns:
    --------
    com : list
        3 element list of COM coordinate
    """
    coords, masses = np.array(coords), np.array(masses)
    com = np.sum(coords * masses.T[np.newaxis, :], axis=1) / np.sum(masses)
    return com


def calc_moi(coords, masses):
    """ Calculate the moment of inertia tensor of a list of coordinates
          [Ixx, Ixy, Ixz]
    moi = [Iyx, Iyy, Iyz]
          [Izx, Izy, Izz]

    see http://www.kwon3d.com/theory/moi/iten.html for more.

    TODO: this could probably be done in a loop + vectorized

    Parameters:
    -----------
    coords : list
        list of coordinates
    masses : list
        list of masses corresponding to coordinates in coords.
        Must have the same length as coords

    Returns:
    --------
    moi : list
        3x3 array MOI tensor
    """

    I = np.zeros((3, 3))
    I[0, 0] = np.sum((coords[:, 1] ** 2 + coords[:, 2] ** 2) * masses)
    I[1, 1] = np.sum((coords[:, 0] ** 2 + coords[:, 2] ** 2) * masses)
    I[2, 2] = np.sum((coords[:, 0] ** 2 + coords[:, 1] ** 2) * masses)
    I[0, 1] = I[1, 0] = -1* np.sum((coords[:, 0] * coords[:, 1]) * masses)
    I[0, 2] = I[2, 0] = -1* np.sum((coords[:, 0] * coords[:, 2]) * masses)
    I[1, 2] = I[2, 1] = -1* np.sum((coords[:, 1] * coords[:, 2]) * masses)
    return I


def calc_director(moi):
    """ Calculate the director from a moment of inertia.
    The director is the dominant eigenvector of the MOI tensor

    Parameters:
    -----------
    moi : list
        3x3 array; MOItensor

    Returns:
    --------
    director : list
        3 element list of director vector
    """

    w, v = np.linalg.eig(moi)
    director = v[:, np.argmin(w)]
    return director
