import mdtraj as md
import numpy as np

# imports all python files
from analysis.angles import calc_angle
from analysis.directors import calc_com, calc_moi, calc_director
from analysis.s2 import calc_q, calc_s2

__all__ = ["calc_all_directors", "calc_tilt_angle", "calc_order_parameter"]


def calc_all_directors(xyz, masses, residues):
    """ Calculates directors for all residues in a frame. This is
    a wrapper for the calc_director function which only works for
    a single residue

    Parameters:
    -----------
    frame : mdtraj.Trajectory
        frame to analyze
    masses : list
        list of masses corresponding to each bead in the frame

    Returns:
    --------
    dict
        dict with a np.array of directors and coms
    """
    def la_region_worker(atoms):
        """ worker function for calculating a director. This allows for
        list comprehension

        Parameters:
        -----------
        atom_idxs : list
            list of indices corresponding to a the tail

        Returns:
        --------
        director : list
            returns a single director for a tail
        """

        res_coords = xyz[atoms]
        res_mass = masses.take([atoms])
        com = calc_com(res_coords, res_mass)
        centered_coords = res_coords - com
        moi = calc_moi(centered_coords, res_mass)
        director = calc_director(moi)
        return [director, com]

    masses = np.array(masses)
    la_region_idxs = [la_region for residue in residues for la_region in residue.la_regions]

    results = [la_region_worker(atom_indices) for atom_indices in la_region_idxs]

    directors = [result[0] for result in results]
    directors = np.array(directors)

    coms = [result[1] for result in results]
    coms = np.array(coms)
    return {"directors": directors,
            "coms": coms}


def calc_order_parameter(directors):
    Q = calc_q(directors)
    S2 = calc_s2(Q)
    return S2


def calc_tilt_angle(directors):
    vec1 = directors
    vec2 = [[0, 0, 1]] * len(vec1)
    tilt = calc_angle(vec1, vec2)
    return tilt
