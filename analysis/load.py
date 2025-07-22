import mdtraj as md
import numpy as np
from analysis.molecules.molecules import collect_molecules
from analysis.residue import Residue
from xml.etree import cElementTree as ET
import pickle

__all__ = [
    "get_cg_residuename",
    "get_standard_topology",
    "load_masses",
    "to_residuelist",
    "load_from_pickle",
    "load_from_trajectory",
    "extract_range",
]


def _is_lipid(resname, cg):
    """ Determine whether a residue name is an analyzeable lipid

    Parameters:
    -----------
    resname : string
        name of the residue to check

    Returns:
    --------
    boolean
        True if resnam is in molecule. False otherwise
    """
    if cg:
        molecule = collect_molecules("cg")
    else:
        molecule = collect_molecules("atomistic")

    return resname in molecule


def get_cg_residuename(residue):
    name = None
    atoms = [atom.name for atom in residue.atoms]
    if "chead" in atoms:
        name = "chol"
    elif "head" in atoms:
        if len(atoms) == 9:
            if "ter2" in atoms:
                name = "ffa24"
            else:
                name = "ffa25"
        elif len(atoms) == 8:
            if "ter2" in atoms:
                name = "ffa21"
            else:
                name = "ffa22"
        elif len(atoms) == 7:
            if "ter2" in atoms:
                name = "ffa18"
            else:
                name = "ffa19"
        elif len(atoms) == 6:
            if "ter2" in atoms:
                name = "ffa15"
            else:
                name = "ffa16"
    elif "oh4" in atoms and "oh3" in atoms:
        if "ter2" in atoms:
            name = "ucer6"
        else:
            name = "ecer6"
    elif "oh4" in atoms:
        if "ter2" in atoms:
            name = "ucer5"
        else:
            name = "ecer5"
    elif "oh3" in atoms:
        if "ter2" in atoms:
            name = "ucer3"
        else:
            name = "ecer3"
    elif "oh2" in atoms:
        if "ter2" in atoms:
            name = "ucer9"
        else:
            name = "ecer9"
    elif "water" in atoms:
        name = "water"
    if name == None:
        raise KeyError(
            "Residue containing {} ".format(atoms) + "is not in the database"
        )
    return name


def get_standard_topology(traj, cg):
    """ Load system information into a trajectory.
    CG systems: Based on the number and name of CG beads in a residue,
    load in the correct residue name

    AA systems: Validate that all residue names are in the molecule
    list.

    Parameters:
    -----------
    traj : md.Trajectory
        Trajectory in which system information is loaded.
    cg :  boolean
        True if the system is coarse-grained. False for atomistic

    Returns:
    --------
    traj : md.Trajectory
        Trajectory with information loaded.
    """

    if cg:
        for i, residue in enumerate(traj.top.residues):
            name = get_cg_residuename(residue)
            traj.top.residue(i).name = name
    else:
        for res in traj.top.residues:
            if not _is_lipid(res.name, cg):
                raise KeyError(
                    "Residue {} ".format(res.name)
                    + "is not in the molecules database "
                    + "See analysis/molecules.py"
                )

    return traj


def load_masses(cg, topology=None, topfile=None):
    if cg:
        if topfile == None:
            raise ValueError("topfile is a required arguement " + "if cg=True")
        print(topfile)
        tree = ET.parse(topfile)
        root = tree.getroot()
        masses = np.fromstring(root[0].find("mass").text, sep="\n")
    else:
        if topology == None:
            raise ValueError("topology is a required arguement " + "if cg=True")
        masses = []
        for atom in topology.atoms:
            if atom.element.symbol == "C":
                masses.append(12.0107)
            elif atom.element.symbol == "H":
                masses.append(1.00794)
            elif atom.element.symbol == "N":
                masses.append(14.0067)
            elif atom.element.symbol == "O":
                masses.append(15.999)
    return np.array(masses)


def to_residuelist(topology, cg):
    """ Convert a topology into a list of Residue objects.
    Residue objects are intended to reduce memory consumption by
    eliminating unnecessary information.

    Parameters:
    -----------
    topology : md.Topology
        Topology to be convert

    Returns:
    --------
    residuelist : list
        list of Residue objects
    """
    if cg:
        molecule = collect_molecules("cg")
    else:
        molecule = collect_molecules("atomistic")
    assert type(topology) == md.Topology
    residuelist = []
    for residue in topology.residues:
        if not _is_lipid(residue.name, cg=cg):
            continue
        res_idx = topology.select("resid {}".format(residue.index))
        tails = []
        for tail_idx in molecule[residue.name].tails:
            tails.append(np.array(res_idx).take(tail_idx))
        la_regions = []
        if len(molecule[residue.name].la_regions) == 0:
            la_regions = tails.copy()
            #this print statement can probably be eventually removed once we are sure all
            #molecule files have been updated appropriately
            print('la_regions not defined, copying indeces from tails for molecule ', residue.name)
        else:
            for lar_idx in molecule[residue.name].la_regions:
                la_regions.append(np.array(res_idx).take(lar_idx))

        new_residue = Residue(name=residue.name, tails=tails,la_regions=la_regions)
        residuelist.append(new_residue)
    return residuelist


def load_from_pickle(filename):
    with open(filename, "rb") as f:
        frames = pickle.load(f)
    print("Loading trajectory from {}".format(filename))
    return frames


def load_from_trajectory(trajfile, topfile=None):
    if topfile:
        traj = md.load(trajfile, top=topfile)[-300:]
        print(
            "Loading trajectory from {} ".format(trajfile)
            + "and topology from {}".format(topfile)
        )
    else:
        traj = md.load(trajfile)
        print("Loading trajectory from {}".format(trajfile))
    return traj


def extract_range(traj, masses, cg, z_min=None, z_max=None):
    if cg:
        molecule = collect_molecules("cg")
    else:
        molecule = collect_molecules("atomistic")
    assert z_min or z_max
    if z_min and z_max:
        fxn = lambda res: res.atom(molecule[res.name].head).index
        sel_atoms = [
            atom.index
            for residue in traj.top.residues
            for atom in residue.atoms
            if residue.name in molecule
            and z_min < np.mean(traj.xyz[:, fxn(residue), 2]) < z_max
        ]
        sel_atoms = np.array(sel_atoms)
        traj.atom_slice(sel_atoms, inplace=True)
        masses = masses.take(sel_atoms)
    elif z_min:
        fxn = lambda res: res.atom(molecule[res.name].head).index
        sel_atoms = [
            atom.index
            for residue in traj.top.residues
            for atom in residue.atoms
            if residue.name in molecule
            and z_min < np.mean(traj.xyz[:, fxn(residue), 2])
        ]
        sel_atoms = np.array(sel_atoms)
        traj.atom_slice(sel_atoms, inplace=True)
        masses = masses.take(sel_atoms)
    elif z_max:
        fxn = lambda res: res.atom(molecule[res.name].head).index
        sel_atoms = [
            atom.index
            for residue in traj.top.residues
            for atom in residue.atoms
            if residue.name in molecule
            and np.mean(traj.xyz[:, fxn(residue), 2]) < z_max
        ]
        sel_atoms = np.array(sel_atoms)
        traj.atom_slice(sel_atoms, inplace=True)
        masses = masses.take(sel_atoms)

    return traj, masses
