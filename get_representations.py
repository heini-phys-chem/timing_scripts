import qml
import qml2

from qml import Compound as Compound_old
from qml2 import Compound as Compound_new

from qml.representations import generate_fchl_acsf, generate_slatm, get_slatm_mbtypes
import numpy as np

from time import time

def get_mols(filename, version):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    data = dict()

    for line in lines:
        tokens = line.split()
        xyz_name = tokens[0]
        prop = float(tokens[1])
        data[xyz_name] = prop

    mols = []

    for xyz_file in sorted(data.keys()):
        if version == "old":
            mol = Compound_old()
            mol.read_xyz(xyz_file)
        elif version == "new":
            mol = Compound_new(xyz=xyz_file)

        mol.properties = data[xyz_file]
        mols.append(mol)

    return mols

def get_representations(mols, version, representation):
    if representation == "CM":
        for mol in mols:
            mol.generate_coulomb_matrix(size=23)


    if representation == "BoB":
        if version == "Old":
            bags = {
              "H":  max([mol.atomtypes.count("H" ) for mol in mols]),
              "C":  max([mol.atomtypes.count("C" ) for mol in mols]),
              "N":  max([mol.atomtypes.count("N" ) for mol in mols]),
              "O":  max([mol.atomtypes.count("O" ) for mol in mols]),
              "S":  max([mol.atomtypes.count("S" ) for mol in mols]),
            }
            for mol in mols:
                mol.generate_bob(asizse=23, bags=bags)
        elif version == "New":
            for mol in mols:
                mol.generate_bob(asize=23)


    if representation == "fchl19":
        if version == "old":
            for mol in mols:
                rep = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, pad=23)
                mol.representation = rep
        if version == "new":
            for mol in mols:
                mol.generate_fchl19()


    if representation == "SLATM":
        nuclear_charges = np.array([mol.nuclear_charges for mol in mols], dtype=object)
        mbtypes = get_slatm_mbtypes(nuclear_charges)

        for i, mol in enumerate(mols):
            mol.generate_slatm(mbtypes)


    return mols

def save_representations():
    return "Done"

def main():
    #filename = "hof_qm7.txt"
    filename = "hof_qm7_perturbed.txt"

    mols_old = get_mols(filename, "old")
    mols_new = get_mols(filename, "new")


    start = time()
    mols_old = get_representations(mols_old, "old", "CM")
    end = time()
    print(f"CM old: {(end-start)/60.}")

    start = time()
    mols_new = get_representations(mols_new, "new", "CM")
    end = time()
    print(f"CM new: {(end-start)/60.}")

    rep_old = np.array([mol.representation for mol in mols_old])
    rep_new = np.array([mol.representation for mol in mols_new])

    print(f"Rep diff: {np.mean(np.abs(rep_old-rep_new))}\n")



    start = time()
    mols_old = get_representations(mols_old, "old", "BoB")
    end = time()
    print(f"BoB old: {(end-start)/60.}")

    start = time()
    mols_new = get_representations(mols_new, "new", "BoB")
    end = time()
    print(f"BoB new: {(end-start)/60.}")

    rep_old = np.array([mol.representation for mol in mols_old])
    rep_new = np.array([mol.representation for mol in mols_new])

    print(f"Rep diff: {np.mean(np.abs(rep_old-rep_new))}\n")



    start = time()
    mols_old = get_representations(mols_old, "old", "fchl19")
    end = time()
    print(f"fchl19 old: {(end-start)/60.:.4f} sec")

    start = time()
    mols_new = get_representations(mols_new, "new", "fchl19")
    end = time()
    print(f"fchl19 new: {(end-start)/60.:.4f} sec")

    rep_old = np.array([mol.representation for mol in mols_old])
    rep_new = np.array([mol.representation for mol in mols_new], dtype=object)

    reps_new = []
    for rep in rep_new:
        rep_pad = np.zeros((23, 720))
        rep_pad[:rep.shape[0], :] += rep
        reps_new.append(rep_pad)
    reps_new = np.array(reps_new)

    print(f"Rep diff: {np.mean(np.abs(rep_old-reps_new))}\n")

    start = time()
    mols_old = get_representations(mols_old, "old", "SLATM")
    end = time()
    print(f"SLATM old: {(end-start)/60.:.4f} sec")

    start = time()
    mols_new = get_representations(mols_new, "new", "SLATM")
    end = time()
    print(f"SLATM new: {(end-start)/60.:.4f} sec")

    rep_old = np.array([mol.representation for mol in mols_old])
    rep_new = np.array([mol.representation for mol in mols_new])

    print(f"Rep diff: {np.mean(np.abs(rep_old-rep_new))}\n")





if __name__ == "__main__":
    main()
