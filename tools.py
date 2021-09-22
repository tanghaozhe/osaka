import torch
import torch.nn.functional as F
import torch.utils.data
import h5py
from rdkit import rdBase, Chem
import numpy as np
from matplotlib import colors
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import Descriptors, Fragments
from configure import *


class MolCalculation():
    def num_atoms(self, m):
        return m.GetNumAtoms()

    def tpsa(self, m):
        return Descriptors.TPSA(m)

    def mol_log(self, m):
        return Descriptors.MolLogP(m)

    def mol_weight(self, m):
        return Descriptors.MolWt(m)

    def num_aromatic_rings(self, m):
        return Descriptors.NumAromaticRings(m)

    def arom(self, m):
        return calc_AROM(m)


def radical_group_num_check(mol):
    for key in radical_group_num_restraint:
        func = 'Fragments.' + key
        max_num = radical_group_num_restraint[key]
        if 0 < max_num < eval(func)(mol):
            return False
    return True


def custom_patt_check(mol):
    for key in custom_pattern:
        max_num = custom_pattern[key]
        if max_num > 0:
            smi_patt = custom_pattern[key]
            total_num = 0
            patt = Chem.MolFromSmarts(smi_patt)
            if mol.HasSubstructMatch(patt):
                hit_ats = mol.GetSubstructMatches(patt)
                total_num += len(hit_ats)
            if total_num > max_num:
                return False
    return True


def calc_AROM(mh):
    m = Chem.RemoveHs(mh)
    ring_info = m.GetRingInfo()
    atoms_in_rings = ring_info.AtomRings()
    num_aromatic_ring = 0
    for ring in atoms_in_rings:
        aromatic_atom_in_ring = 0
        for atom_id in ring:
            atom = m.GetAtomWithIdx(atom_id)
            if atom.GetIsAromatic():
                aromatic_atom_in_ring += 1
        if aromatic_atom_in_ring == len(ring):
            num_aromatic_ring += 1
    return num_aromatic_ring


def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])


def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss


def get_image(mol, atomset, name):
    """Save image of the SMILES for vis purposes"""
    hcolor = colors.to_rgb('green')
    if atomset is not None:
        # highlight the atoms set while drawing the whole molecule.
        img = MolToImage(mol, size=(600, 600), fitImage=True, highlightAtoms=atomset, highlightColor=hcolor)
    else:
        img = MolToImage(mol, size=(400, 400), fitImage=True)

    img = img.save(name + ".jpg")
    return img