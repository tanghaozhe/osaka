import torch
import torch.nn.functional as F
import torch.utils.data
import h5py
import numpy as np
from matplotlib import colors
from rdkit.Chem.Draw import MolToImage


def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])


def make_one_hot(data, tokenizer, max_len=120):
    """Converts the Strings to onehot data"""
    vocab = tokenizer.stoi
    data_one_hot = np.zeros((len(data), max_len, len(vocab)))
    for i, smiles in enumerate(data):
        smiles = tokenizer.text_to_sequence(smiles)
        smiles = smiles[:max_len] + [0] * (max_len - len(smiles))

        for j, sequence in enumerate(smiles):
            if sequence is not vocab['<UNK>']:
                data_one_hot[i, j, sequence] = 1
            else:
                data_one_hot[i, j, vocab['<UNK>']] = 1
    return data_one_hot


def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()


def load_dataset(filename, split = True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)


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