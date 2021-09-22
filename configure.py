# http://rdkit.org/docs/source/rdkit.Chem.Fragments.html
vocab_len = 101
train_data_dir = "./data/chembl_500k_train.csv"
max_num_mol = 10


mol_calc_setting = {
    'num_atoms': True,
    'tpsa': True,
    'mol_log': False,
    'mol_weight': False,
    'num_aromatic_rings': True,
    'arom': False
}

# To disable this parameter, set the value to minus one
radical_group_num_restraint = {
    "fr_Al_COO": -1,  # Number of aliphatic carboxylic acids
    "fr_Ar_OH": -1,  # Number of aromatic hydroxyl groups
    "fr_nitro": 0,  # Number of nitro groups
}

# You can set arbitrary pattern to filter undesirable SMILES
custom_pattern = {
    "OH": -1,
}