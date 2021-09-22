# https://future-chem.com/rdkit-aromatic-descriptor/#toc3


from model_vae import VAE
from tools import *
from rdkit import Chem
import pickle
from tokenizer import Tokenizer
import pandas as pd
from configure import *
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = VAE().to(device)
initial_checkpoint = './result/checkpoint/0100_model.pth'
f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
state_dict = f['state_dict']
net.load_state_dict(state_dict, strict=True)

data_train = pd.read_csv(train_data_dir)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

data_train = tokenizer.make_one_hot(data_train["SMILES"]).astype(np.float32)
data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
train_loader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True)


calc_list = list(filter(lambda key: mol_calc_setting[key], mol_calc_setting.keys()))
results = []

for t, batch in enumerate(train_loader):
    batch_size = len(batch[0])
    batch = batch[0].to(device)
    output, mean, logvar = net(batch)
    inp = batch.cpu().numpy()
    outp = output.cpu().detach().numpy()
    # print('input:', tokenizer.predict_caption(map(from_one_hot_array, inp[0])))
    # print("output:", tokenizer.predict_caption(outp[0].reshape(1, 120, len(tokenizer.itos)).argmax(axis=2)[0]))

    for i in range(len(outp)):
        sampled = outp[i].reshape(1, 120, len(tokenizer.itos)).argmax(axis=2)[0]
        smi = tokenizer.predict_caption(sampled)

        # check if the generated molecule is chemically reasonable
        m = Chem.MolFromSmiles(smi, sanitize=True)

        if m and radical_group_num_check(m) and custom_patt_check(m):
            mol_calc = MolCalculation()
            result = [smi]
            for key in calc_list:
                result.append(getattr(mol_calc, key)(m))
            results.append(result)


    if len(results) >= max_num_mol:
        break

results = pd.DataFrame(results, columns=['smiles']+calc_list).round(2)
print(results)
results.to_csv('./result/result.csv')
