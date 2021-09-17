from model_vae import VAE
from tools import *
from rdkit import Chem
import pickle
from tokenizer import Tokenizer
import pandas as pd
from configure import *

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

valid_smi = []
for t, batch in enumerate(train_loader):
    batch_size = len(batch[0])
    batch = batch[0].to(device)
    output, mean, logvar = net(batch)
    inp = batch.cpu().numpy()
    print('input:', tokenizer.predict_caption(map(from_one_hot_array, inp[0])))
    outp = output.cpu().detach().numpy()
    sampled = outp[0].reshape(1, 120, len(tokenizer.itos)).argmax(axis=2)[0]
    smi = tokenizer.predict_caption(sampled)
    print("output:", smi)
    # smi = decode_smiles_from_indexes(sampled, tokenizer.itos)
    # m = Chem.MolFromSmiles(smi, sanitize=False)
    # if m is not None:
    #     valid_smi.append(smi)
    #     print(smi)
with open("valid_smi.pkl", "wb") as fp:
    pickle.dump(valid_smi, fp)
