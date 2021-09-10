from model_vae import MolecularVAE
from tools import *
from rdkit import Chem
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = MolecularVAE().to(device)
initial_checkpoint = './result/checkpoint/00012000_model.pth'
f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
state_dict = f['state_dict']
net.load_state_dict(state_dict, strict=True)

data_train, data_test, charset = load_dataset('./data/processed.h5')
charset = list(map(lambda x: x.decode("utf-8"), charset))
data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
train_loader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True)

valid_smi = []
for t, batch in enumerate(train_loader):
    batch_size = len(batch[0])
    batch = batch[0].to(device)
    output, mean, logvar = net(batch)
    inp = batch.cpu().numpy()
    outp = output.cpu().detach().numpy()
    lab = batch.cpu().numpy()
    sampled = outp[0].reshape(1, 120, len(charset)).argmax(axis=2)[0]
    smi = decode_smiles_from_indexes(sampled, charset)
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is not None:
        valid_smi.append(smi)
        print(smi)
with open("valid_smi.pkl", "wb") as fp:
    pickle.dump(valid_smi, fp)
