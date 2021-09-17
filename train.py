import pandas as pd
from tokenizer import Tokenizer
from tools import *
from model_vae import VAE
from file import Logger
import torch.utils.data
import torch.optim as optim
import os
import pickle
from configure import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    initial_checkpoint = None #"./result/checkpoint/00014000_model.pth"
    out_dir = "./result"
    for f in ['checkpoint']:
        os.makedirs(out_dir + '/' + f, exist_ok=True)
    num_iteration = 1000 * 1000
    iter_save = list(range(0, num_iteration, 500))
    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')

    net = VAE().to(device)

    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch = f['epoch']
        state_dict = f['state_dict']
        net.load_state_dict(state_dict, strict=True)
    else:
        start_iteration = 0
        start_epoch = 0

    log.write('** start training here! **\n')
    log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
    log.write('iter  | epoch | loss  \n')
    log.write('-----------------------\n')
    #               0  0.00  | 0.000

    optimizer = optim.Adam(net.parameters())

    def message(mode='print'):
        if mode == ('print'):
            asterisk = ' '
        if mode == ('log'):
            asterisk = '*' if iteration in iter_save else ' '

        text = \
            '%5d %s %4.2f  | ' % (iteration, asterisk, epoch,) + \
            '%4.3f  ' % (sum_train_loss / sum_train)

        return text

    iteration = start_iteration
    epoch = start_epoch
    net.train()
    while iteration < num_iteration:
        sum_train_loss = 0
        sum_train = 0
        for t, batch in enumerate(train_loader):
            if iteration in iter_save:
                if iteration != start_iteration:
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, out_dir + '/checkpoint/%08d_model.pth' % (iteration))
                    pass

            batch_size = len(batch[0])
            batch = batch[0].to(device)
            optimizer.zero_grad()
            output, mean, logvar = net(batch)
            loss = vae_loss(output, batch, mean, logvar)
            loss.backward()
            optimizer.step()

            epoch += 1 / len(train_loader)
            iteration += 1

            batch_loss = loss.item()
            sum_train_loss += batch_loss
            sum_train += batch_size
        print('\r', end='', flush=True)
        log.write(message(mode='log') + '\n')


torch.manual_seed(42)
data_train = pd.read_csv(train_data_dir)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

data_train = tokenizer.make_one_hot(data_train["SMILES"]).astype(np.float32)
data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
train_loader = torch.utils.data.DataLoader(data_train, batch_size=250, shuffle=True)
train()

