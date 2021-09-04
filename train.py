from tools import *
from model_vae import MolecularVAE
from file import Logger
import torch.utils.data
import torch.optim as optim
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    initial_checkpoint = None
    out_dir = "./result"
    for f in ['checkpoint']:
        os.makedirs(out_dir + '/' + f, exist_ok=True)
    num_iteration = 10000 * 1000
    iter_save = list(range(0, num_iteration, 1000))
    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')

    net = MolecularVAE().to(device)

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
    log.write('iter   epoch | loss  |\n')
    log.write('----------------------\n')
    #          0.00* 0.00  | 0.000  |

    optimizer = optim.Adam(net.parameters())

    def message(mode='print'):
        if mode == ('print'):
            asterisk = ' '
            loss = batch_loss
        if mode == ('log'):
            asterisk = '*' if iteration in iter_save else ' '
            loss = train_loss

        text = \
            '%5.4f%s %4.2f  | ' % (iteration / 10000, asterisk, epoch,) + \
            '%4.3f  ' % (loss)

        return text

    train_loss = 0
    batch_loss = 0
    sum_train_loss = 0
    sum_train = 0

    iteration = start_iteration
    epoch = start_epoch
    net.train()
    while iteration < num_iteration:
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
            if iteration % 100 == 0:
                # print('\r', end='', flush=True)
                # log.write(message(mode='log') + '\n')
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss = 0
                sum_train = 0
            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)


    # log.write('\n')


data_train, data_test, charset = load_dataset('./data/processed.h5')
charset = list(map(lambda x: x.decode("utf-8"), charset))
data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
train_loader = torch.utils.data.DataLoader(data_train, batch_size=250, shuffle=True)

torch.manual_seed(42)
train()

