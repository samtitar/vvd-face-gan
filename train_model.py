import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import os
import json
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import *

DATA_ROOT = 'data/vvders'
LOG_INTERVAL = 100
CPT_INTERVAL = 500

IMG_SIZE = 64
LATENT_SIZE = 100

BATCH_SIZE = 128
N_EPOCHS = 100000
LEARNING_RATE = 0.0002

def get_resume_id():
    try:
        with open('wandb/resume.json', 'r') as f:
            return json.load(f)['run_id']
    except:
        raise RuntimeError('No run to resume.')

def set_resume_id():
    with open('wandb/resume.json', 'w+') as f:
        json.dump({ 'run_id': run_id }, f)

def log_progress(epoch):
    wandb.log({ 'epoch': epoch,
                'd_loss': d_loss.item(),
                'g_loss': g_loss.item(),
                'D(G(x))': d_fake_mean,
                'D(x)': d_real_mean })

def create_grid(batch):
    grid = vutils.make_grid(batch[0], padding=2, normalize=True).numpy()
    return np.transpose(grid, (1,2,0))

def compose_state_dict(epoch):
    return { 'epoch': epoch,
             'noise': fixed_noise,
             'gen': gen.state_dict(),
             'dis': dis.state_dict(),
             'gen_o': gen_optimzer.state_dict(),
             'dis_o': dis_optimzer.state_dict() }

def save_checkpoint(epoch):
    # Save state dict in run dir
    state_dict = compose_state_dict(epoch)
    torch.save(state_dict, f'{run_dir}/{epoch}.pt')

    # Create symbolic link to latest checkpoint
    if os.path.exists(f'{run_dir}/latest.pt'):
        os.unlink(f'{run_dir}/latest.pt')
    os.symlink(f'{epoch}.pt', f'{run_dir}/latest.pt')

# Compose data transforms
norms = (0.5, 0.5, 0.5)
transform = transforms.Compose([ transforms.Resize(IMG_SIZE),
                                 transforms.CenterCrop(IMG_SIZE), 
                                 transforms.ToTensor(),
                                 transforms.Normalize(norms, norms) ])

# Initialize datloader
dataset = dset.ImageFolder(root=DATA_ROOT, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)

# Parse training options
parser = argparse.ArgumentParser(description='Start or resume GAN training.')
parser.add_argument('--resume', action='store_const', const=True, default=False)
args = parser.parse_args()

# Initialize models
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
gen = Generator(latent_size=LATENT_SIZE).to(device)
dis = Discriminator().to(device)

# Initialize loss & optimizers
criterion = nn.BCELoss()
gen_optimzer = optim.Adam(gen.parameters(), lr=LEARNING_RATE)
dis_optimzer = optim.Adam(dis.parameters(), lr=LEARNING_RATE)

# Load checkpoint if needed
if args.resume:
    run_id = get_resume_id()
    wandb.init(project='vvd-face-gan', resume='allow', id=run_id)
    run_dir = f'wandb/{run_id}'

    checkpoint = torch.load(f'{run_dir}/latest.pt')
    start_epoch = checkpoint['epoch']
    fixed_noise = checkpoint['noise']

    # Load models' states
    gen.load_state_dict(checkpoint['gen'])
    dis.load_state_dict(checkpoint['dis'])

    # Load optimizers' states
    gen_optimzer.load_state_dict(checkpoint['gen_o'])
    dis_optimzer.load_state_dict(checkpoint['dis_o'])
else:
    run_id = wandb.util.generate_id()
    wandb.init(project='vvd-face-gan', resume='allow', id=run_id)
    run_dir = f'wandb/{run_id}'
    set_resume_id()

    start_epoch = 1
    os.mkdir(run_dir)
    fixed_noise = torch.randn(64, LATENT_SIZE, 1, 1).to(device)
    save_checkpoint(0)    

# Start training from start epoch
for epoch in range(start_epoch, N_EPOCHS + 1):
    for i, (batch, _) in enumerate(dataloader):
        # Clear gradients
        gen_optimzer.zero_grad()
        dis_optimzer.zero_grad()

        # Get batch data and create target tensor
        batch = batch.to(device)
        b_size = batch.size(0)
        target = torch.full((b_size,), 1).float().to(device)

        # Forward real batch trough discriminator
        d_out = dis(batch).view(-1)
        d_loss_r = criterion(d_out, target)
        d_real_mean = d_out.mean().item()
        d_loss_r.backward()
        
        # Generate fake batch trough generator
        target.fill_(0)
        g_in = torch.randn(b_size, LATENT_SIZE, 1, 1).to(device)
        g_out = gen(g_in)

        # Forward fake batch trough discriminator
        d_out = dis(g_out.detach()).view(-1)
        d_loss_f = criterion(d_out, target)
        d_fake_mean = d_out.mean().item()

        # Optimze discriminator
        d_loss = d_loss_f + d_loss_r
        dis_optimzer.step()

        # Optimize generator
        target.fill_(1)
        d_out = dis(g_out).view(-1)
        g_loss = criterion(d_out, target)
        g_loss.backward()
        gen_optimzer.step()

        if epoch % LOG_INTERVAL == 0:
            log_progress(epoch)
        
        if epoch % CPT_INTERVAL == 0:
            save_checkpoint(epoch)
save_checkpoint(N_EPOCHS)