from torch.utils import data
from generator import *
from evaluation_metrics import *
from autoencoder import AutoEncoder
from torch.utils.data import Subset
from dataloader import EraiCpcDataset,EraiCpcWrfDataset
from networks_ import AE
import random
import os


from unet import UNet
from diffusion import Diffusion


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def weights_init_orthogonal(self, m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


epochs=100
lr=0.0001
l1_lambda=1
device='cuda:0'




#SR3 Setup

img_size = 40
LR_size = 40

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
schedule_opt = {'schedule':'linear', 'n_timestep':5000, 'linear_start':1e-4, 'linear_end':0.05}
loss_type='l1' 
save_path='./SR3.pt' 
load_path='./SR3.pt' 
load=True 
inner_channel=64 
norm_groups=32 
channel_mults=(1, 2, 2, 2) 
dropout=0.2
res_blocks=2
lr=1e-5 
distributed=False
in_channel = 2
out_channel = 1
model = UNet(in_channel, out_channel, inner_channel, norm_groups, channel_mults, res_blocks, dropout, img_size)
sr3 = Diffusion(model, device, img_size, LR_size, out_channel)

#sr3.apply(weights_init_orthogonal)
sr3.set_loss(loss_type)
sr3.set_new_noise_schedule(schedule_opt)


#for l1_lambda in [0.01,0.05,0.1,0.2,0.5,1,5,10]:
res_folder='results_full/'
reso=40
if not os.path.exists(res_folder):
    os.mkdir(res_folder)
for mode in ['SR3/']:
    for test_set in ['new_test', 'test']:
        if not os.path.exists(res_folder+mode):
            os.mkdir(res_folder+mode)
        style_dim=512
        if reso==40:
            test_data=EraiCpcDataset('./tensordata-precip-40',test_set)
        else:
            test_data=EraiCpcWrfDataset('./tensordata-precip-160',test_set)
        
        test_loader=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
        
        if  mode=='ours/':
            network=Generator(size1=reso,size2=reso,style_dim=style_dim,coord_size=4)
            network.load_state_dict(torch.load('_generator10.pt'))
        
        elif mode=='EAD/':
            network=Generator(size1=reso,size2=reso,style_dim=style_dim,coord_size=3)
            network.load_state_dict(torch.load('_generator_160_10_.pt'))
        
        elif mode=='naive/':
            network=AutoEncoder(size1=reso,size2=reso)
            network.load_state_dict(torch.load('_autoencoder_160_naive.pt'))
        
        elif mode=='AE/':
            network=AE(input_channels=1,num_layers=3,base_num=16)
            network.load_state_dict(torch.load('autoencoder_naive.pt')) 
        
        elif mode=='SR3/':
            network=sr3 
            network.load_state_dict(torch.load('diffusion_40.pt'))
            
            #network.load_state_dict(torch.load('diffusion__160_.pt'))
             
        
        
        network=network.to(device)
        network=network.eval()
        n = 0
        with torch.no_grad():
            for batch_num,data in enumerate(test_loader):
                hr_img,lr_img,prev_hr_img,coord,name=data['hr_img'].to(device),data['lr_img'].to(device),data['prev_hr_img'].to(device),data['coord'].to(device),data['name']
                if mode=='EAD/':
                    coord=coord[:,:3,:,:]
                noise=mixing_noise(1,style_dim,1,device)
                
                if mode=='ours/' or mode=='EAD/':
                    pred=network(coord,lr_img,prev_hr_img,noise)
                
                elif mode=='naive/' or mode=='AE/':
                    pred=network(lr_img)
                
                elif mode=='SR3/':
                    pred = network.super_resolution(lr_img) 
                
                torch.save(pred.detach(),res_folder+mode+name[0])
                n+= 1
                print(f"N -> {n}") 
                #if n == 25: 
                    #exit() 
                    

