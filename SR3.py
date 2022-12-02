import torch, torchvision
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from einops import rearrange, repeat
from tqdm.notebook import tqdm
from functools import partial
from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
import math, os, copy


from unet import UNet
from diffusion import Diffusion
from evaluation_metrics import *


class SR3():
    def __init__(self, device, img_size, LR_size, loss_type, dataloader, testloader, valloader,
                    schedule_opt, save_path, load_path=None, load=False, 
                    in_channel=2, out_channel=1, inner_channel=64, norm_groups=8, 
                    channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, lr=1e-5, distributed=False):
        
        super(SR3, self).__init__() 
        self.dataloader = dataloader
        self.testloader = testloader
        self.valloader = valloader
        self.device = device
        self.save_path = save_path
        self.img_size = img_size
        self.LR_size = LR_size

        self.max_loss = 1e10

        model = UNet(in_channel, out_channel, inner_channel, norm_groups, channel_mults, res_blocks, dropout, img_size)
        self.sr3 = Diffusion(model, device, img_size, LR_size, out_channel)

        # Apply weight initialization & set loss & set noise schedule
        self.sr3.apply(self.weights_init_orthogonal)
        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.sr3 = nn.DataParallel(self.sr3)

        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

        params = sum(p.numel() for p in self.sr3.parameters())
        print(f"Number of model parameters : {params}")

        if False: #load:
            self.load(load_path)

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

    def train(self, epoch, verbose):
        #fixed_imgs = copy.deepcopy(next(iter(self.testloader)))
        #fixed_imgs = fixed_imgs[0].to(self.device)
        # Transform to low-resolution images
        #fixed_imgs = transforms.Resize(self.img_size)(transforms.Resize(self.LR_size)(fixed_imgs))

        for i in tqdm(range(epoch)):
            train_loss = 0
            for _, imgs in enumerate(self.dataloader):
                self.sr3.train() 
                hr_img , lr_img = imgs['hr_img'].to(self.device) , imgs['lr_img'].to(self.device)

                #import pdb; pdb.set_trace()


                # Initial imgs are high-resolution
                #imgs = imgs[0].to(self.device)
                b, c, h, w = hr_img.shape
    
                self.optimizer.zero_grad()
                loss = self.sr3(hr_img, lr_img)
                loss = loss.sum() / int(b*c*h*w)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * b

            if (i+1) % verbose == 0:
                
                val_loss = self.validation(self.sr3, self.valloader)
                rmse_m,rmse_s,pc_m,pc_s=self.test(self.sr3,self.testloader)
                
                #best=[rmse_m,rmse_s,pc_m,pc_s]
                
                #print(epoch,loss.item(),total_loss,rmse_m,rmse_s,pc_m,pc_s)


                if val_loss < self.max_loss:
                    self.loss_max = val_loss
                    torch.save(self.sr3.state_dict(),'diffusion_'+'_160_'+'.pt')
                    
                    best = [rmse_m,rmse_s,pc_m,pc_s]

                #print(epoch,loss.item(),rmse_m,rmse_s,pc_m,pc_s)

                train_loss = train_loss / len(self.dataloader)
                
                print(f"Epoch: {i+1} loss: {train_loss: .3f} val_loss: {val_loss.item(): .3f} RMSE_M: {rmse_m}, RMSE_S: {rmse_s}, PC_M: {pc_m} , PC_S: {pc_s}")

                """ 
                self.sr3.eval()
                test_imgs = next(iter(self.testloader))
                test_imgs = test_imgs[0].to(self.device)
                b, c, h, w = test_imgs.shape

                with torch.no_grad():
                    val_loss = self.sr3(test_imgs)
                    val_loss = val_loss.sum() / int(b*c*h*w)
                self.sr3.train()

                train_loss = train_loss / len(self.dataloader)
                print(f'Epoch: {i+1} / loss:{train_loss:.3f} / val_loss:{val_loss.item():.3f}')
                """

        print(f"BEST {best}")


    def validation(self, network, val_loader):
        network.eval()
        n = 0
        val_loss = 0
        
        with torch.no_grad(): 
            for batch, data in enumerate(val_loader):
                hr_img,lr_img,prev_hr_img,coord=data['hr_img'].to(self.device),data['lr_img'].to(self.device),data['prev_hr_img'].to(self.device),data['coord'].to(self.device)
                #hr_img=hr_img.to(device)
                #lr_img=lr_img.to(device)
                b, c, h, w = hr_img.shape

                val_loss = network(hr_img, lr_img)
                val_loss = val_loss.sum() / int(b*c*h*w)
                n += 1      
            
            return val_loss / n    
            #print(f'Epoch: {i+1} / loss:{train_loss:.3f} / val_loss:{val_loss.item():.3f}')
    


    def test(self, network, dataloader):
        network.eval()
        rmse=[]
        pc=[]
        with torch.no_grad():
            for batch_num,data in enumerate(dataloader):
                hr_img,lr_img,prev_hr_img,coord=data['hr_img'].to(self.device),data['lr_img'].to(self.device),data['prev_hr_img'].to(self.device),data['coord'].to(self.device)
                #hr_img=hr_img.to(self.device)
                #lr_img=lr_img.to(self.device)
               

                #pred=network(hr_img, lr_img)
                pred = network.super_resolution(lr_img) 
                pred=pred.detach().cpu().numpy()
                
                hr_img=hr_img.detach().cpu().numpy()
                rmse.append(RMSE(pred,hr_img))
                pc.append(PCorrelation(pred,hr_img))
            return np.mean(rmse),np.std(rmse),np.mean(pc),np.std(pc)

    def save(self, save_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")