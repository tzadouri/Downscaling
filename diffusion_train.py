from torch.utils import data
from autoencoder import *
from evaluation_metrics import *
from torch.utils.data import Subset
from dataloader import EraiCpcDataset,EraiCpcWrfDataset
from networks_ import AE

from unet import UNet
from diffusion import Diffusion
from SR3 import SR3


epochs=100
lr=0.0001
#device='cuda:0'


def validate(network,dataloader,criterion):
    network.eval()
    n=0
    total_loss=0
    for batch_num,data in enumerate(dataloader):
        hr_img,lr_img,prev_hr_img,coord=data['hr_img'].to(device),data['lr_img'].to(device),data['prev_hr_img'].to(device),data['coord'].to(device)
        hr_img=hr_img.to(device)
        lr_img=lr_img.to(device)
        pred=network(lr_img)
        loss=criterion(pred,hr_img).mean()
        total_loss+=loss.item()
        n+=1
    return total_loss/n

def test(network,dataloader):
    network.eval()
    rmse=[]
    pc=[]
    for batch_num,data in enumerate(dataloader):
        hr_img,lr_img,prev_hr_img,coord=data['hr_img'].to(device),data['lr_img'].to(device),data['prev_hr_img'].to(device),data['coord'].to(device)
        hr_img=hr_img.to(device)
        lr_img=lr_img.to(device)
        pred=network(lr_img)
        pred=pred.detach().cpu().numpy()
        hr_img=hr_img.detach().cpu().numpy()
        rmse.append(RMSE(pred,hr_img))
        pc.append(PCorrelation(pred,hr_img))
    return np.mean(rmse),np.std(rmse),np.mean(pc),np.std(pc)




#train_data=EraiCpcDataset('./tensordata-precip-160','train')
#val_data=EraiCpcDataset('./tensordata-precip-160','val')
#test_data=EraiCpcDataset('./tensordata-precip-160','test')

train_data=EraiCpcWrfDataset('./tensordata-precip-160','train')
val_data=EraiCpcWrfDataset('./tensordata-precip-160','val')
test_data=EraiCpcWrfDataset('./tensordata-precip-160','test')


batch_size = 8

train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
val_loader=torch.utils.data.DataLoader(val_data,batch_size=batch_size,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)


#batch_size = 32
img_size = 160
LR_size = 160
cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda else "cpu")
schedule_opt = {'schedule':'linear', 'n_timestep':1000, 'linear_start':1e-4, 'linear_end':0.05}






sr3 = SR3(device, img_size=img_size, LR_size=LR_size, loss_type='l1', 
            dataloader=train_loader, testloader=test_loader,valloader=val_loader, schedule_opt=schedule_opt, 
            save_path='./SR3.pt', load_path='./SR3.pt', load=True, inner_channel=64, 
            norm_groups=32, channel_mults=(1, 2, 2, 2), dropout=0.2, res_blocks=2, lr=1e-4, distributed=False)

sr3.train(epoch=100, verbose=5)







"""
loss_max=10000000
for epoch in range(0,epochs):
    for batch,data in enumerate(train_loader):
        optimizer.zero_grad()
        network.train()
        hr_img,lr_img,prev_hr_img,coord=data['hr_img'].to(device),data['lr_img'].to(device),data['prev_hr_img'].to(device),data['coord'].to(device)
        hr_img=hr_img.to(device)
        lr_img=lr_img.to(device)
        pred=network(lr_img)
        loss=criterion(pred,hr_img).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        total_loss=validate(network,val_loader,criterion)
        rmse_m,rmse_s,pc_m,pc_s=test(network,test_loader)
        if total_loss<loss_max:
            loss_max=total_loss
            if dataset==40:
                torch.save(network.state_dict(),'autoencoder_'+mode+'.pt')
            else:
                torch.save(network.state_dict(),'_autoencoder_'+mode+'.pt')
            
            best=[rmse_m,rmse_s,pc_m,pc_s]
        print(epoch,loss.item(),total_loss,rmse_m,rmse_s,pc_m,pc_s)
print(mode,best)
"""







