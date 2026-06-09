import time
import matplotlib.pyplot as plt 
import os
os.chdir('../')
import torch
import numpy as np
import torch.nn as nn
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import random 
import psutil
from decimal import Decimal
import deepwave
import warnings
import torch.nn.functional as F
from functools import partial
from torchaudio.functional import biquad
from scipy.signal import butter
warnings.filterwarnings('ignore')
import os
from ultils.utils import *
from pyramid_loss import gaussian_kernel, create_laplacian_pyramid
os.environ['KMP_DUPLICATE_LIB_OK']='True'

gpu_count = torch.cuda.device_count()
print(f"The number of available GPUs is: {gpu_count}")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print(f"The selected GPU device is: {torch.cuda.get_device_name(DEVICE)}")
else:
    DEVICE = torch.device("cpu")
    print("No available GPUs detected, switched to using CPU")
    
######### load data #########
dx = 10

vp_true = torch.tensor(np.load('./marmousi_model/vp_truex352x1150x10.npy'))
vs_true = torch.tensor(np.load('./marmousi_model/vs_truex352x1150x10.npy'))
rho_true = torch.tensor(np.load('./marmousi_model/rho_truex352x1150x10.npy'))


submarine_deep = 43
submarine_vp = 1500                                      #water layer vp
submarine_vs = 800  ##to obay the CFL condition                                         #water layer vs
submarine_rho = 1009
  
vp_true[:submarine_deep,:] = submarine_vp
vs_true[:submarine_deep,:] = submarine_vs
rho_true[:submarine_deep,:] = submarine_rho


import scipy.ndimage
import scipy.io

sigma = 30

times = 6
if times != 0:
    vp_initial,vs_initial,rho_initial = gaussian_smooth_n_times(vp_true, vs_true, rho_true, sigma, times, DEVICE)
else:
    vp_initial,vs_initial,rho_initial = gaussian_smooth_once(vp_true, vs_true, rho_true, sigma, DEVICE)

snr_vp = ComputeSNR(vp_initial.detach().cpu().numpy(), \
                  vp_true.detach().cpu().numpy())

print(snr_vp)
print(vp_initial.max(),vp_initial.min())
vp_initial[:submarine_deep,:] = submarine_vp
vs_initial[:submarine_deep,:] = submarine_vs
rho_initial[:submarine_deep,:] = submarine_rho


loss_fn = 'l1'  

if times != 0:
    ##### save path
    vp_save_path = f'./rec/reconstruction_ms/EFWI_lap_en/init_2/vp/'
    vs_save_path = f'./rec/reconstruction_ms/EFWI_lap_en/init_2/vs/'
    rho_save_path = f'./rec/reconstruction_ms/EFWI_lap_en/init_2/rho/'
    main_path = f'./log_data/log_data_ms/EFWI_lap_en/init_2/'
else:
    vp_save_path = f'./rec/reconstruction_ms/EFWI_lap_en/init_1/vp/'
    vs_save_path = f'./rec/reconstruction_ms/EFWI_lap_en/init_1/vs/'
    rho_save_path = f'./rec/reconstruction_ms/EFWI_lap_en/init_1/rho/'
    main_path = f'./log_data/log_data_ms/EFWI_lap_en/init_1/'


if not os.path.exists(main_path):
    os.makedirs(main_path)

if not os.path.exists(vp_save_path):
    os.makedirs(vp_save_path)
if not os.path.exists(vs_save_path):
    os.makedirs(vs_save_path)
if not os.path.exists(rho_save_path):
    os.makedirs(rho_save_path)
    
Physics = Physics_deepwave                              
                           
model_shape = [vp_true.shape[0], vp_true.shape[1]]                                 
                                                                                       
DT = 0.006                                               
F_PEAK = 8                                              
DH = dx                                                  

inpa = {                                
    'fdom': F_PEAK, 
    'dh': DH,   
    'dt': DT
}

NT = 2500

### load obs and src, src_loc, rec_loc
obs_file_vx = './obs_data/d_obs_vx_src.npy'
obs_file_vy = './obs_data/d_obs_vy_src.npy'
d_obs_vx = torch.tensor(np.load(obs_file_vx))
d_obs_vy = torch.tensor(np.load(obs_file_vy))

src_loc = torch.tensor(np.load('./src_rec_positions/src_loc.npy'))
rec_loc = torch.tensor(np.load('./src_rec_positions/rec_loc.npy'))
src     = torch.tensor(np.load('./sources/src.npy'))

### sum source
num_super_sources = 10
sources_per_super = 10
source_distribution = 'random'# ('random' 或 'uniform')

#### 1.
result = encode_sources(obs_data_vx = d_obs_vx.squeeze(0),
                        obs_data_vy = d_obs_vy.squeeze(0),
                        source_locations = src_loc, 
                        receiver_locations  = rec_loc, 
                        source_functions = src,
                        num_super_sources = num_super_sources,
                        sources_per_super = sources_per_super,
                        source_distribution=source_distribution
                        )

#### 2
print(f"Encoded observation data shape (vx): {result['encoded_obs_data_vx'].shape}")
print(f"Encoded observation data shape (vy): {result['encoded_obs_data_vy'].shape}")
print(f"Encoded source location shape: {result['encoded_source_locations'].shape}")
print(f"Encoded receiver location shape: {result['encoded_receiver_locations'].shape}")
print(f"Encoded source function shape: {result['encoded_source_functions'].shape}")


d_obs_vx_encode = result['encoded_obs_data_vx'].unsqueeze(0).to(DEVICE)
d_obs_vy_encode = result['encoded_obs_data_vy'].unsqueeze(0).to(DEVICE)
src_loc_encode  = result['encoded_source_locations'].to(DEVICE)
rec_loc_encode  = result['encoded_receiver_locations'].to(DEVICE)
src_encode      = result['encoded_source_functions'].to(DEVICE)

vp_initial = vp_initial.to(DEVICE)
vs_initial = vs_initial.to(DEVICE)
rho_initial = rho_initial.to(DEVICE)
vp = vp_initial.requires_grad_(True)
vs = vs_initial.requires_grad_(True)
rho = rho_initial.requires_grad_(True)

criteria = torch.nn.L1Loss(reduction='sum')
criteria_model = torch.nn.L1Loss(reduction='sum')

optimer = torch.optim.Adam([{'params': [vp], 'lr': 6.0},
                            {'params': [vs], 'lr': 4.0},
                            {'params': [rho], 'lr': 2.0}])


all_loss_data = []
all_loss_vx_model = []
all_loss_vy_model = []
all_loss_rho_model = []
all_loss_model =[]
SNR_vp = []
SSIM_vp = []
Loss_vp = []
ERROR_vp = []
SNR_vs = []
SSIM_vs = []
Loss_vs = []
ERROR_vs = []
SNR_rho = []
SSIM_rho = []
Loss_rho = []
ERROR_rho = []
time_each_iter = []
def get_cpu_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2  
    
def get_gpu_memory():
    if torch.cuda.is_available():
        
        return torch.cuda.memory_allocated() / 1024 ** 2
    else:
        return 0
cpu_mem_log = [] 
gpu_mem_log = []

# Run optimisation/inversion

# import time
t_start = time.time()
mini_batches = 5
ITERATION = 50 


levels  = 6

levels_1 = np.arange(0,levels,1)

for level in reversed(levels_1):
    print('Now inversion on level:',level)

    for iter in range(ITERATION):
        loss_data_minibatch = []
        
        time_each_bath_start = time.time()
        for batch in range(mini_batches):
       
            optimer.zero_grad()
        
            src_loc_batch = src_loc_encode[batch::mini_batches].to(DEVICE)
            rec_loc_batch = rec_loc_encode[batch::mini_batches].to(DEVICE)
            src_batch = src_encode[batch::mini_batches].to(DEVICE) ## for true source
                
            physics = Physics(inpa['dh'], inpa['dt'],inpa['fdom'] ,size=NT,src=src_batch,
                        src_loc=src_loc_batch, rec_loc=rec_loc_batch
                        )

            with torch.no_grad():
            
                vp[:submarine_deep,:] = submarine_vp
                vs[:submarine_deep,:] = submarine_vs
                rho[:submarine_deep,:] = submarine_rho
            
                vp[vp>vp_true.max()] = vp_true.max()
                vp[vp<vp_true.min()] = vp_true.min()
        
                vs[vs>vs_true.max()] = vs_true.max()
                vs[vs<vs_true.min()] = vs_true.min()
        
                rho[rho>rho_true.max()] = rho_true.max()
                rho[rho<rho_true.min()] = rho_true.min()
               
                vp = vp.requires_grad_(True)
                vs = vs.requires_grad_(True)
                rho = rho.requires_grad_(True)          
        

            vp = vp.to(DEVICE)
            vs = vs.to(DEVICE)
            rho = rho.to(DEVICE)
            
    
            taux_est = physics(vp,vs,rho) 
            taux_vx_est_filtered = taux_est[0].to(DEVICE)
            taux_vy_est_filtered = taux_est[1].to(DEVICE)
            
            
            kernel_size = 5
            channels = taux_vx_est_filtered.shape[1]
            sigma = 3
            dtype = torch.float
            kernel = gaussian_kernel(size=kernel_size, channels=channels, sigma=sigma, dtype=dtype, device=DEVICE)
            
            
            pyramids_taux_vx_est_filtered = create_laplacian_pyramid(taux_vx_est_filtered, kernel=kernel, levels=levels
                                                        )
            pyramids_taux_vy_est_filtered = create_laplacian_pyramid(taux_vy_est_filtered, kernel=kernel, levels=levels
                                                        )
            
            taux_est_all = torch.cat((pyramids_taux_vx_est_filtered[level],pyramids_taux_vy_est_filtered[level]),dim=1).to(DEVICE)
            #print(taux_est_all.shape)
            #print(taux_est_all)
            d_obs_vx_filtered = d_obs_vx_encode[:, batch::mini_batches].to(DEVICE)
            d_obs_vy_filtered = d_obs_vy_encode[:, batch::mini_batches].to(DEVICE)
            
            pyramids_d_obs_vx_filtered = create_gaussian_pyramid(d_obs_vx_filtered, kernel=kernel, levels=levels
                                                        )
            pyramids_d_obs_vy_filtered = create_gaussian_pyramid(d_obs_vy_filtered, kernel=kernel, levels=levels
                                                        )
            
            d_obs_filtered_all = torch.cat((pyramids_d_obs_vx_filtered[level],pyramids_d_obs_vy_filtered[level]),dim=1).to(DEVICE)


            if loss_fn == 'l1':
                criteria = torch.nn.L1Loss(reduction='mean')
                loss_data = 1.0e10*criteria(taux_est_all, d_obs_filtered_all)
            if loss_fn == 'l2':
                criteria = torch.nn.MSELoss(reduction='mean')
                loss_data = 1.0e12*criteria(taux_est_all, d_obs_filtered_all)
            loss = loss_data
        
            loss.backward()
        
            optimer.step()
        
     
        all_loss_data.append(loss_data.detach().cpu().item())
        time_each_bath_end = time.time()
        time_each_iter.append(time_each_bath_end - time_each_bath_start)
        
        mem_now_cpu = get_cpu_memory()
        cpu_mem_log.append(mem_now_cpu)
        mem_now_gpu = get_gpu_memory()
        gpu_mem_log.append(mem_now_gpu)
        
        with torch.no_grad():
            all_loss_vx_model.append(
                criteria_model(vp.cpu(),vp_true.cpu()).detach().numpy().item()
            )
        
            all_loss_vy_model.append(
                criteria_model(vs.cpu(),vs_true.cpu()).detach().numpy().item()
                )
            all_loss_rho_model.append(
                criteria_model(rho.cpu(),rho_true.cpu()).detach().numpy().item()
            )
            all_loss_model.append(
                criteria_model(vp.cpu(),vp_true.cpu()).detach().numpy().item()+ \
                criteria_model(vp.cpu(),vp_true.cpu()).detach().numpy().item()+ \
                criteria_model(rho.cpu(),rho_true.cpu()).detach().numpy().item()
            )
    
        snr_vp = ComputeSNR(vp.detach().cpu().numpy(), \
                  vp_true.detach().cpu().numpy())
        SNR_vp = np.append(SNR_vp, snr_vp)
        snr_vs = ComputeSNR(vs.detach().cpu().numpy(), \
                  vs_true.detach().cpu().numpy())
        SNR_vs = np.append(SNR_vs, snr_vs)
        snr_rho = ComputeSNR(rho.detach().cpu().numpy(), \
                  rho_true.detach().cpu().numpy())
        SNR_rho = np.append(SNR_rho, snr_rho)


        if (iter+1)%5 == 0:
            print(f"Iteration {iter + 1} = loss: {all_loss_data[-1]:.4f},model loss: {all_loss_model[-1]:.4f},time:{time_each_iter[-1]:.2f},snr_vp:{SNR_vp[-1]:.3f},snr_vs:{SNR_vs[-1]:.3f},snr_rho:{SNR_rho[-1]:.3f}")

        if (iter+1)%10==0:
            np.save(vp_save_path + 'recx_iter_%s_%s.npy' % (iter + 1, level), vp.cpu().detach().numpy())
            np.save(vs_save_path + 'recx_iter_%s_%s.npy' % (iter + 1, level), vs.cpu().detach().numpy())
            np.save(rho_save_path + 'recx_iter_%s_%s.npy' % (iter + 1, level), rho.cpu().detach().numpy())
    
t_end = time.time()
elapsed_time = t_end - t_start
print('Running complete in {:.0f}m  {:.0f}s' .format(elapsed_time //60 , elapsed_time % 60))


#print(all_loss_data)
### path for log data

with torch.no_grad():
    
    np.savetxt(main_path+'all_loss_data.txt', all_loss_data,delimiter=',')
    np.savetxt(main_path+'all_loss_model.txt', all_loss_model, delimiter=',')
    np.savetxt(main_path+'all_loss_vp_model.txt', all_loss_vx_model, delimiter=',')
    np.savetxt(main_path+'all_loss_vs_model.txt', all_loss_vy_model, delimiter=',')
    np.savetxt(main_path+'all_loss_rho_model.txt', all_loss_rho_model, delimiter=',')
    
    np.savetxt(main_path+'vp_snr.txt', SNR_vp,delimiter=',')
    np.savetxt(main_path+'vs_snr.txt', SNR_vs,delimiter=',')
    np.savetxt(main_path+'rho_snr.txt', SNR_rho,delimiter=',')
    
    
    np.savetxt(main_path+'time.txt',time_each_iter , delimiter=',')
    
    np.savetxt(main_path+'cpu_men_log.txt',cpu_mem_log , delimiter=',')
    np.savetxt(main_path+'gpu_men_log.txt',gpu_mem_log , delimiter=',')
