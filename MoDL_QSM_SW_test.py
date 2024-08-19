#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:02:28 2022

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:24:27 2021

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:44:32 2021

@author: venkatesh
"""


import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import nn



import numpy as np
import time
import scipy.io
import tqdm
import matplotlib.pyplot as plt
import scipy.io
import os
#%%
from config import Config
from qsm_modules.qsm_data_loader.QSM_Dataset_updated import mydataloader
from qsm_modules.qsm_dw_models.model_for_dw_deepqsm_lambda_p_trainable import DeepQSM
from qsm_modules.qsm_dw_models.model_for_dw_QSMnet_lambda_p_trainable import QSMnet
from qsm_modules.qsm_loss_modules.loss import *
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from qsm_modules.qsm_dw_models.model_for_dw_normal_cnn_lambda_p_trainable import Dw
from qsm_modules.qsm_dw_models.u2net_for_3D import U2NETP_for_3D_cus_ch

#%%

matrix_size = [176,176, 160]
voxel_size = [1,  1,  1]

#%%
#loading the model\
K_unrolling=2
batch_size=1
device_id=1
data_source_no=1
is_data_normalized=True

model='WideResNet'

#%%
#2  5  8  3  6 13 
epoch=33

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="epoch number for testing ",type=int,default=epoch)
args = parser.parse_args()

epoch=args.epoch;

print('epoch:',epoch)
#%%
data_source='given_single_patient_data'
Training_patient_no=4
data_source_no=1

if(data_source=='generated_data'):

    raw_data_path='../QSM_data/data_for_experiments/generated_data/raw_data_noisy_sigma_0.05/'
    data_path='../QSM_data/data_for_experiments/generated_data/data_source_1/'
    patients_list =[7,32,9,10]

elif(data_source=='given_data'):

    raw_data_path='../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'

    if(data_source_no==1):
        patients_list =[7,8,9,10,11,12]
        data_path='../QSM_data/data_for_experiments/given_data/data_source_1/'
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_1//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==2):
        patients_list =[10,11,12,1,2,3]
        data_path='../QSM_data/data_for_experiments/given_data/data_source_2/'
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_2//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==3):
        patients_list =[1,2,3,4,5,6]
        data_path='../QSM_data/data_for_experiments/given_data/data_source_3/'
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_3//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'


    elif(data_source_no==4):
        patients_list =[4,5,6,7,8,9]
        data_path='../QSM_data/data_for_experiments/given_data/data_source_4/'
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_4//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

elif(data_source=='generated_noisy_data'):
    raw_data_path='../QSM_data/data_for_experiments/generated_data/raw_data/'
    data_path='../QSM_data/data_for_experiments/generated_data/data_source_1/'
    patients_list =[7,32,9,10]



elif(data_source=='generated_undersampled_data'):
    raw_data_path='../QSM_data/data_for_experiments/generated_data/sampling_data/sampled_0.05//'
    data_path='../QSM_data/data_for_experiments/generated_data/data_source_1/'
    patients_list =[7,32,9,10]

elif(data_source=='given_single_patient_data'):
    raw_data_path='../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
    if(Training_patient_no==1):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_1/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[2,3,4,5,7,8,9,10,11,12]
    elif(Training_patient_no==2):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_2/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[1,3,4,5,7,8,9,10,11,12]

    elif(Training_patient_no==3):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_3/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[1,2,4,5,7,8,9,10,11,12]
    if(Training_patient_no==4):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_4/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[1,2,3,5,7,8,9,10,11,12]

import os
print(os.listdir(data_path))
print(os.listdir(raw_data_path))
print('csv_path',csv_path)
print('data_path:',data_path)
print('raw_data_path',raw_data_path)
#%%


experiments_folder="savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/single_patient/"
experiment_name="Jun_28_02_32_pm_model_K_2_given_single_patient_data_dw_WideResNet_patient_4///"


# experiments_folder="savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_3D_U2NETP_32_16/full_data_training_without_sampling/"
# experiment_name="Jun_19_03_11_pm_model_K_1_given_data_dw_3D_U2NETP_32_16//"


model_name="Spinet_QSM_model_"+str(epoch)+"_.pth"
model_path=experiments_folder+"/"+experiment_name+"/"+model_name
print('model_path:',model_path)

try:
    os.makedirs(experiments_folder+"/"+experiment_name+"/output_csv")
except:
    print("Exception...")

#%%

if(model=='deepqsm'):
    dw = DeepQSM().cuda(device_id)
elif(model=='QSMnet'):
    dw=QSMnet().cuda(device_id)
elif(model=='WideResNet'):
    dw=WideResNet().cuda(device_id)
elif(model=='simple_cnn'):
    dw=Dw().cuda(device_id)
elif(model == '3D_U2NETP'):
     dw = U2NETP_for_3D_cus_ch(in_ch=2,out_ch=2,cus_ch=64,mid_ch=16)
elif(model == 'dw_3D_U2NETP_32_16'):
     dw = U2NETP_for_3D_cus_ch(in_ch=2,out_ch=2,cus_ch=32,mid_ch=16)
#%%
print("dw.lambda_val",dw.lambda_val)
print("dw.p",dw.p)
#%%
dw.load_state_dict(torch.load(model_path))
#dw.load_state_dict(torch.load('./savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_WideResNet/ablation_study/Dec_02_06_12_pm_model_K_1_B_2_N_2000_dw_WideResNet_data_source_1_p_2.0/Spinet_QSM_model_32_.pth'))


#%%
#dw = torch.nn.DataParallel(dw, device_ids=[device_id])  


dw.eval()
dw = dw.cuda(device_id)
#%%
print("Evaluation happening")
print('dw.lambda_val',dw.lambda_val)
print('dw.p',dw.p)

#%%
last_string=model_path.split("/")[-1]
directory=model_path.replace(last_string,"")

print('directory:',directory)
print(os.listdir(directory))

#%%
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk = torch.unsqueeze(dk, dim=0)
print(dk.shape)

dk=dk.float().cuda(device_id)
Dk_square=torch.multiply(dk, dk)
Dk_square=Dk_square.cuda(device_id)
#%%
# define the train data stats

stats = scipy.io.loadmat(csv_path+'/csv_files/tr-stats.mat')


if(not is_data_normalized):
    sus_mean=0
    sus_std=1
    print('\n\n data is not normalized..................\n\n ')

else:
    stats = scipy.io.loadmat(csv_path+'/csv_files/tr-stats.mat')
    sus_mean= torch.tensor(stats['out_mean']).cuda(device_id)
    sus_std = torch.tensor(stats['out_std' ]).cuda(device_id)
    print(sus_mean,sus_std)


#%%

def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        #print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        print(str(time.time() - startTime_for_tictoc) )
    else:
        print("Toc: start time not set")

#%%

def z_real_to_z_complex(z_real):
  z_complex_recon=torch.complex(z_real[:,0,:,:,:].unsqueeze(1),z_real[:,1,:,:,:].unsqueeze(1))
  return z_complex_recon

def z_complex_to_z_real(z_complex):
  z_real=z_complex.real
  z_imag=z_complex.imag
  z_real_recon=torch.cat([z_real,z_imag],axis=1)
  return z_real_recon



#%%

def b_gpu(y,lambda_val, z_k):
    
    # print('\t \t  calling b_gpu:')
    # print('y.shape:',y.shape)
    # print('z_k.shape:',z_k.shape)

    # print(y)
    # print(lambda_val)
    # print(z_k)

    output1 = torch.fft.fftn(y)
    output2 = dk * output1
    output3 = torch.fft.ifftn(output2)
    
    # print('at b_gpu output3:',output3.dtype,output3.shape)
    #print('output3.get_device:', output3.get_device())
    #print('lambda_val.get_device:', lambda_val.get_device())
    #print('z_k.get_device:',z_k.get_device())

    # code added for spinet
    w_square_z_k=w_square*z_k
    # print('w_square_z_k.shape:',w_square_z_k.shape)

    # code added for spinet
    output4 = output3+lambda_val*w_square_z_k
    
    # print('output4.shape',output4.shape,output4.dtype)


    return output4

# x sshould be in gpu....
    
def A_gpu(x,lambda_val,p):
    # print('\t \t calling A')
    # print('---------------------')
    output1 = Dk_square*torch.fft.fftn(x)
    output2 = torch.fft.ifftn(output1)
    
    # print('at A_gpu',output2.dtype,output2.shape)
    
    
    # print('output2.shape:', output2.shape)
    # print('w_square.shape:',w_square.shape)
    # print('x.shape:',x.shape)
    
    # code added for spinet
    
    # print('output2',output2)

    w_square_x=w_square*x
    
    # print('w_square',w_square)
    # print('w_square_x.shape:',w_square_x.shape)
    
    # code added for spinet
    output3 = output2+lambda_val * w_square_x
    # print('at A_gpu',output3.dtype,output3.shape)

    
    
    return output3

def CG_GPU(local_field_gpu, z_k_gpu):

    # print('CG GPU Calling............')
    #print('--------------------------')
    
    x_0 = torch.zeros(size=(1, 1, 176, 176, 160),dtype=torch.float64).cuda(device_id)
    
    temp=b_gpu(local_field_gpu, dw.lambda_val,z_k_gpu)
    
    # print(temp.shape,temp.dtype)

    r_0 = b_gpu(local_field_gpu, dw.lambda_val,z_k_gpu)-A_gpu(x_0,dw.lambda_val,dw.p)
    
    #print('r_0.shape',r_0.shape)
    p_0 = r_0

    #print('r_0.shape', r_0.shape)
    #print('P_0 shape', p_0.shape)

    r_old = r_0
    p_old = p_0
    x_old = x_0

    r_stat = []
    
    r_stat.append(torch.sum(r_old.conj()*r_old).real.item())
    # print('r_stat',r_stat)

    for i in range(25):

        # alpha calculation
        r_old_T_r_old = torch.sum(r_old.conj()*r_old)
        # print('\t r_old_T_r_old',r_old_T_r_old,r_old_T_r_old.shape)


        if(r_old_T_r_old.real.item()<1e-10):
            # print('r_stat',r_stat,r_old_T_r_old.item(),'iteration:',len(r_stat))
            
            # logging.warning('r_stat')
            # logging.warning(r_stat)
            # logging.warning(r_old_T_r_old.item())
            # logging.warning('iteration:'+str(len(r_stat)))
            
            return x_old
        
        
        if(r_old_T_r_old.real.item()>r_stat[-1] and r_stat[-1] < 1e-06):
            # print("Convergence issue:",r_old_T_r_old.item(),r_stat[-1])
            
            # logging.warning("Convergence issue:")
            # logging.warning(r_old_T_r_old.item())
            # logging.warning(r_stat[-1])
            return x_old


        r_stat.append( torch.sum(r_old.conj()* r_old).real.item())

        # print('dw.lambda_val,dw.p',dw.lambda_val.item(),dw.p.item())
        # print('dw.lambda_val',dw.lambda_val)
        # print('dw.p',dw.p)
        Ap_old = A_gpu(p_old,dw.lambda_val,dw.p)
        # print('\t Ap_old.shape',Ap_old.shape)
        # print('\t Ap_old',Ap_old)
        
        p_old_T_A_p_old = torch.sum(p_old.conj() * Ap_old)
        # print('\t p_old_T_A_p_old',p_old_T_A_p_old.item())
        
        alpha = r_old_T_r_old/p_old_T_A_p_old
        # print('\t alpha',alpha.item())

        # updating the x
        x_new = x_old+alpha*p_old
        # print('\t x_new',x_new)
        # print('\t x_new.shape',x_new.shape)
        

        # updating the remainder
        r_new = r_old-alpha*Ap_old
        # print('\t r_new.shape',r_new.shape)
        
        # beta calculation
        r_new_T_r_new = torch.sum(r_new.conj() * r_new)

        #r_stat.append(r_new_T_r_new.real.item())
        
        beta = r_new_T_r_new/r_old_T_r_old
        
        # print('beta',beta)

        # new direction p calculationubu 
        p_new = r_new+beta*p_old
        
        
        # print('p_new',p_new)
        # print('p_new.shape',p_new.shape)

        # preparing for the new iteration...

        r_old = r_new
        p_old = p_new
        x_old = x_new

    # print('x_new.shape',x_new.shape,x_new.dtype)
    # print(r_stat)
    
    return x_new


#%%

outdir = directory+'predictions_'+str(epoch)+"/"
print(outdir)
import os
try:
    os.makedirs(outdir)
except:
    print("aalready tested..")


#%%



#%%








# from qsm_modules.qsm_loss_modules.pytorch_ssim import *
# #%%


# ssim_calculation = SSIM3D(window_size = 11)

# def SSIM_loss_on_Minibatch(prediction, taget):
#     batch_size=prediction.shape[0]

#     total_ssim_loss=0
#     for i in range(batch_size):
#         ssim_loss_val=1-ssim_calculation(prediction[i:i+1,:,:,:,:],taget[i:i+1,:,:,:,:])
#         total_ssim_loss=total_ssim_loss+ssim_loss_val
#         # print(ssim_loss_val)
#     total_ssim_loss=total_ssim_loss/batch_size;
#     return total_ssim_loss
#%%
# temp={}

with torch.no_grad():


    for i in patients_list:
        print("Patinte:"+str(i)+"\n")
        for j in range(1,6):
            phs=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/phs'+str(j)+'.mat')['phs']
            sus=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/cos'+str(j)+'.mat')['cos']
            msk=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/msk'+str(j)+'.mat')['msk']
            
            # for saving spinet_qsm_output_with_all_stages
            # temp['msk']=msk
            # temp['cos']=msk

            phs=torch.unsqueeze(torch.unsqueeze(torch.tensor(phs),0),0)
            sus=torch.unsqueeze(torch.unsqueeze(torch.tensor(sus),0),0)
            msk=torch.unsqueeze(torch.unsqueeze(torch.tensor(msk),0),0)
    
            phs=phs.cuda(device_id)
            sus=sus.cuda(device_id)
            msk=msk.cuda(device_id)
    
            tic()
    
            # initialization ....
            # taking x_0 = A^Hb
            
            dk_repeat = dk.repeat(batch_size,1,1,1,1)
            phs_F = torch.fft.fftn(phs,dim=[2,3,4])
            phs_F = dk_repeat * phs_F
            x_0_complex = torch.fft.ifftn(phs_F,dim=[2,3,4])
            x_0_real=z_complex_to_z_real(x_0_complex)

            # for saving spinet_qsm_output_with_all_stages
            # temp['x_0_real']=(x_0_real*msk).detach().cpu().numpy()

            x_k_real=x_0_real
            x_k_complex=x_0_complex
    
            
            for k in range(K_unrolling):
                x_k_complex=x_k_complex*msk
                x_k_real=z_complex_to_z_real(x_k_complex).float()
                
     
                x_k_real=(x_k_real-sus_mean)/sus_std
                z_k_real = dw(x_k_real)
                z_k_real=z_k_real*sus_std+sus_mean

                # for saving spinet_qsm_output_with_all_stages
                # temp['z_'+str(k+1)+"_real"]=(z_k_real*msk).detach().cpu().numpy()
            
                # print(z_k_real,z_k_real.shape)
                z_k_complex=z_real_to_z_complex(z_k_real)
    
                w=torch.pow( (x_k_complex-z_k_complex) , ((dw.p/2)-1.0))
                w_square=w.conj()*w
                w_square_sum=torch.sum(w_square)
                
                
                x_k_complex=CG_GPU(phs,z_k_complex)
                x_k_complex=x_k_complex*msk

                # for saving spinet_qsm_output_with_all_stages
                # temp['x_'+str(k+1)+"_real"]=z_complex_to_z_real(x_k_complex).detach().cpu().numpy()
                           
    
            toc()
            
            
            
            # print('SSIM: ',ssim_calculation(x_k_complex.real.float(), sus))
            # print(SSIM_loss_on_Minibatch(x_k_complex.real.float(), sus))
            
            x_k_cpu=(x_k_complex.real.detach().cpu().numpy())*(msk.detach().cpu().numpy() )
            mdic  = {"modl" : x_k_cpu}
            filename  = outdir + 'modl-net-'+ str(i)+'-'+str(j)+'.mat'
            scipy.io.savemat(filename, mdic)

            # spinet_qsm_output_with_all_stages
            # scipy.io.savemat(outdir + 'spinet_qsm_output_with_all_stages'+ str(i)+'-'+str(j)+'.mat', temp)
            # print(temp.keys())


#%%











