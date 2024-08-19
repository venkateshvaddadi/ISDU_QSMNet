#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:35:07 2022

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:28:22 2022

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:30:38 2022

@author: venkatesh
"""

#%%

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import lr_scheduler


import numpy as np
import torch.optim as optim
import time
import tqdm
import scipy.io
import matplotlib.pyplot as plt
import os
from datetime import datetime

#%%
from config import Config

from qsm_modules.qsm_loss_modules.loss import *
from qsm_modules.qsm_data_loader.QSM_Dataset_updated import mydataloader
from qsm_modules.qsm_dw_models.model_for_dw_QSMnet_lambda_p_trainable import QSMnet
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from qsm_modules.qsm_dw_models.model_for_dw_normal_cnn_lambda_p_trainable import Dw
from qsm_modules.qsm_data_loader.utils import *
from qsm_modules.qsm_dw_models.model_for_dw_deepqsm_lambda_p_trainable import DeepQSM
from qsm_modules.qsm_dw_models.u2net_for_3D import U2NETP_for_3D_cus_ch

#%%





def z_real_to_z_complex(z_real):
  z_complex_recon=torch.complex(z_real[:,0,:,:,:].unsqueeze(1),z_real[:,1,:,:,:].unsqueeze(1))
  # print('\n at z_real_to_z_complex sum',torch.sum(z_real[:,1,:,:,:].unsqueeze(1)))
  
  return z_complex_recon

def z_complex_to_z_real(z_complex):
  z_real=z_complex.real
  z_imag=z_complex.imag
  # print('\n at z_complex_to_z_real sum',torch.sum(z_complex.imag))
  z_real_recon=torch.cat([z_real,z_imag],axis=1)
  return z_real_recon


#%%

# paramaters for dipole kernel
matrix_size = [64, 64, 64]
voxel_size = [1,  1,  1]

device_id=0


#for restoring weights
restore=False



K_unrolling=1
batch_size=2
No_samples=16800
epoch=0
epsilon=1e-5
learning_rate=1e-4
break_amount=1000
is_data_normalized=True
model='3D_U2NETP'
MM_steps=1
#%%
data_source='given_data'
Training_patient_no=1
data_source_no=1
if(data_source=='generated_data'):

    raw_data_path='../QSM_data/data_for_experiments/generated_data/raw_data/'
    data_path='../QSM_data/data_for_experiments/generated_data/data_source_1/'
    #data_path='../QSM_data/data_for_experiments/generated_data/single_patient_patches/patient_'
    #patients_list =[7,32,9,10]

elif(data_source=='given_data'):

    raw_data_path='../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
    
    if(data_source_no==1):
        patients_list =[7,8,9,10,11,12]
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_1//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==2):
        patients_list =[10,11,12,1,2,3]
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_2//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==3):
        patients_list =[1,2,3,4,5,6]
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_3//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==4):
        patients_list =[4,5,6,7,8,9]
        csv_path='../QSM_data/data_for_experiments/given_data/data_source_4//'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    
elif(data_source=='generated_noisy_data'):
    raw_data_path='../QSM_data/data_for_experiments/generated_data/raw_data_noisy_sigma_0.1/'
    data_path='../QSM_data/data_for_experiments/generated_data/data_source_1_sigma_0.1/'

elif(data_source=='given_single_patient_data'):
    raw_data_path='../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
    if(Training_patient_no==1):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_1/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(Training_patient_no==2):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_2/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(Training_patient_no==3):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_3/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'

    if(Training_patient_no==4):
        csv_path='../QSM_data/data_for_experiments/given_data/single_patient/patient_4/'
        data_path='../QSM_data/data_for_experiments/given_data/data_as_patches/'



import os
#print(os.listdir(raw_data_path))
#print(os.listdir(data_path))
print('csv_path:',csv_path)

print("data_path:",data_path)
#%%
# making directory for sving models
print ('*******************************************************')
start_time=time.time()
experiments_folder="savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_WideResNet//"
experiments_folder="savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/"
experiments_folder="savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_3D_U2NetP/full_data_training_without_sampling/"

experiment_name=datetime.now().strftime("%b_%d_%I_%M_%P_")+"model_K_"+ str(K_unrolling)+"_"+data_source+'_dw_'+model;
cwd=os.getcwd()

directory=experiments_folder+"/"+experiment_name+"/"
print(directory)
print('Model will be saved to  :', directory)

#%%


#%%

loader = mydataloader(csv_path+'/csv_files/train.csv', data_path)
trainloader = DataLoader(loader, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)

valdata    = mydataloader(csv_path+'/csv_files/val.csv', data_path)
valloader  = DataLoader(valdata, batch_size = batch_size, shuffle=True, num_workers=1,drop_last=True)


#%%

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
try:
    os.makedirs(directory)
except:
    print("Exception...")

# making a log file........
import logging

logging.basicConfig(filename=directory+'app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('BATCH SIZE:'+str(batch_size))
logging.warning('K_UNROLLING:'+str(K_unrolling))
logging.warning('restore:'+str(restore))
logging.warning('No training samples:'+str(batch_size*len(trainloader)))
logging.warning('No validation samples:'+str(batch_size*len(valloader)))
logging.warning('learning_rate:'+str(learning_rate))
logging.warning('break_amount:'+str(break_amount))
logging.warning('data_normalized:'+str(is_data_normalized))
logging.warning('data_source:'+str(data_source))
logging.warning('data_path:'+(data_path))
logging.warning('csv_path:'+(csv_path))

logging.warning('no traiiiining smaples'+str(len(trainloader)))
logging.warning('no validation smaples'+str(len(valloader)))
logging.warning('MMsteps: '+str(MM_steps))

#%%


#%%
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk=dk.float()
dk = torch.unsqueeze(dk, dim=0)

dk=dk.cuda(device_id)
Dk_square=dk * dk

print('dk.shape',dk.shape)
print('Dk_square.shape',Dk_square.shape)

#%%

config  = Config()

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

dw1 = torch.nn.DataParallel(dw, device_ids=[device_id])  



print("p",dw.p)
print("lambda_val",dw.lambda_val)
#%%
#%%
if restore:
    restore_weights_path='savedModels/Spinet_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/experiments_on_given_data/dw_WideResNet/full_data_training_without_sampling/Jun_28_11_12_pm_model_K_1_given_data_dw_WideResNet_data_source_2/Spinet_QSM_model_29_.pth'
    dw.load_state_dict(torch.load(restore_weights_path))
    print("we have restored weights of the model:",restore_weights_path)
    logging.warning('restore_weights_path:'+(restore_weights_path))
    print("p",dw.p)
    print("lambda_val",dw.lambda_val)

#%%

# Experiment details....
print('----------------------------------------------------------------')
print('\n Experiment_name:')
print(directory)
print('----------------------------------------------------------')

print('K_unrolling:',K_unrolling)
print('batch_size:',batch_size)
print('No_samples:',No_samples)
print('break_amount:',break_amount)
print('learning_rate:',learning_rate)
print('device_id:',device_id)
print('----------------------------------------------------------')

print('no of training batches:',len(trainloader))
print('----------------------------------------------------------')
print('no of validation batches:',len(valloader))



logging.warning('no of training batches:'+str(len(trainloader)))
print('----------------------------------------------------------')
logging.warning('no of validation batches:'+str(len(valloader)))


print('restored model',restore)
logging.warning('restored model:'+str(restore))

if(restore):
        print("we have restored weights of the model:",restore_weights_path)
        logging.warning("we have restored weights of the model:"+restore_weights_path)


#%%
ss = sobel_kernel()
ss=ss.float()
ss = ss.cuda(device_id)
# print(ss.shape)

#%%
optimizer = optim.Adam(dw.parameters(), lr=learning_rate)
#%%
loss_Train=[]
loss_Val=[]
lambda_list=[]
p_list=[]
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
    
    x_0 = torch.zeros(size=(1, 1, 64, 64, 64),dtype=torch.float64).cuda(device_id)
    
    temp=b_gpu(local_field_gpu, dw.lambda_val,z_k_gpu)
    
    # print(temp.shape,temp.dtype)

    r_0 = b_gpu(local_field_gpu, dw.lambda_val,z_k_gpu)-A_gpu(x_0,dw.lambda_val,dw.p)
    
    # print('\t r_0.shape',r_0.shape)
    p_0 = r_0

    # print('\t r_0.shape', r_0.shape)
    # print('\t P_0 shape', p_0.shape)

    r_old = r_0
    p_old = p_0
    x_old = x_0

    r_stat = []
    
    r_stat.append(torch.sum(r_old.conj()*r_old).real.item())
    # print('\t r_stat',r_stat)

    for i in range(30):

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
        # print('\t x_new.shape',x_new.shape,x_new.dtype)
        

        # updating the remainder
        r_new = r_old-alpha*Ap_old
        # print('\t r_new.shape',r_new.shape)
        
        # beta calculation
        r_new_T_r_new = torch.sum(r_new.conj() * r_new)

        #r_stat.append(r_new_T_r_new.real.item())
        
        beta = r_new_T_r_new/r_old_T_r_old
        
        # print('\t beta',beta,beta.dtype)

        # new direction p calculationubu 
        p_new = r_new+beta*p_old
        
        
        # print('p_new',p_new)
        # print('\t p_new.shape',p_new.shape,p_new.dtype)

        # preparing for the new iteration...

        r_old = r_new
        p_old = p_new
        x_old = x_new

    # print('\t x_new',x_new.shape,x_new.dtype)
    # print(r_stat)
    
    return x_new

#%%


    


epoch=0
#%%
for epoch in range(60):
    runningLoss = 0

    print('\n-------------------------------------------------------------------------------------------------\n')
    print('epoch---',epoch)

    dw.train() 

    for i, data in tqdm.tqdm(enumerate(trainloader)):
            phs, msk, sus,file_name = data
            #print(file_name)
            #dk_repeat=torch.repeat_interleave(dk, repeats=batch_size,dim=(0))
            dk_repeat = dk.repeat(batch_size,1,1,1,1)
            
            phs=phs.cuda(device_id)
            msk=msk.cuda(device_id)
            sus=sus.cuda(device_id)
            
            # if(i==break_amount):
            #     break;

            # print('\n phs.shape,msk.shape,sus.shape',phs.shape,msk.shape,sus.shape)
            
            # initialization ....
            # taking x_0 = A^Hb
            
            phs_F = torch.fft.fftn(phs,dim=[2,3,4])
            phs_F = dk_repeat * phs_F
            x_0_complex = torch.fft.ifftn(phs_F,dim=[2,3,4])
            
            
            x_0_real=z_complex_to_z_real(x_0_complex)
            
            x_k_real=x_0_real
            x_k_complex=x_0_complex
            
            for k in range(K_unrolling):

                
                x_k_complex=x_k_complex*msk
                x_k_real=z_complex_to_z_real(x_k_complex).float()
                #print(x_k_real.shape)
                x_k_real=(x_k_real-sus_mean)/sus_std
                z_k_real = dw1(x_k_real)
                z_k_real=z_k_real*sus_std+sus_mean

                z_k_complex=z_real_to_z_complex(z_k_real)


                solutions=[]
                for j in range(batch_size):
                    
                    # code added for spinet

                    
                    x_k_j_complex=x_k_complex[j,:,:,:,:].unsqueeze(0)
                    z_k_j_complex=z_k_complex[j,:,:,:,:].unsqueeze(0)
                    
                    for mm_steps in range(MM_steps):
                        
                        w=torch.pow( (x_k_j_complex-z_k_j_complex) , ((dw.p/2)-1.0))
                        w_square=w.conj()*w
                        x_k_j_complex=CG_GPU(phs[j,:,:,:,:].unsqueeze(0),z_k_complex[j,:,:,:,:].unsqueeze(0))
                        x_k_j_complex=x_k_j_complex*msk[j:j+1,:,:,:,:]
                        # print('\n mm_steps:',mm_steps)
                        
                    solutions.append(x_k_j_complex)
                    
                
                x_k_complex_new=torch.cat(solutions,dim=0)
                #print('x_k_complex.shape:',x_k_complex.shape)
                x_k_complex=x_k_complex_new
                x_k_complex=x_k_complex*msk
 

                
            x_k_complex = x_k_complex * msk
            
            optimizer.zero_grad()
            loss=total_loss_l1(chi=x_k_complex.real, y=sus, b=phs, d=dk, m=msk, sobel=ss)
            loss.backward()

            #nn.utils.clip_grad_value_(dw.parameters(), clip_value=1.0)

            optimizer.step()
            
            # print('lambda',dw.lambda_val.item(),dw.lambda_val.grad.item())
            # print('p',dw.p.item(),dw.p.grad.item())
            runningLoss += loss.item()

# printing the training loss.........

    loss_Train.append(runningLoss/len(trainloader))
    print('Training_loss:', loss_Train)    
    
    import time

    print('---------------------------------------------------------')
    print('lambda',dw.lambda_val)
    print('p',dw.p.item())
    print('---------------------------------------------------------')
    torch.save(dw.state_dict(), directory+'Spinet_QSM_model_'+str(epoch)+'_.pth')
    time.sleep(10)

    # validation code is here
    running_val_Loss=0
    dw.eval()


    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(valloader)):
            try:
                phs, msk, sus,file_name = data
                


                dk_repeat = dk.repeat(batch_size,1,1,1,1)
                phs=phs.cuda(device_id)
                msk=msk.cuda(device_id)
                sus=sus.cuda(device_id)
                
                # initialization ....
                # taking x_0 = A^Hb
                
                phs_F = torch.fft.fftn(phs,dim=[2,3,4])
                phs_F = dk_repeat * phs_F
                x_0_complex = torch.fft.ifftn(phs_F,dim=[2,3,4])
                
                x_0_real=z_complex_to_z_real(x_0_complex)
    
                x_k_real=x_0_real
                x_k_complex=x_0_complex
                
                
                # we are running for the k=1 (unrolling for k=1)
                for k in range(K_unrolling):
                    
                    x_k_complex = x_k_complex * msk
                    x_k_real=z_complex_to_z_real(x_k_complex).float()

                    x_k_real=(x_k_real-sus_mean)/sus_std
                    z_k_real = dw1(x_k_real)
                    z_k_real=z_k_real*sus_std+sus_mean
                    
                    # print(torch.mean(torch.square(temp-z_k_real)))



                    z_k_complex=z_real_to_z_complex(z_k_real)
    
                    solutions=[]
                    for j in range(batch_size):
                        
                        # code added for spinet
    
                        x_k_j_complex=x_k_complex[j,:,:,:,:].unsqueeze(0)
                        z_k_j_complex=z_k_complex[j,:,:,:,:].unsqueeze(0)
                        
                        for mm_steps in range(MM_steps):
                            w=torch.pow( (x_k_j_complex-z_k_j_complex) , ((dw.p/2)-1.0))
                            w_square=w.conj()*w
                            x_k_j_complex=CG_GPU(phs[j,:,:,:,:].unsqueeze(0),z_k_complex[j,:,:,:,:].unsqueeze(0))
                            x_k_j_complex=x_k_j_complex*msk[j:j+1,:,:,:,:]
        
                            # print('\n mm_steps:',mm_steps)
                        
                    solutions.append(x_k_j_complex)
                    
                    x_k_complex_new=torch.cat(solutions,dim=0)
                    
                    x_k_complex=x_k_complex_new
                    x_k_complex = x_k_complex * msk

                    
                x_k_complex = x_k_complex * msk

                loss=total_loss_l1(chi=x_k_complex.real, y=sus, b=phs, d=dk, m=msk, sobel=ss)
                running_val_Loss += loss.item()
    
            except Exception as e: 
                print(e)
                print('error at',i)

    loss_Val.append(running_val_Loss/len(valloader))
    print('Validation_loss:',loss_Val)

    # printing the validation loss...
    lambda_list.append(dw.lambda_val.item())
    p_list.append(dw.p.item())
    
    print('Checking what is the K value:',K_unrolling)
    # saving mdodel details    
    import pandas as pd
    
    model_details={"train_loss":loss_Train,"valid_loss":loss_Val,'lambda':lambda_list,'p':p_list}
    df = pd.DataFrame.from_dict(model_details) 
    
    path=directory+'model_details.csv'
    df.to_csv (path, index = False, header=True)
    scipy.io.savemat(directory+"/model_details.mat", model_details)
    
    print('\n-------------------------------------------------------------------------------------------------\n')

#%%


