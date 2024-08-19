#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:21:10 2024

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:42:42 2024

@author: venkatesh
"""

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
from qsm_modules.qsm_data_loader.QSM_Dataset_updated import mydataloader
from qsm_modules.qsm_dw_models.model_for_dw_deepqsm_lambda_p_trainable import DeepQSM
from qsm_modules.qsm_dw_models.model_for_dw_QSMnet_lambda_p_trainable import QSMnet
from qsm_modules.qsm_loss_modules.loss import *
from qsm_modules.qsm_dw_models.WideResnet import WideResNet
from qsm_modules.qsm_dw_models.model_for_dw_normal_cnn_lambda_p_trainable import Dw
from qsm_modules.qsm_dw_models.u2net_for_3D import U2NETP_for_3D_cus_ch

#%%

matrix_size = [160,160, 160]
voxel_size = [1,  1,  1]

#%%
#loading the model\
K_unrolling=4
batch_size=1
device_id=0
data_source_no=1
is_data_normalized=True

model='WideResNet'

#%%


experiments_folder="savedModels/MoDL_QSM_with_unshared_weights_with_random_sampling/experiments_on_SNU_data/dw_WideResNet/training_on_full_data/"
experiment_name="Oct_11_09_36_am_model_K_4_given_data_dw_WideResNet_data_source_1_sampled_4000//"

model_name="best_model_with_K_4.pt"
model_path=experiments_folder+"/"+experiment_name+"/"+model_name
print('model_path:',model_path)

try:
    os.makedirs(experiments_folder+"/"+experiment_name+"/output_csv")
except:
    print("Exception...")
#%%





#%%

if(model=='deepqsm'):
    dw = DeepQSM().cuda(device_id)
elif(model=='QSMnet'):
    dw=QSMnet().cuda(device_id)
elif(model=='WideResNet'):
    dw=WideResNet().cuda(device_id)
elif(model=='simple_cnn'):
    dw=Dw().cuda(device_id)
elif(model == 'U2NETP_for_3D_cus_ch'):
     net = U2NETP_for_3D_cus_ch(in_ch=1,out_ch=1,cus_ch=64,mid_ch=16)

#%%

# loading the model
checkpoint = torch.load(model_path)

print('K (Unrolling parameter)',checkpoint['K_unrolling'])
print(checkpoint['epoch_no'])
print(checkpoint['training_loss'])
print(checkpoint['validation_loss'])
#print(checkpoint['model_0'])

K_unrolling=checkpoint['K_unrolling']

model_dic={}

for i in range(K_unrolling):
    dw=WideResNet().cuda(device_id)
    print('before adding into list:',id(dw))

    dw.load_state_dict(checkpoint['model_'+str(i)])

    model_dic['model_'+str(i)]=dw

    temp=model_dic['model_'+str(i)]

    print('after adding to list',id(model_dic['model_'+str(i)]))
    print('after adding to list',id(temp))

    print('dw_'+str(i),model_dic['model_'+str(i)].lambda_val,model_dic['model_'+str(i)].p)



#%%
# making evalutation mode

for eval_i in range(K_unrolling):
    dw=model_dic['model_'+str(eval_i)]
    dw.eval() 


#%%
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk = torch.unsqueeze(dk, dim=0)
print(dk.shape)

dk=dk.float().cuda(device_id)
Dk_square=torch.multiply(dk, dk)
Dk_square=Dk_square.cuda(device_id)
#%%
# define the train data stats



if(not is_data_normalized):
    sus_mean=0
    sus_std=1
    print('\n\n data is not normalized..................\n\n ')

else:
    stats = scipy.io.loadmat('data/training_stats/tr-stats.mat')
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
#%%


def CG_GPU_for_given_unrolling_k(local_field_gpu, z_k_gpu,given_unrolling_k):

    # print('Inside CG_GPU_for_given_unrolling_k Calling............')
    # print('Unrolling:',given_unrolling_k)
    dw=model_dic['model_'+str(given_unrolling_k)]
    # print('In side CG_GPU_for_given_unrolling_k:',given_unrolling_k,id(dw))
    # print('\n','-'*100,'\n')

    

#    x_0 = torch.zeros(size=(1, 1, 176, 176, 160),dtype=torch.float64).cuda(device_id)
    x_0 = torch.zeros(size=(1, 1, local_field_gpu.shape[2],local_field_gpu.shape[3], local_field_gpu.shape[4]),dtype=torch.float64).cuda(device_id)

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

def padding_data(input_field):
    N = np.shape(input_field)
    N_16 = np.ceil(np.divide(N,16.))*16
    N_dif = np.int16((N_16 - N) / 2)
    npad = ((N_dif[0],N_dif[0]),(N_dif[1],N_dif[1]),(N_dif[2],N_dif[2]))
    pad_field = np.pad(input_field, pad_width = npad, mode = 'constant', constant_values = 0)
    # pad_field = np.expand_dims(pad_field, axis=0)
    # pad_field = np.expand_dims(pad_field, axis=0)
    return pad_field, N_dif, N_16

def crop_data(result_pad, N_dif):
    result_pad = result_pad.squeeze()
    N_p = np.shape(result_pad)
    result_final  = result_pad[N_dif[0]:N_p[0]-N_dif[0],N_dif[1]:N_p[1]-N_dif[1],N_dif[2]:N_p[2]-N_dif[2]]
    return result_final


#%%
# temp={}

with torch.no_grad():


            phs=scipy.io.loadmat('data/qsm_2016_recon_challenge/input/phs1.mat')['phs_tissue']
            sus=scipy.io.loadmat('data/qsm_2016_recon_challenge/input/cos1.mat')['cos']
            msk=scipy.io.loadmat('data/qsm_2016_recon_challenge/input/msk1.mat')['msk']
            
            print(phs.shape)
            print(msk.shape)
            print(sus.shape)
            phs, N_dif_phs, N_16_phs=padding_data(phs)
            msk, N_dif_msk, N_16_msk=padding_data(msk)
            sus, N_dif_sus, N_16_sus=padding_data(sus)
            print(phs.shape)
            print(msk.shape)
            print(sus.shape)

            phs=torch.unsqueeze(torch.unsqueeze(torch.tensor(phs),0),0)
            sus=torch.unsqueeze(torch.unsqueeze(torch.tensor(sus),0),0)
            msk=torch.unsqueeze(torch.unsqueeze(torch.tensor(msk),0),0)
    
            phs=phs.cuda(device_id)
            sus=sus.cuda(device_id)
            msk=msk.cuda(device_id)
    
            tic()
            temp_shape=phs.shape
            matrix_size = [temp_shape[2],temp_shape[3], temp_shape[4]]
            voxel_size = [1,  1,  1]



            # initialization ....
            # taking x_0 = A^Hb
            #-------------------------
            dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
            dk = torch.unsqueeze(dk, dim=0)
            print(dk.shape)
            
            dk=dk.float().cuda(device_id)
            Dk_square=torch.multiply(dk, dk)
            Dk_square=Dk_square.cuda(device_id)
            #-------------------------




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
                
                dw=model_dic['model_'+str(k)]

                z_k_real = dw(x_k_real)
                
                
                
                z_k_real=z_k_real*sus_std+sus_mean

                # for saving spinet_qsm_output_with_all_stages
                # temp['z_'+str(k+1)+"_real"]=(z_k_real*msk).detach().cpu().numpy()
            
                # print(z_k_real,z_k_real.shape)
                z_k_complex=z_real_to_z_complex(z_k_real)
    
                w=torch.pow( (x_k_complex-z_k_complex) , ((dw.p/2)-1.0))
                w_square=w.conj()*w
                w_square_sum=torch.sum(w_square)
                
                
                x_k_complex=CG_GPU_for_given_unrolling_k(phs,z_k_complex,k)

                x_k_complex=x_k_complex*msk

                # for saving spinet_qsm_output_with_all_stages
                # temp['x_'+str(k+1)+"_real"]=z_complex_to_z_real(x_k_complex).detach().cpu().numpy()
                           
    
            toc()
            
            
            
            
            
            
            x_k_cpu=(x_k_complex.real.detach().cpu().numpy())*(msk.detach().cpu().numpy() )
            x_k_cpu=crop_data(x_k_cpu, N_dif_sus)
            print('x_k_cpu.shape after cropping',x_k_cpu.shape)
            mdic  = {"sus_cal" : x_k_cpu}
            filename  = 'data/qsm_2016_recon_challenge/output/susc_cal_by_ISDU.mat'
            scipy.io.savemat(filename, mdic)

            # spinet_qsm_output_with_all_stages
            # scipy.io.savemat(outdir + 'spinet_qsm_output_with_all_stages'+ str(i)+'-'+str(j)+'.mat', temp)
            # print(temp.keys())




#%%

# temp=scipy.io.loadmat('./../QSM_data/data_for_experiments/data_source_1/data_distribution.mat')
# print(temp.keys())
# for key in temp.keys():
#     print(key,temp[key])











