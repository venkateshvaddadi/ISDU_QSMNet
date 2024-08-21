
clear all; 
close all; 
clc;
%%
% loading the input files

input_dir='data//qsm_2016_recon_challenge/input/'
output_dir='data/qsm_2016_recon_challenge/output/'

%%
% loading the input phase and mask
input_phs=load(strcat(input_dir,'/phs1.mat')).phs_tissue;
input_msk=load(strcat(input_dir,'/msk1.mat')).msk;

% loading the Ground-Truth file
cosmos_ground_truth=load(strcat(input_dir,'/cos1.mat')).cos;
addpath('metrics/')

%%
% loading the output files
ISDU_QSMnet_output=load(strcat('data/qsm_2016_recon_challenge/output/susc_cal_by_ISDU.mat')).sus_cal;
ISDU_QSMnet_output = squeeze(ISDU_QSMnet_output);

%%

% comaprision between the output from SpiNet-QSM model and COSMOS  
ssim_measured= round(compute_ssim(ISDU_QSMnet_output,cosmos_ground_truth), 4);      
rmse_measured = round(compute_rmse(ISDU_QSMnet_output,cosmos_ground_truth), 4);      
psnr_measured= round(compute_psnr(ISDU_QSMnet_output,cosmos_ground_truth), 4);      
hfen_measured= round(compute_hfen(ISDU_QSMnet_output,cosmos_ground_truth), 4);     
xsim_measured= round(compute_xsim(ISDU_QSMnet_output,cosmos_ground_truth), 4);     


N = size(input_phs);
spatial_res = [1 1 1];
[ky,kx,kz] = meshgrid(-N(1)/2:N(1)/2-1, -N(2)/2:N(2)/2-1, -N(3)/2:N(3)/2-1);
kx = (kx / max(abs(kx(:)))) / spatial_res(1);
ky = (ky / max(abs(ky(:)))) / spatial_res(2);
kz = (kz / max(abs(kz(:)))) / spatial_res(3);

% Compute magnitude of kernel and perform fftshift
k2 = kx.^2 + ky.^2 + kz.^2;
kernel = 1/3 - (kz.^2 ./ (k2 + eps)); % Z is the B0-direction
kernel = fftshift(kernel);

phi_x=real(ifftn(fftn(ISDU_QSMnet_output).*kernel)).*single(input_msk);
diff=phi_x-input_phs;

model_loss=norm(diff(:));
%%

% displaying the results
disp('Comparison with Ground-Truth (COSMOS):')
fprintf('SSIM: %.4f \n',ssim_measured)
fprintf('xSIM: %.4f \n',xsim_measured)
fprintf('pSNR: %.4f \n',psnr_measured)
fprintf('RMSE: %.4f \n',rmse_measured)
fprintf('HEFN: %.4f \n',hfen_measured)

fprintf('Model Loss: %.4f \n',model_loss)

%%

% making the output figure
% The figure will be saved at: data/output/
disp_fig(ISDU_QSMnet_output, cosmos_ground_truth,output_dir);