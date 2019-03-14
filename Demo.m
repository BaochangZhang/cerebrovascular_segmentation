close all;
clc;
clear;

path(path,'./prepare_step/');


numid=1;

% load a data from ./data/mat_data_prepared/

% load(strcat( './data/mat_data_prepared/MIDAS',num2str(numid,'%03d'),'.mat'))
load(strcat( './data/mat_data_prepared/AVM',num2str(numid,'%03d'),'.mat'))
% load(strcat( './data/mat_data_prepared/GZ',num2str(numid,'%03d'),'.mat'))

VesselPs=find(TempVessel>0);
rand_index = randperm(length(VesselPs));
draw_rand_index = rand_index(1:round(length(VesselPs)*1/10));
VesselPoint_Matlab=[];
[VesselPoint_Matlab(:,1),VesselPoint_Matlab(:,2),VesselPoint_Matlab(:,3)] = ind2sub(size(TempVessel),VesselPs(draw_rand_index));

[Dx,vessel,vessel_ratio]=GMM_MRF_SEMI(ImageT,Iout,3,3,[1:20],6,1.2,3,500,VesselPoint_Matlab);

% close all;

figure;
subplot(2,2,1)
set(gca,'color','black'); 
[a,b,c] = size(vessel);
[Vessel_maxConnect, C_number,~] = Connection_Judge_3D(vessel,15,[1],0,1);
Vessel_maxConnect=(Vessel_maxConnect+TempVessel)>0;
patch(isosurface(Vessel_maxConnect,0.5),'FaceColor',[1,0,0],'EdgeColor','none');
axis([0 b 0 a 0 c]);view([270,270]);
daspect([1 1 1]);
title(['SMPM提取结果Normale',num2str(numid,'%03d')]);camlight; camlight(-80,-10); lighting phong; 
subplot(2,2,2)
imshow(imrotate(squeeze(max(ImageT,[],3)),-90),[0,400]);title('MIP\_Origin Image');
subplot(2,2,3)
imshow(imrotate(squeeze(max(ImageT,[],1)),90),[0,400]);title('MIP\_Origin Image');
subplot(2,2,4)
imshow(imrotate(squeeze(max(ImageT,[],2)),90),[0,400]);title('MIP\_Origin Image');