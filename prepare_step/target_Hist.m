%  This script is used to creat a common histogram for all MRA-TOF data
%  (brain part)
%  before this step, please make sure that you have done skull-stripping
%  step using FSL-BET
%  You can obtain more information about FSL from the website: 
%  https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation#Installing_FSL
%
%  Writed by Baochang Zhang
%  E-mail£ºbc.zhang@siat.ac.cn or 1748879448@qq.com
clc;clear;
NiiFilepath =strcat( '../data/MIDAS_109/Normal002-MRA_brain.nii.gz');
Data=load_untouch_nii(NiiFilepath);
Image=Data.img;

Mask=Image>0;
kernal= strel('sphere',3);
Mask=imclose(Mask,kernal);
Mask=imerode(Mask,kernal);
Mask = logical(Mask);
ImageT = Image.*Mask;

img = ImageT((ImageT~=0));
LengthIMG = numel(img);
Max_img = max(img(:));
[~,~,c] = size(ImageT);
figure(1);
subplot(1,2,1);imshow(imrotate(ImageT(:,:,fix(c/2)),-90),[]);
[N,X] = hist(img(:),0:Max_img); 
hc=N'/LengthIMG;
LN = length(hc);
subplot(1,2,2);
plot(1:LN,hc,'-b','LineWidth',2);hold on;
axis([0 Max_img 0 max(hc)+0.1*max(hc)]);
grid on;axis square;hold off;
Target_Hist = hc;
save('../data/Target_Hist.mat','Target_Hist');
