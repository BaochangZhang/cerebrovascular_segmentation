function [ImageT,Iout,TempVessel] = prepare_MIDASdata(Image_path,brainMask_path,Empirical_value)
% This function is used to prepare the data, which will implement the
% following steps: 
% 1.obtain some high confidence vascular points
% 2.multi-scalar vessel enhanment using jermans'method
% 3.histogram matching
% 4.obtain the Image without skull
% [ImageT,Iout,TempVessel] = prepare_MIDASdata(Image_path,brainMask_path,Empirical_value)
% Input:
%      Image_path: the path of original image(datatype:nii nii.gz)
%      brainMask_path: the path of image without skull
%      Empirical_value:a hard threshold to segment the
%      vessel,i.e.Empirical_value = 220
% Output:
%     ImageT: the Image without skull after histogram matching 
%     Iout : the result of multi-scalar vessel enhancement without skull
%     TempVessel: the result after a hard threshold segmentation
% Writed by Baochang Zhang
% E-mail: bc.zhang@siat.ac.cn or 1748879448@qq.com

%   load data(original image and brain mask)    
    Data=load_untouch_nii(Image_path);
    Image=double(Data.img);
    
    Data=load_untouch_nii(brainMask_path);
    Mask=double(Data.img);
    Mask(Mask>0)=1;
    
    kernal= strel('sphere',3);
    Mask=imclose(Mask,kernal);
    Mask=imerode(Mask,kernal);
    Mask = logical(Mask);

    ImageT = Image.*Mask;
    
    TempVessel=ImageT>Empirical_value;
    
    figure;
    set(gca,'color','white'); 
    [a,b,c] = size(TempVessel);
    [Vessel_maxConnect, ~,~] = Connection_Judge_3D(TempVessel, 0,[],200,3);
    patch(isosurface(Vessel_maxConnect,0.5),'FaceColor',[1,0,0],'EdgeColor','none');
    axis([0 b 0 a 0 c]);view([270,270]);daspect([0.8,0.8,0.4297]);
    title('经验阈值提取结果');camlight; camlight(-80,-10); lighting phong; 

    [TempVessel, ~,~] = Connection_Judge_3D(TempVessel, 0,[],200,3);

    I = Image - min(Image(:));
    I = I / prctile(I(I(:) > 0.5 * max(I(:))),90);
    I(I>1) = 1;
    Iout = vesselness3D(I, 1:2, [1;1;1], 0.7, true);
    Iout=Iout.*Mask;

    g_keranl = fspecial3('gaussian',3,0.4);
    FImage=imfilter(Image,g_keranl).*Mask;

    % 加载母（直方图）分布
    load('../data/Target_Hist.mat','Target_Hist')
    [Trans_Image] = Hist_match3D(Target_Hist,FImage);
    ImageT = Trans_Image.*Mask;
end

