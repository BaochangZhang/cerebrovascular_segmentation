clear
clc
close all
processing_dataset = 3;
switch (processing_dataset)
    case 1
    %==================FOR MIDAS_109======================
    numid=1;
    MaskFilepath =strcat( '../data/MIDAS_109/Normal',num2str(numid,'%03d'),'-MRA_brain.nii.gz');
    DataFilepath =strcat( '../data/MIDAS_109/Normal',num2str(numid,'%03d'),'-MRA.nii.gz');
    [ImageT,Iout,TempVessel]= prepare_MIDASdata(DataFilepath,MaskFilepath,220);
    save(strcat('../data/mat_data_prepared/MIDAS',num2str(numid,'%03d'),'.mat'),'ImageT','Iout','TempVessel');
    disp(['save ',strcat('../data/mat_data_prepared/MIDAS',num2str(numid,'%03d'),'.mat'),'done']);
    
    case 2
    %==================FOR AVM_20======================
    numid=1;
    MaskFilepath =strcat( '../data/AVM_20/Re_AVM',num2str(numid,'%03d'),'_brain.nii.gz');
    DataFilepath =strcat( '../data/AVM_20/Re_AVM',num2str(numid,'%03d'),'.nii');
    [ImageT,Iout,TempVessel]= prepare_AVMdata(DataFilepath,MaskFilepath,780);
    save(strcat('../data/mat_data_prepared/AVM',num2str(numid,'%03d'),'.mat'),'ImageT','Iout','TempVessel');
    disp(['save ',strcat('../data/mat_data_prepared/AVM',num2str(numid,'%03d'),'.mat'),'done']);
        
    case 3
     %==================FOR GZ_10======================     
    numid=1;
    MaskFilepath =strcat( '../data/GZ_10/GZ',num2str(numid,'%03d'),'_brain.nii.gz');
    DataFilepath =strcat( '../data/GZ_10/GZ',num2str(numid,'%03d'),'.nii.gz');
    [ImageT,Iout,TempVessel]= prepare_GZdata(DataFilepath,MaskFilepath,880);
    save(strcat('../data/mat_data_prepared/GZ',num2str(numid,'%03d'),'.mat'),'ImageT','Iout','TempVessel');
    disp(['save ',strcat('../data/mat_data_prepared/GZ',num2str(numid,'%03d'),'.mat'),'done']);
end