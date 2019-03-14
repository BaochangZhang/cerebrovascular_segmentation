function [Dx,vessel,vessel_ratio] = GMM_MRF_SEMI(IMG,Iout,K,Object,SelecNum,NB,gama,IL,iterations,VesselPoints)
% This function is used to extract cerebral vessels from TOF-MRA, which
% will implement the following steps: 
% A.Gaussian mixture model to fit the histogram curve and solve the
%   parameters using semi-EM algorithm
% B.Markov random filed(MRF) to optimize the initial segmentation, which
%   integrating the result of multi-scalar vessel enhancement and the
%   initial segmentation
%
% [Dx,vessel,vessel_ratio] = GMM_MRF_SEMI(IMG,Iout,K,Object,SelecNum,NB,gama,IL,iterations,VesselPoints)
% Input:
%      IMG:     original image without skull
%      Iout:	the result of multi-scalar vessel enhancement without skull
%      K :      the total class for K-mean algorithm. This function divides
%               the image to three class (i.e. the first class is 
%               Cerebrospinal fluid regin; the second class is gray and
%               white matter region; the third class is vessel region. K=3
%      Object:  the index of the vessel target, this function defines the
%               the index of vessel-class as 3. Object = 3
%      SelecNum:A list,the values of list are some indexes ofconnection
%               region, which will decide the connection preporty of
%               segmentation result.
%      NB:      NB=6, the number of neighborhoods that are considered in
%               MRF
%      gama:    parameter that controls segmentation uniformity
%               - lower gama -> more voxels are segmented as vessel
%               gama usually is set to be 1.0
%      IL:      iterations of MRF
%      iterations: The iterations of EM
%      VesselPoints: The initial vessel points, which is used in the
%                    semi-EM algorithm.
% Output:
%     Dx: the whole segmentation result, which contain all the connetctions
%     vessel : the segmentation with SelecNum connections
%     vessel_ratio: the ratio between the vessel and the cerebral region(data space
%                   without skull)
%
% Writed by Baochang Zhang
% E-mail: bc.zhang@siat.ac.cn or 1748879448@qq.com


Iout = double(Iout);
IMG = double(IMG);
[VBN_EM,criValue] = SegParameter_MRA_pdf_curl(IMG,Iout,K,VesselPoints,iterations);
VBN = VBN_EM;
vessel_ratio = VBN_EM(3,3);
[flagIMG] = GetRegion(Iout,gama*criValue,[1:40]); % ��ȡ����ֲ��ռ�flagIMG,criValueΪ�ٽ���ֵ/ϣ������ռ��㹻�Ĵ�Ӷ���������Ѫ�ܡ�
[Dx_MLE,sort_D] = ML_estimation(IMG,VBN,Object,flagIMG);  % ������Ȼ���Ƶĳ�ʼ��ǳ�, sort_D�а�����ԭʼ�ĸ�����Ϣ

figure;
set(gca,'color','black'); 
[a,b,c] = size(Dx_MLE);
patch(isosurface(Dx_MLE,0.5),'FaceColor',[1,0,0],'EdgeColor','none');
axis([0 b 0 a 0 c]);view([270,270]);
daspect([0.8,0.8,0.4297]);
%daspect([1,1,1]);
title('Ѫ�ܺ�����Ϻ�Ľ��');camlight; camlight(-80,-10); lighting phong; 

inPLX = sort_D(:,4);%��������
pxl_k = sort_D(:,1:3);
figure(8);imshow_Process(IMG,Iout,Dx_MLE);pause(0.5);
% figure(11);imshow3D_patch(flagIMG,flagIMG,[0.5 0.5 0.5]);title('Ѫ�ֲܷ���ʼ�ռ�');pause(1);% ��ʾѪ�ܵ�3D��ʼ�ռ�
disp(['��ѡ�ռ�������Ϊ' num2str(length(find(flagIMG==1))) '�� �ܿ���������Ϊ' num2str(numel(IMG(IMG>0))) '����ѡ�ռ����Ϊ' num2str(100*length(find(flagIMG==1))/numel(IMG(IMG>0))) '%']);
% ��������Ǹ�����������Ҫ�����Ż�
OptiumBeTa = (NB==6)*0.7 +(NB==0)*0.7 + (NB==26)*0.19;
disp(['OptiumBeTa = ' num2str(OptiumBeTa)]);

figure(9);% �ڽز�ͼ������ʾ�������
Dx_init = Dx_MLE;
for t = 1:IL
    tic;
    disp(['BeTa = ' num2str(OptiumBeTa) ';' 'ICM iteration ' num2str(t) ' times...']); 
    %Dx = ICM_estimation(VBN,pxl_k,inPLX,Object,Dx_init,OptiumBeTa,NB);
    Dx = ICM_estimation(VBN,pxl_k,inPLX,Object,Dx_init,OptiumBeTa,NB,Iout,criValue);
    subplot(1,IL,t);imshow(Dx(:,:,fix(c/2)),[]); pause(0.2);
    title(['MRF-ICM����' num2str(t) '��' ]); 
    Dx_init = Dx;
    ti = toc;disp(['Iteration runtime = ' num2str(ti)]);
end
vessel=OveralShows(SelecNum,Dx,Dx_MLE,flagIMG,Object);
disp('----------- All FINISHED -------------------------');

function [VBN_EM,criValue] = SegParameter_MRA_pdf_curl(IMG,Iout,K,labeledPs,iterations)

% �������壺��ʾCTCAֱ��ͼ��K��ֵ�����ࡢ�������ٷֱȡ���K��ֵ����ǰ��������EM����ȷ���Ʋ���
% Ŀ������ΪMRF�ָ��ṩ��ȷ����mu��sigma��w
% ��ʾ�ز�ͼ���������ס��ͼfigure(1)�������������ͼfigure(2)��������������ͼfigure(3)
% threthold = theta * criValue,criValueΪ�ٽ���ֵ
% close all
%��ʾͼ��ֱ��ͼ
img = IMG((IMG~=0));
OriginId=find(IMG~=0);
LengthIMG = numel(img);
Max_img = max(img(:));
[~,~,c] = size(IMG);
figure(1);
subplot(1,2,1);imshow(imrotate(IMG(:,:,fix(c/4)),-90),[]);
[N,X] = hist(img(:),0:Max_img); 
%��ʾֱ��ͼ�ϵļ���
[Imax,Imin,N2] = peaks_Histogram(N);
hc = N2'/LengthIMG;
LN = length(hc);
subplot(1,2,2);
plot(1:LN,hc,'-b','LineWidth',2);hold on % ��ʾֱ��ͼ����
plot(Imax,hc(Imax),'*r','MarkerSize',3);
plot(Imin,hc(Imin),'ob','MarkerSize',3);
axis([0 400 0 max(hc)+0.1*max(hc)]);
xlabel('Intensity');ylabel('Frequency')
grid on;axis square;hold off;pause(0.5);
% disp(num2str([Imax(1) Imin Imax(2)]));
% K��ֵ���࣬����������ֵ K_mu��������K_var�ٷֱ�K_percent
tic;
disp('kmeans...')
Imax=Imax(length(Imax));
[idx,ctrs] = kmeans(img(:),K,'start',[Imax(1)*2/8; Imax(1); 300]); %MIDAS
[Idx,Ctrs] = Kmean_reorder(idx,ctrs);% ���ջҶ��������ɵ����ߵ�˳���������idx��ctrs

% ��ʾkmeans������Ѫ�ܵľ�����
ClusterVessel=zeros(size(IMG));
ClusterVessel(OriginId(find(Idx(:)==K)))=1;
figure(2);
set(gca,'color','black'); 
[a,b,c] = size(ClusterVessel);
patch(isosurface(ClusterVessel,0.5),'FaceColor',[1,0,0],'EdgeColor','none');
axis([0 b 0 a 0 c]);view([270,270]);daspect([0.8,0.8,0.4297]);%daspect([0.8,0.8,0.4297]);
title('kmeans������Ѫ�ܵľ�����')
camlight; camlight(-80,-10); lighting phong; pause(1);

K_mu = Ctrs;
K_var = zeros(K,1);
K_sigma = zeros(K,1);
Omega = zeros(K,1);
MG_curl = zeros(K,LN);
figure(3);
plot(1:LN,hc,'-k','LineWidth',1.5);% ��ʾֱ��ͼ����
axis([0 400 0 max(hc)+0.1*max(hc)]);grid on;hold on;
flag = {'-.c';'-.m';'-g';'-.b';'-r';'-k';'-.k';'-.y';'--k';':k';':g'};
for i = 1:K % ������������K_var���ٷֱ�K_percent�����˹����ͼ
    Omega(i) = length(find(Idx==i))/LengthIMG;% ������ֲ����ߵ����ֵ
    K_var(i) = var(img(Idx==i));
    K_sigma(i) = sqrt(K_var(i));
    MG_curl(i,:) = (Omega(i)*(1/sqrt(2*pi)/K_sigma(i)).*exp(-(X-K_mu(i)).^2/(2*K_var(i))));% ��������˹����ģ��Ŀ������   
    plot(1:LN,MG_curl(i,:),char(flag(i)),'LineWidth',1);%���Ƹ�����ֲ�����
end
t = toc; disp(['using ' num2str(t) '��']);
legend_char = cell(K+2,1);
legend_char{1} = char('Original histogram');
for i = 1:K % �༭legend
        legend_char{1+i} = char(['Gaussian curl-line' num2str(i) ': lamit=' num2str((K_mu(i)))...
          ' w=' num2str(Omega(i))]);
end
plot(1:LN,sum(MG_curl,1),'--r','LineWidth',1);% ��ʾ��Ϻ������
legend_char{K+2} = char('Init-fitting histogram');
legend(legend_char{1:K+2});
xlabel('Intensity');
ylabel('Frequency');
title('��ʾ��ʼ״̬��ֱ��ͼ����')
hold off

VBN_Init = [K_mu(1)  K_sigma(1) Omega(1);
            K_mu(2)  K_sigma(2)  Omega(2);
            K_mu(3)  K_sigma(3)  Omega(3)];
        
VBN_Rect=VBN_Init;

%%%%%%%%%%%%%%���������������ȷ�������ϸ�����K_mean��K_sigma��K_percent
disp('GMM_EM...');tic;
if isempty(labeledPs)
    [VBN_EM, SumError] = GMM_EM(IMG,VBN_Rect,iterations,0);
else
    [VBN_EM, SumError] = Semi_GMM_EM(IMG,VBN_Rect,iterations,0,labeledPs);%����ԭʼ��ߴ�����IMG���㾫ȷ���� %RGMM_EM(IMG,VBN_Init,1000,1);RGMM_EM(IMG,VBN_Rect,500,0)
end
    % disp(['Finished, the curl_lines fitting error after EM step is: ' num2str(minError)]);
EM_mu = zeros(K,1);
EM_var = zeros(K,1);
EM_sigma = zeros(K,1);
Omega = zeros(K,1);
MG_curl = zeros(K,LN);

figure(6);
plot(1:LN,hc,'-k','LineWidth',1.5);% ��ʾֱ��ͼ����
axis([0 400 0 max(hc)+0.1*max(hc)]);grid on;hold on;
for i = 1:K % ��������ֵK_mean��������K_sigma�ٷֱ�K_percent
    EM_mu(i) = VBN_EM(i,1);
    Omega(i) = VBN_EM(i,3);% ������ֲ����ߵ����ֵ
    EM_var(i) =  VBN_EM(i,2)^2+eps(1);
    EM_sigma(i) = VBN_EM(i,2)+eps(1);
    MG_curl(i,:) = Omega(i)*(1/sqrt(2*pi)/EM_sigma(i)).*exp(-(X-EM_mu(i)).^2/(2*EM_var(i))); 
    plot(1:LN,MG_curl(i,:),char(flag(i)),'LineWidth',1);
end
t = toc; disp(['using ' num2str(t) '��']);
legend_char = cell(K+2,1);
legend_char{1} = char('Original histogram');
for i = 1:K % % �༭legend
       legend_char{1+i} = char(['EM Gaussian curl-line ' num2str(i) ': mu=' num2str(uint16(EM_mu(i)))...
           ' sigma=' num2str(uint16(EM_sigma(i))) ' w=' num2str(Omega(i))]);
end
plot(1:LN,sum(MG_curl,1),'--r','LineWidth',1);% ��ʾ��Ϻ������
legend_char{K+2} = char('EM fitting histogram');
legend(legend_char{1:K+2});
xlabel('Intensity');
ylabel('Frequency');
title('��ʾ���֮���ֱ��ͼ')
hold off
pause(0.5);
[criValue,~] = Iout_vessel_perscent(IMG,Iout,1.5*Omega(3)); % FOR MIDAS ��ȡEM���º��Ѫ��-�ٽ���ֵ/����ռ䡣
%[criValue,~] = Iout_vessel_perscent(IMG,Iout,Omega(3)); % FOR GZ ��ȡEM���º��Ѫ��-�ٽ���ֵ/����ռ䡣
%----���Kmeans�Ĳ������ƽ��
VBN_Initshow = VBN_Init;
disp('VBN_Init =');disp(num2str(VBN_Initshow));
disp(['VBN_Init by KmeansSize: [' num2str(size(IMG)) ']']);
%----���������Ĳ������ƽ��
% VBN_Rectshow = VBN_Rect;
% disp('VBN_Rect =');disp(num2str(VBN_Rectshow));
% disp(['VBN_Init by RectSize: [' num2str(size(IMG)) ']']);
%----�����������������ƽ��
VBN_EM_show = VBN_EM; 
disp('VBN_EM =');disp(num2str(VBN_EM_show));
disp(['VBN_EM by EMSize: [' num2str(size(IMG)) ']']);
%---------------
disp(['All over. The EM Fitting error is ' num2str(SumError)]);

function [Imax,Imin,N2] = peaks_Histogram(N)%======���ԸĽ�һ�¡�
% ����3D-CTCAֱ��ͼ���ߵķ�ֵ��͹�ֵ��Imax,Imin
N2 = smooth(N,32,'loess'); %21
LN = length(N2);
DN2 = diff([N2;N2(LN)]);
n1 = 0;
n2 = 0;
I_max=[];
I_min=[];
I_MIN=[];
for i = 5:200 % �ӵ�10���㿪ʼ 200=>length(N2)-2
    if ((DN2(i-1)>0) && (DN2(i+1)<0)) && N2(i)>max(N2(i-1),N2(i+1)) && N2(i)>10
       n1 = n1+1;
       I_max(n1) = i;
    end
    if ((DN2(i-1)<0) && (DN2(i+1)>0)) && N2(i)<min(N2(i-1),N2(i+1))
       n2 = n2+1;
       I_min(n2) = i;
    end 
end
% �ڸ�����ֵ��I_max֮���ҳ����ŵĹȵ�
n_max = length(I_max); % ��ֵ����
for k = 1:n_max
    if k ~= n_max
       nums = find(I_min>I_max(k) & I_min<I_max(k+1));
       [~,m] = min(N2(I_min(nums)));
       I_MIN(k) = I_min(nums(m));
    end
    if k == n_max && ~isempty(find(I_min>I_max(k)))
       nums = find(I_min>I_max(k));
       [~,m] = min(N2(I_min(nums)));
       I_MIN(k) = I_min(nums(m));
    end
end
Imax = I_max;
Imin = I_MIN;

function [Idx,Ctrs] = Kmean_reorder(idx,ctrs)
% �������壺��idx��ctrs�е����ݰ���ctrs��С�����˳����������
K = length(ctrs);
Ctrs_index = [ctrs,(1:length(ctrs))'];
sort_Ctrs = sortrows(Ctrs_index,1);% sort_Ctrs(:,1)�ɵ����ߴ�ž������ģ�sort_Ctrs(:,2)��Ŷ�Ӧ��ԭʼ����k
K_index = cell(1,K);
for k = 1:K % ��idx��ԭʼ�����k�ֱ�洢��K_index�ṹ��
    K_index{k} = find(idx==k);
end
Idx = zeros(size(idx));
Ctrs = zeros(size(ctrs));
for i = 1:K
    Ctrs(i) = sort_Ctrs(i,1);
    Idx(K_index{sort_Ctrs(i,2)}) = i;
end

function [criValue,paraV] = Iout_vessel_perscent(IMG,Iout,percentEM)
tic;
disp('adjusting the optiumal parameters for vessel-class ...');
[y,x,z] = size(Iout);
b = IMG>0;
b=sum(b(:));
N = 50;
Iout =Iout/max(Iout(:));%��һ��Ϊ[0,1]
thre_value = linspace(0.01,0.08,N);
ratio = zeros(N,1);
for i = 1:N
    Gi = GetRegion(Iout,thre_value(i)); % �����ֵthre_value(i)�µ����Ѫ�ܷ�֧��
    ratio(i) =sum(Gi(:))/b;
end
[~,numr] = min(abs(ratio-percentEM));
h=figure(7);
close(h);
figure(7);
subplot(1,2,1);plot(thre_value,ratio,'-b',thre_value,ratio(numr)*ones(1,N),'-.r',thre_value(numr),ratio(numr),'*r');
xlabel('threshold values');ylabel('ratios')
legend('ratio curve of cerebral vessel to head volume','ratio given by prior knowledge');
axis([min(thre_value) max(thre_value) 0 max(ratio)])

criValue = thre_value(numr);                 % criValue��Ϊ�ٽ�㣬����������Ѫ�ܵĳ�ʼ����,��������gama*criValue����Ѫ����̽�ռ�
Stemp = GetRegion(Iout,criValue,[1,2]);   % Ѫ�ܳ�ʼ�ռ�    
IMG_mu4 = mean(IMG(Stemp>0));                  % IMG�е�Ѫ�ܾ�ֵ
IMG_sigma4 = std(IMG(Stemp>0));                % IMG�е�Ѫ�ܱ�׼��
paraV = [IMG_mu4,IMG_sigma4];

% ��ʾ'w4'��Ӧ�����Ѫ�ܷ�֧��
Gout = GetRegion(Iout,thre_value(numr));% ���'w4'��Ӧ��Ѫ�ܷ�֧��
t = toc;
disp(['adjusting the optiumal parameters for vessel-class run time ' num2str(t) ' s']);
subplot(1,2,2);patch(isosurface(Gout,0.5),'FaceColor','r','EdgeColor','none');
axis([0 x 0 y 0 z]);view([270 270]); daspect([1,1,1]);
camlight; camlight(-80,-10); lighting phong; pause(1); 
title('thresholding the multi-scale filtering response at the appointed ratio');

function [Gout,MaxLength] = GetRegion(Iout,threthold,SelecNum)
% ��Iin��ֵ������ȡ����Ŀ��飬���������ҳ��������Max_num�������������Iout����ֵ����
% T��ȡ����Ĵ�������k����ȡk-1��ѡʣ�Ľ��
% TtΪ[1 2 ... T]�ĳ�Ա���飬��������ȡĿ�������
Iout=Iout/max(Iout(:));
J = Iout>threthold;
sIdx  = regionprops(J,'PixelIdxList');      % ��ȡJ������Ŀ������������������
num_sIdx = zeros(length(sIdx),1);           % ���sIdx���ȵľ���num_sIdx
Gout = zeros(size(J));                      % �½�����ͬ�ߴ�ı�Ǿ���Iout
for i = 1:length(sIdx)
    num_sIdx(i) = length(sIdx(i).PixelIdxList);
end
if nargin ==2                               % ������ǰ��������ʱ
   [~,Max_num] = max(num_sIdx);             % �������num_sIdx�е���󳤶����
   MaxPatch = sIdx(Max_num).PixelIdxList;
   Gout(MaxPatch) = 1;                      % �������
   MaxLength = length(MaxPatch);            % �������ĳ���
   return;
end
MaxLength = [];
maxSN = max(SelecNum);                      % ѡ��������������
for t = 1:maxSN                             % Ѱ��ָ������
    [~,Max_num] = max(num_sIdx);            % ����ĳ���    
    num_sIdx(Max_num)=0;                    % ��num_sIdx���ҵ�������ĳ�����0���Ա����Ѱ�Ҵδ�ֵ            
   if ismember(t,SelecNum)                  % ���Ϊָ�������飬�����֮
      MaxPatch = sIdx(Max_num).PixelIdxList;
      Gout(MaxPatch) = 1;    % �γ������Ȼ�ռ䣨����1����
   end
end

function [VBN, SumError] = GMM_EM(IMG,Init,upNum,Flag)
% Flag=1:��ʾ������������ͼfigure(4��5)��������ʾ
% upNum:����������ֵ
% Init������֮ǰVBN�ĳ�ʼ������
% RGMM_EM(IMG,VBN_Rect,500,0,labeledPs);

K = size(Init,1);
N = length(find(IMG(:)>0));
IMG_max = max(IMG(:));
[fc,xi] = hist(IMG(IMG>0),0:IMG_max);
L = length(0:IMG_max);

FC = M_Extand(fc,K,L);
XI = M_Extand(xi,K,L);

Mu = zeros(K,upNum+1);
Var = zeros(K,upNum+1);
W = zeros(K,upNum+1);
Error = zeros(1,upNum);

Mu(:,1) = Init(:,1);
Var(:,1) = Init(:,2).^2;
W(:,1) = Init(:,3);

for i = 1:upNum
    [plx,pxl] = pLX(0:IMG_max,K,Mu(:,i),Var(:,i),W(:,i));
    W(:,i+1) = (1./sum(FC,2)).*sum(FC.*plx,2);
    Mu(1:K-1,i+1) = sum(XI(1:K-1,:).*FC(1:K-1,:).*plx(1:K-1,:),2)./sum(FC(1:K-1,:).*plx(1:K-1,:),2);% ����(1:K-1,:)��Ӧ�ĸ�˹��ֵ
    Mu(K,i+1) = sum(XI(K,:).*FC(K,:).*plx(K,:),2)./sum(FC(K,:).*plx(K,:),2);% ����(1:K-1,:)��Ӧ�ĸ�˹��ֵ
%     Mu(K,i+1) = Mu(K,i);% ����K��Ӧ�ĸ�˹��ֵ
    MU = M_Extand(Mu(:,i+1),K,L);
    Var(1:K,i+1) = sum((XI(1:K,:)-MU(1:K,:)).^2.*FC(1:K,:).*plx(1:K,:),2)./sum(FC(1:K,:).*plx(1:K,:),2);
    Error(i) = sum(abs(sum(pxl,1)-fc/N));
%     if i>2
%         if  Error(i)- Error(i-1)>0
%             disp(['begin going worse iteration at iteration:',num2str(i)]);
%             break;
%         end 
%     end
end
realupNum=i;
dL=realupNum;
VBN = [Mu(:,realupNum+1) sqrt(Var(:,realupNum+1)) W(:,realupNum+1)];

if Flag==1
figure(4);
legend_char = cell(K+1,1);
subplot(1,3,1);plot(1:dL,Mu(:,1:dL));
for k = 1:K
    if k==1
       legend_char{k} = char(['beta updates from ' num2str(sqrt(Mu(k,1))) ' to ' num2str(sqrt(Mu(k,dL)))]);
    else
       legend_char{k} = char(['mu' num2str(k-1) ' updates from ' num2str(Mu(k,1)) ' to ' num2str(Mu(k,dL))]);
    end
end
legend(legend_char{1:K});
axis([0 dL+1 0 1.5*max(Mu(:))+20]);
xlabel('Times of EM iteration');
ylabel('Mean of each classification')

subplot(1,3,2);plot(1:dL,sqrt(Var(:,1:dL)));
for k = 2:K
    legend_char{k} = char(['sigma' num2str(k-1) ' updates from ' num2str(sqrt(Var(k,1))) ' to ' num2str(sqrt(Var(k,dL)))]);
end
legend(legend_char{2:K});
axis([0 dL+1 0 1.5*max(sqrt(Var(:)))+10]);
xlabel('Times of EM iteration');
ylabel('Sigma of each classification')

subplot(1,3,3);plot(1:dL,W(:,1:dL));
for k = 1:K
    legend_char{k} = char(['w' num2str(k) ' updates from ' num2str(fix(100*W(k,1))/100) ' to ' num2str(fix(100*W(k,dL))/100)]);
end
legend(legend_char{1:K});
axis([0 dL+1 0 1.2]);
xlabel('Times of EM iteration');
ylabel('Weight of each classification')

figure(5);
plot(1:dL,Error(1:dL));
axis([0 dL 0 max(Error)]);
xlabel('Times of EM iteration');
ylabel('MSE of the parameters between neiboring iteration')

end
SumError = Error(dL);

function [VBN, SumError] = Semi_GMM_EM(IMG,Init,upNum,Flag,VlabelPs)
% Flag=1:��ʾ������������ͼfigure(4��5)��������ʾ
% upNum:����������ֵ
% Init������֮ǰVBN�ĳ�ʼ������
% RGMM_EM(IMG,VBN_Rect,500,0,labeledPs);

K = size(Init,1);
N=length(find(IMG(:)>0));%N = length(B(:));
IMG_max = max(IMG(:));
[fc,xi] = hist(IMG(IMG>0),0:IMG_max);
L = length(0:IMG_max);
sizeImg = size(IMG);
Vindex = sub2ind(sizeImg,VlabelPs(:,1),VlabelPs(:,2),VlabelPs(:,3));
Vhu= IMG(Vindex);
[Vfc,Vxi] = hist(Vhu,0:IMG_max);
Li = [0 0 sum(Vfc)]'; %����Ѫ����ı������

FC = M_Extand(fc-Vfc,K,L);
XI = M_Extand(xi,K,L);

Mu = zeros(K,upNum+1);
Var = zeros(K,upNum+1);
W = zeros(K,upNum+1);
Error = zeros(1,upNum);
Mu(:,1) = Init(:,1);
Var(:,1) = Init(:,2).^2;
W(:,1) = Init(:,3);
for i = 1:upNum
    %plxwΪ������� pxl Ϊ�������
    [plx,pxl] = pLX(0:IMG_max,K,Mu(:,i),Var(:,i),W(:,i)); % plxΪ������ʣ�plxΪW*��Ȼ����
    W(:,i+1) = (1./(sum(FC,2)+Li)).*(sum(FC.*plx,2)+Li); 
    Mu(1:K-1,i+1) = sum(XI(1:K-1,:).*FC(1:K-1,:).*plx(1:K-1,:),2)./sum(FC(1:K-1,:).*plx(1:K-1,:),2);% ����(1:K-1,:)��Ӧ�ĸ�˹��ֵ
    Mu(K,i+1) =  (sum(XI(K,:).*FC(K,:).*plx(K,:),2)+sum(Vxi.*Vfc,2))./(sum(FC(K,:).*plx(K,:),2)+Li(K));% ����K��Ӧ�ĸ�˹��ֵ
    %Mu(K,i+1) = Mu(K,i);% �̶�Ѫ�ܵĸ�˹�ֲ���ֵ
    MU = M_Extand(Mu(:,i+1),K,L);
    Var(1:K-1,i+1) = sum((XI(1:K-1,:)-MU(1:K-1,:)).^2.*FC(1:K-1,:).*plx(1:K-1,:),2)./sum(FC(1:K-1,:).*plx(1:K-1,:),2);
    Var(K,i+1) = (sum((XI(K,:)-MU(K,:)).^2.*FC(K,:).*plx(K,:),2)+sum((Vxi-MU(K,:)).^2.*Vfc,2))./(sum(FC(K,:).*plx(K,:),2)+Li(K));
    Error(i) = sum(abs(sum(pxl,1)-fc/N));
    if i>2
        if  Error(i)- Error(i-1)>0
            disp(['begin going worse iteration at iteration:',num2str(i)]);
            break;
        end 
    end
end
realupNum=i;
dL=realupNum;
VBN = [Mu(:,realupNum+1) sqrt(Var(:,realupNum+1)) W(:,realupNum+1)];

%== Test ==
% [fc,xi] = hist(IMG(IMG>0),133:IMG_max);
% fc=fc(2:end);
% xi=xi(2:end);
% L = length(xi);
% sizeImg = size(IMG);
% Vindex = sub2ind(sizeImg,VlabelPs(:,1),VlabelPs(:,2),VlabelPs(:,3));
% Vhu= IMG(Vindex);
% [Vfc,Vxi] = hist(Vhu,133:IMG_max);
% Vfc=Vfc(2:end);
% Vxi=Vxi(2:end);
% Li = [0 0 sum(Vfc)]'; %����Ѫ����ı������
% 
% FC = M_Extand(fc-Vfc,K,L);
% XI = M_Extand(xi,K,L);
% 
% Mu = zeros(K,upNum+1);
% Var = zeros(K,upNum+1);
% W = zeros(K,upNum+1);
% Error = zeros(1,upNum);
% Mu(:,1) = VBN(:,1);
% Var(:,1) = VBN(:,2).^2;
% W(:,1) = VBN(:,3);
% for i = 1:upNum
%     %plxwΪ������� pxl Ϊ�������
%     [plx,pxl] = pLX(134:IMG_max,K,Mu(:,i),Var(:,i),W(:,i)); % plxΪ������ʣ�plxΪW*��Ȼ����
%     W(:,i+1) = (1./(sum(FC,2)+Li)).*(sum(FC.*plx,2)+Li); 
%     Mu(1:K-1,i+1) = sum(XI(1:K-1,:).*FC(1:K-1,:).*plx(1:K-1,:),2)./sum(FC(1:K-1,:).*plx(1:K-1,:),2);% ����(1:K-1,:)��Ӧ�ĸ�˹��ֵ
%     Mu(K,i+1) =  (sum(XI(K,:).*FC(K,:).*plx(K,:),2)+sum(Vxi.*Vfc,2))./(sum(FC(K,:).*plx(K,:),2)+Li(K));% ����K��Ӧ�ĸ�˹��ֵ
%     %Mu(K,i+1) = Mu(K,i);% �̶�Ѫ�ܵĸ�˹�ֲ���ֵ
%     MU = M_Extand(Mu(:,i+1),K,L);
%     Var(1:K-1,i+1) = sum((XI(1:K-1,:)-MU(1:K-1,:)).^2.*FC(1:K-1,:).*plx(1:K-1,:),2)./sum(FC(1:K-1,:).*plx(1:K-1,:),2);
%     Var(K,i+1) = (sum((XI(K,:)-MU(K,:)).^2.*FC(K,:).*plx(K,:),2)+sum((Vxi-MU(K,:)).^2.*Vfc,2))./(sum(FC(K,:).*plx(K,:),2)+Li(K));
%     Error(i) = sum(abs(sum(pxl,1)-fc/N));
% %     if i>2
% %         if  Error(i)- Error(i-1)>0
% %             disp(['begin going worse iteration at iteration:',num2str(i)]);
% %             break;
% %         end 
% %     end
% end
% 
% realupNum=i;
% dL=realupNum;
% VBN = [Mu(:,realupNum+1) sqrt(Var(:,realupNum+1)) W(:,realupNum+1)];
%====

if Flag==1
figure(4);
legend_char = cell(K+1,1);
subplot(1,3,1);plot(1:dL,Mu(:,1:dL));
for k = 1:K
       legend_char{k} = char(['mu' num2str(k-1) ' updates from ' num2str(Mu(k,1)) ' to ' num2str(Mu(k,dL))]);
end
legend(legend_char{1:K});
axis([0 dL+1 0 1.5*max(Mu(:))+20]);
xlabel('Times of EM iteration');
ylabel('Mean of each classification')

subplot(1,3,2);plot(1:dL,sqrt(Var(:,1:dL)));
for k = 1:K
    legend_char{k} = char(['sigma' num2str(k-1) ' updates from ' num2str(sqrt(Var(k,1))) ' to ' num2str(sqrt(Var(k,dL)))]);
end
legend(legend_char{1:K});
axis([0 dL+1 0 1.5*max(sqrt(Var(:)))+10]);
xlabel('Times of EM iteration');
ylabel('Sigma of each classification')

subplot(1,3,3);plot(1:dL,W(:,1:dL));
for k = 1:K
    legend_char{k} = char(['w' num2str(k) ' updates from ' num2str(fix(100*W(k,1))/100) ' to ' num2str(fix(100*W(k,dL))/100)]);
end
legend(legend_char{1:K});
axis([0 dL+1 0 1.2]);
xlabel('Times of EM iteration');
ylabel('Weight of each classification')

figure(5);
plot(1:dL,Error(1:dL));
axis([0 dL 0 max(Error)]);
xlabel('Times of EM iteration');
ylabel('MSE of the parameters between neiboring iteration')
end
SumError = Error(dL);

function [D] = M_Extand(Vector,K,L)
% size(Vector) = [K,1] or [1,L]
% ��Vector��չΪ����D��size(D)=[K,L]
D = zeros(K,L);
[a,b] = size(Vector);
if a>1 && b==1 %���Vector��һ��ʸ��K��1
   for j = 1:L
       D(:,j) = Vector;
   end    
end
if a==1 && b>1 %���Vector��һ��ʸ��1��L
    for i = 1:K
        D(i,:) = Vector;
    end
end

%************** �Ӻ�������������,f(k|xi) = wk*f(xi|k)/��j=1:K(wj*f(xi|j))*********
function [plx,pxl] = pLX(xi,K,Mu,Var,W)
% ���������ʾ���plx
pxl = zeros(K,length(xi));% ��ʼ��[w*�������ʾ���]
plx = zeros(K,length(xi));% ��ʼ��������ʾ���
Var = Var + eps(1);       % ʹ�õ�һ�з���Ϊ�����������С��������1/sqrt(2*pi*Var(k))ΪNaN
for k = 1:K               % ���������������ʾ���
    pxl(k,:) =W(k)*(1/sqrt(2*pi*Var(k)))*exp(-(xi-Mu(k)).^2./(2*Var(k)));
end
Sum_pxl = sum(pxl,1)+eps(1);
for k = 1:K
    plx(k,:) = pxl(k,:)./Sum_pxl;
end

function [Dout,sort_D] = ML_estimation(A,VBN,Object,flagIMG)
% �������壺���м��������Ȼ����
% A,:ԭʼ���ݣ�����������Ѫ�ܺͱ���������ݣ�
% VBN�������ֵ�ͷ��
% W��������ϵ��
% sort_D:��Ȼ���ʺ�ͷ������IndexA�ĺϳɾ�����length(IndexA)�У�Object+1��
% ML_estimation(IMG,VBN,Object,flagIMG); 
tic;
disp('ML_estimation ...');
Dout = zeros(size(A));
IndexA = find(flagIMG~=0);%ȡ��flagIMG���Ϊ1����������
N = numel(IndexA);
L = size(VBN,1);
A = repmat(A(IndexA)',L,1);                                                % A���ΪL��1�е�A(:)'����
mu = repmat(VBN(:,1),1,N);                                                 %����L��Ni�е�mu����
sigma = repmat(VBN(:,2),1,N);                                              %����L��Ni�е�sigma����
W = repmat(VBN(:,3),1,N);                                                  %����L��Ni�е�W����
Li = find(1:L~=Object)';                                                   % ����1��K-1��
pxl_k = [W(1:L,:).*(1./sqrt(2*pi*sigma(1:L,:).^2)).*exp(-(A(1:L,:)-mu(1:L,:)).^2./(2*sigma(1:L,:).^2))]; % ��ȡÿ�������������
plx_k=zeros(size(pxl_k));
Sum_pxl = sum(pxl_k,1)+eps(1);
for k = 1:L
    plx_k(k,:) = pxl_k(k,:)./Sum_pxl; %��ȡÿ����ĺ������
end
%���иĽ�����ȡÿ�����Ѫ�ܺ�����ʣ���Ϊ�������ݵ����ţ������д��󡣱�Ҷ˹�б��Ǻ���������
DMPL_vector = Object*((pxl_k(Object,:)./W(Object,:))>(sum(pxl_k(Li,:),1)./sum(W(Li,:),1)));%��һ�֣�Ȩƽ��Լ��Ŀ��ͱ�����
%DMPL_vector = Object*(pxl_k(Object,:)>max(pxl_k(Li,:),[],1));%�ڶ��֣�Ѫ������ڱ���������ֵ
%DMPL_vector = Object*(pxl_k(Object,:)>sum(pxl_k(Li,:),1)/(L-1));%�����֣�Ѫ������ڱ�����ľ�ֵ,��������һ���㷨Ч������
Dout(IndexA)= DMPL_vector';%��ʸ�����������ֵ
index_PLX_object = [pxl_k' IndexA];% ���flagIMG==1�Ŀռ����أ������������飬��һ����pxl_k(Object)���ڶ��������صı��
sort_D = sortrows(index_PLX_object,-3);%����Ŀ���ࣨpxl_k��:,4�����ɸ����͵�˳���������
t = toc;
disp(['finished, time consumption of this step is ' num2str(t) ' s']);

function imshow_Process(IMG,Iout,Dx_init)
c = size(IMG,3);
subplot(1,4,1);imshow(imrotate(IMG(:,:,fix(c/2)),-90),[]);title('Original Image');
subplot(1,4,2);imshow(imrotate(squeeze(max(IMG,[],3)),-90),[]);title('MIP of Original Image');
subplot(1,4,3);imshow(imrotate(squeeze(max(Iout,[],3)),-90),[]);title('MIP after Vessel Enhance');
subplot(1,4,4);imshow(imrotate(Dx_init(:,:,fix(c/2)),-90));title('ML_estimation');

%**************** SubFunction ICM_estimation **********************************
function [Dnew] = ICM_estimation(VBN,pxl_k,Ni,Object,D,beta,NB,Iout,criValue)
% ICM_estimation(VBN,pxl_k,Ni,Object,D,beta,NB)
% A,����������Ѫ�ܺͱ���������ݣ�
% flagIMG����ʴ����ͼD1_EM�Ŀռ䣨�������ࣩ�����ų��˾�ֵ��͵�������𣬲���1�������ռ䣬0�������ռ�
% D,�ο���׼���ݣ�Ѫ�ܺͱ����ֱ�Ϊ��3���ͣ�2,1),EM��ȡ��Ѫ�ܺ�ѡ�ռ䣻
% beta��������ϵ��/������
% Ni����Frangi�����õ�Ѫ�ܺ�ѡ�ռ������ݵĵ��������pxl_kΪ��Ӧ���������ʡ�
W = VBN(:,3);
Li = 1:Object-1;%��Ŀ��������
sizeD = size(D);
Dnew = zeros(sizeD);
[a,b,c] = ind2sub(sizeD,Ni);
s = [a b c];
FN = (NB == 0)*3 + (NB == 6)*1 +(NB == 26)*2;%�Զ��庯�����
f={@clique_6 @clique_26 @clique_MPN};%�Զ������ź���
Iout=Iout/max(Iout(:));
Hessian_map=1./(1+(criValue./(Iout+eps)).^2);
for n = 1:length(Ni)
    [pB,pV] = f{FN}(Object,s(n,:),D,beta,Hessian_map);% s(n,:)ΪFrangi��ȡ��Ѫ�ܵ㣬DΪEM��ȡ�ĵ㡣
    post_V = pV*pxl_k(n,3)/W (Object);
    post_B = pB*sum(pxl_k(n,Li'))/sum(W(Li'));
    Dnew(Ni(n)) = Object * (post_V >post_B);
end
%****************** SubFunction cliqueMPN *************************
function [pB,pV,NsigmaV,NsigmaB] = clique_MPN(K,s,D,beta)
% NsigmaV,Ŀ�������NsigmaB����������
% �ھ�ֵˮƽmu(k)�£��ֱ�����s������Ѫ��Ŀ��ĸ���pV�����ڱ����ĸ���pB
% KΪĿ����ı�ǣ�AΪ���ָ�ͼ��DΪ��ʼ��ǳ�
% flag =0������s�㼰�������ڱ����У�
% flag~=0������s�㼰��������Ŀ����
[i_max,j_max,n_max] = size(D);
i = s(1);j = s(2);n = s(3);
ip = (i+1<=i_max)*(i+1)+(i+1>i_max);im = (i-1>=1)*(i-1)+(i-1<1)*i_max;%----�мӺͼ�
jp = (j+1<=j_max)*(j+1)+(j+1>j_max);jm = (j-1>=1)*(j-1)+(j-1<1)*j_max;%----�мӺͼ�
np = (n+1<=n_max)*(n+1)+(n+1>n_max);nm = (n-1>=1)*(n-1)+(n-1<1)*n_max;%----��Ӻͼ�
% �ڼȶ���26��������D_nb26�У�����ռ�����ľ���ṹ
A = [5 11 13 15 17 23]; % D_nb26�е�6�����㣻
% ��(i,j,n)Ϊ���ģ���26������������ɺ���ǰ���������ҡ��������Ϸ�Ϊ8�������壬ÿ���������(i,j,n)����7������,������Ķ�����ֵö�����£�
C = [1 2 4 5 10 11 13;2 3 5 6 11 12 15;4 5 7 8 13 16 17;5 6 8 9 15 17 18;...
     10 11 13 19 20 22 23;11 12 15 20 21 23 24;13 16 17 22 23 25 26;15 17 18 23 24 26 27];
% ��(i,j,n)Ϊ���ģ���26������������ɺ���ǰ���������ҡ��������Ϸ�Ϊ6�������壬ÿ���������(i,j,n)����9������,������Ķ�����ֵö�����£�
F = [4 5 10 11 13 16 17 22 23;5 6 11 12 15 17 18 23 24;2 5 10 11 12 13 15 20 23; ...
     5 8 13 15 16 17 18 23 26;2 4 5 6 8 11 13 15 17;11 13 15 17 20 22 23 24 26]; 
 
D_nb26 = [D(im,jm,nm) D(i,jm,nm) D(ip,jm,nm) D(im,j,nm) D(i,j,nm) D(ip,j,nm) D(im,jp,nm) D(i,jp,nm) D(ip,jp,nm) ... %3x3x3������ -1����ֵ
          D(im,jm,n)  D(i,jm,n)  D(ip,jm,n)  D(im,j,n)  D(i,j,n)  D(ip,j,n)  D(im,jp,n)  D(i,jp,n)  D(ip,jp,n) ...  %3x3x3������  0����ֵ
          D(im,jm,np) D(i,jm,np) D(ip,jm,np) D(im,j,np) D(i,j,np) D(ip,j,np) D(im,jp,np) D(i,jp,np) D(ip,jp,np)];   %3x3x3������ +1����ֵ          
% flag = sum(D_nb26(A)==K); % ͳ�����ĵ���Χ���ΪK�ĵ���(�Կռ�6����Ϊ��׼)
% 26������ά�ռ��N�����ż���--------------------- 
NsigmaV_6 = sum(D_nb26(A)==K);NsigmaB_6 = sum(D_nb26(A)~=K);
NsigmaV_7 = max([sum(D_nb26(C(1,:))==K) sum(D_nb26(C(2,:))==K) sum(D_nb26(C(3,:))==K) sum(D_nb26(C(4,:))==K) ...
                     sum(D_nb26(C(5,:))==K) sum(D_nb26(C(6,:))==K) sum(D_nb26(C(7,:))==K) sum(D_nb26(C(8,:))==K)]);% C���������µ�����
NsigmaB_7 = max([sum(D_nb26(C(1,:))~=K) sum(D_nb26(C(2,:))~=K) sum(D_nb26(C(3,:))~=K) sum(D_nb26(C(4,:))~=K) ...
                     sum(D_nb26(C(5,:))~=K) sum(D_nb26(C(6,:))~=K) sum(D_nb26(C(7,:))~=K) sum(D_nb26(C(8,:))~=K)]);% C���������µ�����      
NsigmaV_9 = max([sum(D_nb26(F(1,:))==K) sum(D_nb26(F(2,:))==K) sum(D_nb26(F(3,:))==K) ...
                     sum(D_nb26(F(4,:))==K) sum(D_nb26(F(5,:))==K) sum(D_nb26(F(6,:))==K)]);   % F���������µ�����
NsigmaB_9 = max([sum(D_nb26(F(1,:))~=K) sum(D_nb26(F(2,:))~=K) sum(D_nb26(F(3,:))~=K) ...
                     sum(D_nb26(F(4,:))~=K) sum(D_nb26(F(5,:))~=K) sum(D_nb26(F(6,:))~=K)]);   % F���������µ�����       
% D_nb26�о���A���������Ŀ��ͱ�������
Uv6 = 6-NsigmaV_6; Ub6 = 6-NsigmaB_6;
% D_nb26�о���C���������Ŀ��ͱ�������
Uv_bound1 = 7 - NsigmaV_7; Ub_bound1 = 7 - NsigmaB_7;
Uv_bound2 = 9 - NsigmaV_9; Ub_bound2 = 9 - NsigmaB_9;
% ����Ŀ����ʣ�����D(i,j,n)�Ƿ�ΪK����Χ���K�϶�ʱ��������С��Ŀ��������
Uvw = min([Uv6 Uv_bound1 Uv_bound2]);%Uvw = [Uv6 Uv_bound1 Uv_bound2];
Uv = beta*Uvw;
pV = exp(-Uv);
% ���㱳�����ʣ�����Χ��ǲ�ΪK�Ľ϶�ʱ��������С�������������
Ubw = min([Ub6 Ub_bound1 Ub_bound2]);%Ubw = [Ub6 Ub_bound1 Ub_bound2];
Ub = beta*Ubw;
pB = exp(-Ub);
% ����Ŀ�����NsigmaV�ͱ�������NsigmaB
NsigmaV = min([NsigmaV_6 NsigmaV_7 NsigmaV_9]);
NsigmaB = min([NsigmaB_6 NsigmaB_7 NsigmaB_9]);

%****************** SubFunction clique_26 *************************
function [pB,pV,NsigmaV,NsigmaB] = clique_26(K,s,D,beta)
% �ھ�ֵˮƽmu(k)�£��ֱ�����s������Ѫ��Ŀ��ĸ���pV�����ڱ����ĸ���pB
% KΪĿ����ı�ǣ�AΪ���ָ�ͼ��DΪ��ʼ��ǳ�
% flag =0������s�㼰�������ڱ����У�
% flag~=0������s�㼰��������Ŀ���У�

% --------��1��----------�����������꣬�ο�neighbouring2
[i_max,j_max,k_max] = size(D);
i = s(1);j = s(2);k = s(3);
%----�м�1�ͼ�1
ip = (i<i_max)*(i+1)+(i==i_max);
im = (i>1)*(i-1)+(i==1)*i_max;
%----�м�1�ͼ�1
jp = (j<j_max)*(j+1)+(j==j_max);
jm = (j>1)*(j-1)+(j==1)*j_max;
%----���1�ͼ�1
kp = (k<k_max)*(k+1)+(k==k_max);
km = (k>1)*(k-1)+(k==1)*k_max;
% ---------End��1��------------
D_nb26 = [D(im,jm,km) D(i,jm,km) D(ip,jm,km) D(im,j,km) D(i,j,km) D(ip,j,km) D(im,jp,km) D(i,jp,km) D(ip,jp,km) ... %3x3x3������ -1����ֵ
          D(im,jm,k)  D(i,jm,k)  D(ip,jm,k)  D(im,j,k)  D(i,j,k)  D(ip,j,k)  D(im,jp,k)  D(i,jp,k)  D(ip,jp,k) ...  %3x3x3������  0����ֵ
          D(im,jm,kp) D(i,jm,kp) D(ip,jm,kp) D(im,j,kp) D(i,j,kp) D(ip,j,kp) D(im,jp,kp) D(i,jp,kp) D(ip,jp,kp)];   %3x3x3������ +1����ֵ
% A = [5 11 13 15 17 23]; % D_nb26�е�6�����㣻
% flag = sum(D_nb26(A)==K); % ͳ�����ĵ���Χ���ΪK�ĵ���(�Կռ�6����Ϊ��׼)
NsigmaV = sum(D_nb26(1:27~=14)==K);
NsigmaB = sum(D_nb26(1:27~=14)~=K);
Uv26 = 26-NsigmaV;% ȥ�����ĵ�14��D_nb26�о���A��=D_nb26������26�����Ŀ������
Ub26 = 26-NsigmaB;% ȥ�����ĵ�14��D_nb26�о���A��=D_nb26������26����ı�������
% ����Ŀ����ʣ�����D(i,j,k)�Ƿ�ΪK����Χ���K�϶�ʱ��������С��Ŀ��������
Uv = beta*Uv26;
pV = exp(-Uv);
% ���㱳�����ʣ�����Χ��ǲ�ΪK�Ľ϶�ʱ��������С�������������
Ub = beta*Ub26;
pB = exp(-Ub);

%****************** SubFunction clique_6 *************************
function [pB,pV,NsigmaV,NsigmaB] = clique_6(K,s,D,beta,H)
% �ھ�ֵˮƽmu(k)�£��ֱ�����s������Ѫ��Ŀ��ĸ���pV�����ڱ����ĸ���pB
% KΪĿ����ı�ǣ�AΪ���ָ�ͼ��DΪ��ʼ��ǳ�
% fb��¼��s����Χ�����ǵıȽϣ���ͬ��fb=1����ͬ��fb=0
% flag =0������s�㼰�������ڱ����У�
% flag~=0������s�㼰��������Ŀ���У�
[i_max,j_max,n_max] = size(D);
i = s(1);j = s(2);n = s(3);
ip = (i+1<=i_max)*(i+1)+(i+1>i_max);im = (i-1>=1)*(i-1)+(i-1<1)*i_max;%----�мӺͼ�
jp = (j+1<=j_max)*(j+1)+(j+1>j_max);jm = (j-1>=1)*(j-1)+(j-1<1)*j_max;%----�мӺͼ�
np = (n+1<=n_max)*(n+1)+(n+1>n_max);nm = (n-1>=1)*(n-1)+(n-1<1)*n_max;%----��Ӻͼ�
% �ڼȶ���26��������D_nb26�У�����ռ�����ľ���ṹ
D_nb26 = [D(im,jm,nm) D(i,jm,nm) D(ip,jm,nm) D(im,j,nm) D(i,j,nm) D(ip,j,nm) D(im,jp,nm) D(i,jp,nm) D(ip,jp,nm) ... %3x3x3������ -1����ֵ
          D(im,jm,n)  D(i,jm,n)  D(ip,jm,n)  D(im,j,n)  D(i,j,n)  D(ip,j,n)  D(im,jp,n)  D(i,jp,n)  D(ip,jp,n) ...  %3x3x3������  0����ֵ
          D(im,jm,np) D(i,jm,np) D(ip,jm,np) D(im,j,np) D(i,j,np) D(ip,j,np) D(im,jp,np) D(i,jp,np) D(ip,jp,np)];   %3x3x3������ +1����ֵ
Hessian_nb26 = [H(im,jm,nm) H(i,jm,nm) H(ip,jm,nm) H(im,j,nm) H(i,j,nm) H(ip,j,nm) H(im,jp,nm) H(i,jp,nm) H(ip,jp,nm) ... %3x3x3������ -1����ֵ
          H(im,jm,n)  H(i,jm,n)  H(ip,jm,n)  H(im,j,n)  H(i,j,n)  H(ip,j,n)  H(im,jp,n)  H(i,jp,n)  H(ip,jp,n) ...  %3x3x3������  0����ֵ
          H(im,jm,np) H(i,jm,np) H(ip,jm,np) H(im,j,np) H(i,j,np) H(ip,j,np) H(im,jp,np) H(i,jp,np) H(ip,jp,np)];   %3x3x3������ +1����ֵ

A = [5 11 13 15 17 23]; % D_nb26�е�6�����㣻 5-23 / 11-17 / 13-15
% flag = sum(D_nb26(A)==K); % ͳ�����ĵ���Χ���ΪK�ĵ���(�Կռ�6����Ϊ��׼)
% Aind=find(D_nb26(A)==K);
NsigmaV = sum(D_nb26(A)==K);
NsigmaB = sum(D_nb26(A)~=K);
% 26������ά�ռ��N�����ż���--------------------- 
% Uv6 = 6-NsigmaV;% D_nb26�о���A����6�����Ŀ������
% Ub6 = 6-NsigmaB;% D_nb26�о���A����6����ı�������
Uv6_1 = 6 - NsigmaV;% D_nb26�о���A����6�����Ŀ������
Ub6_1 = 6 - NsigmaB;% D_nb26�о���A����6����ı�������
Uv6_2 = 6 - sum(Hessian_nb26(A));
Ub6_2 = 1.5*sum(Hessian_nb26(A));
Uv6 = 0.5*Uv6_1 + 0.5*Uv6_2;
Ub6 = 0.5*Ub6_1 + 0.5*Ub6_2;
% fb = (Uv6~=0);
% ����Ŀ����ʣ�����D(i,j,n)�Ƿ�ΪK����Χ���K�϶�ʱ��������С��Ŀ��������
Uv = beta*Uv6;%Uv = Uv6;
pV = exp(-Uv);
% ���㱳�����ʣ�����Χ��ǲ�ΪK�Ľ϶�ʱ��������С�������������
Ub = beta*Ub6;
pB = exp(-Ub);

function [vessel]=OveralShows(SelecNum,Dx,Dx_MLE,flagIMG,Object)
Dout = zeros([size(Dx) 2]);% DoutΪ0/4��ֵ����
[Dout(:,:,:,1)] = GetRegion(Dx,0,SelecNum);% ��������SelecNum(i)������Ѫ�ܷ�֧   
[Dout(:,:,:,2)] = GetRegion(Dx,0,[1 2 3]);% ����Ѫ�ܷ�֧  
figure;
subplot(1,3,1);imshow3D_patch(Dx_MLE,flagIMG,[1 0 0]);title('ML����');
subplot(1,3,2);imshow3D_patch(Object*Dout(:,:,:,1),flagIMG,[1 0 0]);title('Markov���������');
subplot(1,3,3);imshow3D_patch(Object*Dout(:,:,:,2),flagIMG,[1 0 0]);title('�����ͨ����');

vessel=Dout(:,:,:,1);
disp(['Length of cerebral vessel is ' num2str(length(find(Dout(:,:,:,2)==1))) ' voxels']);

function imshow3D_patch(D,D_original,colormode)
% D��D_original������ͬ�ߴ磻��[a,b,c] = size(D)
% ��ȡ3D��ֵ���ݿռ�D_original��0/1��ֵ���ߴ磬���ڴ˿ռ�����ʾD��Ŀ�ꣻ
index = find(D_original==1);
[a,b,c] = ind2sub(size(D_original),index);
mina = min(a);maxa = max(a);
minb = min(b);maxb = max(b);
minc = min(c);maxc = max(c);
F = zeros((maxa-mina+1)+9,(maxb-minb+1)+9,(maxc-minc+1)+9);
F(4:4+(maxa-mina),4:4+(maxb-minb),4:4+(maxc-minc)) = D(mina:maxa,minb:maxb,minc:maxc);
[a,b,c] = size(F);
% set(gca,'color','black'); 
patch(isosurface(F,0.5),'FaceColor',colormode,'EdgeColor','none');
axis([0 b 0 a 0 c]);view([270 270]);daspect([0.8,0.8,0.4297]);
camlight; camlight(-80,-10); lighting phong; pause(1); 