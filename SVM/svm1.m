function[]=svm_matlabLib(x,y)
% function [  ]= svm(path)
% XTrain =[];
% predTrain=[];
% for i=1:6283 %********************************************************************
%     img=imread([path '\trainResized' '\' num2str(i) '.bmp']);
%     if(ndims(img)==3)
%         img=rgb2gray(img);
%     end
%     img=img';
%     img=img(:);
%     XTrain=int8([XTrain img]);
% end
% YTrain=[];
% f=fopen([path '\trainLabels.csv']);%('D:\matlab\trainLabels.csv');
% label=textscan(f,'%d%s%s');
% fclose(f);
% lab=label{2};
% lab=char(lab);
% lab=lab(:,2);%*************************************************************************
% %Cross
% %validation*******************************************************************
% YTrain=[YTrain lab];
% return
XTrain=x;
YTrain=y;
y_length=length(YTrain);
f_size=floor(y_length/5);
X=double(XTrain);
Y=YTrain;
%Y_validation=[];
%X_validation=[];
Y_predicted=[];
correct=[];
AUC=[];
AUC_old=zeros(62,1);
w_avg=zeros(62,401);  
[c ia ic]=unique(YTrain,'rows');
m=length(c);
opt_w=[];
for i=1:m            %This loop is used to create 62 classifiers 
    %******************************************************************************************************  
    fprintf('\nClass\t  Kernel\t C\tgamma\tAvg_AUC\n');
    for g=1:3
        gamma=[0.01,1,100];
        for h=1:3
            Cs=[0.01,1.0,100.0];
               sum=0;
    for v=1:5
XTrain=X;
YTrain=Y;
XTest=XTrain(:,(f_size*(v-1)+1):f_size*v);
YTest=YTrain((f_size*(v-1)+1):f_size*v);
XTrain(:,(f_size*(v-1)+1):f_size*v)=[];
YTrain((f_size*(v-1)+1):f_size*v)=[];
yp=zeros(size(YTrain));
 n=length(YTrain);  
    %*************************************************************************************************************
y=repmat(c(i),n,1);%creates a column vector by repeating the ith unique value
yp(find(y==YTrain))=1; % yp is the label to be used in one versus other 
yp(find(y~=YTrain))=0;
YTest=YTest==c(i);
YTest=double(YTest);
svmStruct = svmtrain(XTrain, yp, 'kernel_function','rbf', 'boxconstraint',Cs(h),'rbf_sigma',gamma(g));
%opt_w=[opt_w;theta];    %it contains the optimal w's for all predictors
C=svmclassify(svmStruct,XTest','showplot',false);
%errRate(i)=sum(YTest~=C)/length(YTest);
conMat=confusionmat(YTest,C);
[FPR,TPR,T,AUC]=perfcurve(YTest,C,1);
if(AUC>AUC_old(i))
    model(i)=svmStruct
    AUC_old(i)=AUC;
end
sum=sum+AUC;
    end
      Avg_AUC(i)=sum*20;
      fprintf('\n%c\t  RBF\t %f\t%f\t%f\n',c(i),Cs(h),gamma(g),Avg_AUC(i));
        end
    end
  
end
