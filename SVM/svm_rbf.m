function[model table_out]=svm_rbf(x,y)
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
table=[];
table_out=[];
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
options='-s 0 -t 2 -b 1 ';
% e=['-d 0001 -d 002 -d 0003 '];
% f=['-c 0.01 -c 0001 -c 0100 '];
e=['-g 0001 -g 002 -g 0003 '];
f=['-c .001 -c 0001 -c 0100 '];

for i=1:m            %This loop is used to create 62 classifiers 
    %******************************************************************************************************  
%     for g=1:3
%         gamma=[0.01,1,100];
        for h=1:3
            Cs=[0.01,1,100];
            for d=1:3
                deg=[1,2,3];
               sum=0;
               sum_supportVector=0;
    for v=1:5
XTrain=X;
YTrain=Y;
XTest=XTrain(:,(f_size*(v-1)+1):f_size*v);
YTest=YTrain((f_size*(v-1)+1):f_size*v);
XTrain(:,(f_size*(v-1)+1):f_size*v)=[];
YTrain((f_size*(v-1)+1):f_size*v)=[];
yp=zeros(size(YTrain));
yt=zeros(size(YTest));
 n=length(YTrain);
 k=length(YTest);
    %*************************************************************************************************************
y=repmat(c(i),n,1);%creates a column vector by repeating the ith unique value
y1=repmat(c(i),k,1);%creates a column vector by repeating the ith unique value
yp(find(y==YTrain))=1; % yp is the label to be used in one versus other 
yp(find(y~=YTrain))=-1;
yt(find(y1==YTest))=1; % yp is the label to be used in one versus other 
yt(find(y1~=YTest))=-1;

% if(numel(unique(yp))==1)
%    continue;
% end
YTest=YTest==c(i);
yt=double(yt);
yp=double(yp);
%options.MaxIter=25000;
%svmStruct = svmtrain(XTrain, yp, 'kernel_function','polynomial','polyorder',deg(d), 'boxconstraint',Cs(h),'Options',options, 'tolkkt' , 1.0000e-001, 'kktviolationlevel' ,1);
 %svmStruct = svmtrain(yp, XTrain', '-s 0 -t 1 -d 2 -g 3 -b 1 ');
 svmStruct = svmtrain(yp, XTrain', [options  e(d:d+7)  f(h:h+7)]);;;

%opt_w=[opt_w;theta];    %it contains the optimal w's for all predictors
%C=svmclassify(svmStruct,XTest','showplot',false);
[C, accuracy,prob_estimates] = svmpredict(yt, XTest', svmStruct, ' -b 1');

%errRate(i)=sum(YTest~=C)/length(YTest);
%conMat=confusionmat(yt,C);
if(numel(unique(yt))>1)
[FPR,TPR,T,AUC]=perfcurve(yt,C,1);
end
if(AUC>AUC_old(i))
    model(i)=svmStruct;
    AUC_old(i)=AUC;
end
sum_supportVector=sum_supportVector+svmStruct.totalSV;
sum=sum+AUC;
    end
      Avg_AUC(i)=sum*20;
        fprintf('\nClass\t   Kernel\t    C \t            Degree \t   Avg_AUC \t Avg_number_supportVector\n');
      fprintf('\n  %c\t  Rbf   \t %f \t   %f\t      %f\t    %f\n',c(i),Cs(h),deg(d),Avg_AUC(i),sum_supportVector/5);
            %end
         
        end
    end
  
end
