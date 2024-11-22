function[]=svmMatlab(x,y)
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
w_avg=zeros(62,401);
for v=1:5
XTrain=X;
YTrain=Y;
XTest=XTrain(:,(f_size*(v-1)+1):f_size*v);
YTest=YTrain((f_size*(v-1)+1):f_size*v);
XTrain(:,(f_size*(v-1)+1):f_size*v)=[];
YTrain((f_size*(v-1)+1):f_size*v)=[];
  
[c ia ic]=unique(YTrain,'rows');
n=length(YTrain);
m=length(c);
yp=zeros(size(YTrain));
opt_w=[];
for i=1:m            %This loop is used to create 62 classifiers 
y=repmat(c(i),n,1);%creates a column vector by repeating the ith unique value
yp(find(y==YTrain))=1; % yp is the label to be used in one versus other 
yp(find(y~=YTrain))=0;
YTest=YTest==c(i);
YTest=double(YTest);
svmStruct(i) = svmtrain(XTrain', yp, 'kernel_function','rbf', 'boxconstraint',01,'rbf_sigma',100);
%opt_w=[opt_w;theta];    %it contains the optimal w's for all predictors
C=svmclassify(svmStruct(i),XTest','showplot',false);
errRate(i)=sum(YTest~=C)/length(YTest);
conMat=confusionmat(YTest,C)
perfcurve(YTest,C,1)
end
 
w_avg=w_avg+opt_w;
[correct]=vadlidation( opt_w, XTest,YTest, c);%******************************************************************************************************
AUC=[AUC;correct ];

end
w_avg=w_avg./5;
for i=1:62
    fprintf('The AUC of model (%d) =%f\n',i,sum(AUC(:,i))/5*100); 
end

