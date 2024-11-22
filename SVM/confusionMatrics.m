function [] = confusionMatrics( model,X,Y,c)
XTest=double(X);
Y=double(Y);
pred=[];
n=size(XTest,2);
yp=double(zeros(n,1));
estimates=[];
confusionMatrix=[];
Avg_confusionmatrix=zeros(2);
Avg_Accuracy=[];
for i=1:62
    y=repmat(c(i),n,1);%creates a column vector by repeating the ith unique value
yp(find(y==Y))=1; % yp is the label to be used in one versus other 
yp(find(y~=Y))=-1;

    
    
    
    [C, accuracy,prob_estimates] = svmpredict(yp, XTest', model(i), ' -b 1');
   %C=svmclassify(model(i),XTest','showplot',false);
%errRate(i)=sum(YTest~=C)/length(YTest);
estimates=[estimates prob_estimates(:,1)];
confusionMatrix=confusionmat(C,yp);
Avg_confusionmatrix=Avg_confusionmatrix +confusionMatrix;
Avg_Accuracy=Avg_Accuracy+accuracy(1);
end
fprintf('The average Accuracy of the model is\t =%f\n',Avg_Accuracy/5);
fprintf('The average confusion matrix of the model is\n');
disp(Avg_confusionmatrix/5);

end
