function w = regularizelogRegGD(features, y, lambda)
%% w = lmsSolveGDFeatNormal(features, y)
%% INPUT: features - un-normalized features, one row per sample
%%        y - true labels, corresponding to features
%% OUTPUT: w - optimal weight vector (including w0) found WITH feature normalization/scaling using Gradient Descent

%% Write your code below this line

len=length(features(:,1)) + 1;
%display(len);
w=ones(len);
w=w(1:1,:);
%display(length(w));
%display(w);
[newW, jhist] = regularizegradientDescent(features,y,w,0.000000001,100, lambda);
w=newW;


end

