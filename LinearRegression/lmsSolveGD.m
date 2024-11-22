function w = lmsSolveGD(features, y)
%% w = lmsSolveGD(features, y)
%% INPUT: features - un-normalized features, one row per sample
%%        y - true labels, corresponding to features
%% OUTPUT: w - optimal weight vector (including w0) found without feature normalization/scaling using Gradient Descent method

%% Write your code below this line
w=[1,1,1,1,1,1,1,1,1]
[newW, jhist] = gradientDescent(features,y,w,0.000000001,100);
w=newW;
end
