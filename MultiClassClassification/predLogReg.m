function predy = predLogReg(features, w)
%% w = lmsSolveGD(features, y)
%% INPUT: features - un-normalized features, one row per sample
%%        y - true labels, corresponding to features
%% OUTPUT: w - optimal weight vector (including w0) found without feature normalization/scaling using Gradient Descent method

%% Write your code below this line

len=length(features);
B=ones(len);
newC = [B(:,1:1) X(:,1:length(X(1,:)))];
T = newC*w';
len = length(T);

predy = zeros(len,1);

for iter=1:len
    if(T(1:iter)>=0)
        predy(1:iter) = 1
    end
end


end
