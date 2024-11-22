function w = lmsSolveNormal(features, y)
%% w = lmsSolveNormal(features, y)
%% INPUT: features - (un)normalized features, one row per sample
%%        y - true labels, corresponding to features
%% OUTPUT: w - optimal weight vector (including w0)

%% Write your code below this line
len=length(y);
B=ones(len);
newC = [B(:,1:1) features(:,1:8)];
w=pinv(newC'*newC)*newC'*y

end
