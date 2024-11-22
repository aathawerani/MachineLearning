function sigX = sigmoid(vecX)
%% w = lmsSolveNormal(features, y)
%% INPUT: features - (un)normalized features, one row per sample
%%        y - true labels, corresponding to features
%% OUTPUT: w - optimal weight vector (including w0)

%% Write your code below this line
sigX = sigmf(vecX,[1 0]);

end
