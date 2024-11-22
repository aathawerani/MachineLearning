function error = computeError(features, y, w)
%% error = computeError(features, y, x)
%% INPUT: features - (un)normalized features, one row per sample
%%        y - true labels, corresponding to features
%%        w - weight vector (including w0)
%% OUTPUT: MEAN squared error for Linear Regression, Classification error for Logistic Regression

%% Write your code below this line

newW = lmsSolveGD(features,y);
error = sum((newW - w).^2)/9
end
