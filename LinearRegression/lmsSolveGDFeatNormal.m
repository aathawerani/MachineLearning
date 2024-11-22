function w = lmsSolveGDFeatNormal(features, y)
%% w = lmsSolveGDFeatNormal(features, y)
%% INPUT: features - un-normalized features, one row per sample
%%        y - true labels, corresponding to features
%% OUTPUT: w - optimal weight vector (including w0) found WITH feature normalization/scaling using Gradient Descent

%% Write your code below this line

[xnorm] = featureNormalize(features);
w=lmsSolveGD(xnorm, y);

end

