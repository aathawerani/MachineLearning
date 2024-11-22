function w = logRegGD(features, y)
%% w = lmsSolveGDFeatNormal(features, y)
%% INPUT: features - un-normalized features, one row per sample
%%        y - true labels, corresponding to features
%% OUTPUT: w - optimal weight vector (including w0) found WITH feature normalization/scaling using Gradient Descent

%% Write your code below this line

[xnorm] = featureNormalize(features);
w=[1,1,1,1,1,1,1,1,1]
[newW, jhist] = gradientDescent(xnorm(1:461,:),y(1:461,:),w,0.000000001,100);
w=newW;

trainingy = predLogReg(xnorm(1:461,:), w);
holdouty = predLogReg(xnorm(462:768,:), w);

trainingaccuracy = y(1:461,:)-trainingy;
holdoutaccuracy = y(462:768,:)-holdouty;

display(sum(trainingaccuracy));
display(sum(holdoutaccuracy));

end

