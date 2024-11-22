function [J, gradient] = computeCost(X, y, w)
%COMPUTECOST Compute cost for linear regression with multiple variables
%   J = COMPUTECOST(X, y, w) computes the cost of using w as the
%   parameter for linear/logistic regression to fit the data points in X and y. For Linear Regression, J(w) = 1/2 * RSS(w) and for Logistic Regression use J(w) = cross-entropy error(w)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
gradient = [];

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of w
%               You should set J to the cost and gradient to the computed gradient.
len=length(y);
B=ones(len);
newC = [B(:,1:1) X(:,1:8)];
T=newC*w'-y;  
J=sum(T.^2)/2; 
for iter = 1:9 
    gradient(:,iter)=sum(T.*newC(:,iter)); 
end; 

% =========================================================================

end
