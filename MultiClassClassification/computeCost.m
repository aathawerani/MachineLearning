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

X=X';
len=length(X(:,1));
%display(len);
B=ones(len);
%display(length(X(:,1)));
%display(length(B(:,1:1)))
%display(length(X(:,1:length(X(1,:)))))
newC = [B(:,1:1) X(:,1:length(X(1,:)))];
%display(newC);
%display(w');
%display(length(w'));
%display(length(newC(1,:)));
%display(length(newC(:,1)));
T=sigmoid(double(newC)*double(w'));
%display('T');
%display(length(T));
%display(T);
%display('y');
%display(y);
%display(length(y));
S=sum(y'.*log(T) + (1-y').*log(1-T));
J=-S; 
for iter = 1:length(w) 
    gradient(:,iter)=sum((double(T)-double(y')).*double(newC(:,iter))); 
end; 

% =========================================================================

end
