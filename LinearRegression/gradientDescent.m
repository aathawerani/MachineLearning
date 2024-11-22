function [w, J_history] = gradientDescent(X, y, w, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   w = GRADIENTDESCENT(x, y, w, alpha, num_iters) updates w by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               w. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
[j,gradient] = computeCost(X,y,w);
temp = w - alpha * gradient;
w = temp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = j;

end

end
