function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
n = length(theta);

%hypothesis
hx = sigmoid(X * theta);

%cost function
J = -(1/m) * sum(y .* log(hx) + (ones(size(m,1)) - y) .* ...
    log(ones(m,1) - hx)) + 0.5 * (lambda/m) * sum(theta(2:n).^2) ;

%Gradient for theta(0)
grad(1) = (1/m) * sum((hx - y)' * X(:,1));

%Gradient for theta(1 ... n)
for i = 2:length(theta)
    grad(i) = (1/m) * sum((hx - y)' * X(:,i)) + lambda/m * sum(theta(i));
end

% =============================================================

end
