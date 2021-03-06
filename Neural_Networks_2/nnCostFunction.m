function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%reshape y
nc = size(Theta2,1); %number of categories

%initialize categorzied y matrix
mapped_y = zeros(nc, length(y));
for i = 1:length(y)
   mapped_y(y(i), i) = 1; 
end

%add bias term to X
bias = ones(size(X, 1), 1);
Xb = [bias X]; 

%Calculate h(x)
hx = zeros(size(mapped_y))';

%forward propagations
alpha2 = [ones(size(Xb, 1), 1) sigmoid(Xb * Theta1')];
hx = sigmoid(alpha2 * Theta2');

%compute cost
J = -(1/m) *  sum(sum(mapped_y' .* log(hx) + (ones(size(mapped_y)) - mapped_y)' .* ...
    log(ones(size(mapped_y))' - hx),2));

% %compute cost regularized
J = J + (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + ...
    sum(sum((Theta2(:,2:end).^2))));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

reg1 = zeros(size(Theta1));
reg2 = zeros(size(Theta2));


for it = 1:m %iterate over training examples
    
    %forward propagation
    alpha_one = Xb(it,:);
    alpha_two = [1 sigmoid(alpha_one * Theta1')];
    alpha_three = sigmoid(alpha_two * Theta2');
    
    %calculate gradients
    delta_three = alpha_three - mapped_y(:,it)';
    Theta2_grad = Theta2_grad + delta_three' * alpha_two;
    
    z2 = alpha_one*Theta1';
    delta_two = (delta_three * Theta2(:, 2:end)) .* sigmoidGradient(z2);
    Theta1_grad = Theta1_grad + (alpha_one' * delta_two)';
    
end


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% 
reg1 = Theta1;
reg2 = Theta2; 

reg1(:,1) = 0;
reg2(:,1) = 0;

Theta1_grad = (Theta1_grad + lambda .* reg1) ./ m;
Theta2_grad = (Theta2_grad + lambda .* reg2) ./ m; 


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
