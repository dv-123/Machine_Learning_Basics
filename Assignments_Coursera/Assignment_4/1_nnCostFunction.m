function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

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

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1) X];

% forward propagation
z1 = X*Theta1';
a2 = sigmoid(z1);
a2 = [ones(m,1) a2];

z2 = a2*Theta2';
a3 = sigmoid(z2);

h = a3';

% transforming y into matrix form

y_matrix = zeros(num_labels, m);

for i=1:num_labels
  y_matrix(i,:) = (y==i);
endfor

% computation of cost_function

m_1 = size(Theta1,2);
m_2 = size(Theta2,2);

J = -1/m * sum(sum(y_matrix.*log(h) + (1-y_matrix).*log(1-h)));

Reg = (lambda/(2*m)) * ( sum(sum( Theta1(:,2:m_1).^2 )) + sum(sum( Theta2(:,2:m_2).^2 )) );

J = J + Reg;


% Backpropagation
for k=1:m
  %first we do forward propagation on X that already contains the bais node 
  % the forward propagation is to be done element/row wise
  a1 = X(k,:);
  z1 = a1*Theta1';
  
  a2 = sigmoid(z1);
  a2 = [1 a2];
  
  z2 = a2*Theta2';
  a3 = sigmoid(z2);
  
  % alfter taking the activation layer
  % we can go backwards
  a3 = a3';
  d2 = a3 - y_matrix(:,k);
  
  z1 = [1 z1];
  z1 = z1';
  
  d1 = (Theta2' * d2).*sigmoidGradient(z1);
  d1 = d1(2:end);
  
  Theta2_grad = (Theta2_grad + d2*a2);
  Theta1_grad = (Theta1_grad + d1*a1);
  
endfor

% Implementing Regularization

% for lambda = 0
Theta1_grad(:,1) = Theta1_grad(:,1)./m;
Theta2_grad(:,1) = Theta2_grad(:,1)./m;

% for lambda > 0
Theta1_grad(:,2:size(Theta1_grad,2)) = Theta1_grad(:,2:size(Theta1_grad,2))./m + ((lambda/m) * Theta1(:,2:m_1));
Theta2_grad(:,2:size(Theta2_grad,2)) = Theta2_grad(:,2:size(Theta2_grad,2))./m + ((lambda/m) * Theta2(:,2:m_2));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
