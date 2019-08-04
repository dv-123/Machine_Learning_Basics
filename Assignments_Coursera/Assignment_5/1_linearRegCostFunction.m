function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%X = [ones(m,1) X];

% ======================= FASTER WAY ==========================

h = X*theta;

% regularize theta by removing first value
theta_reg = [0;theta(2:end, :);];
J = (1/(2*m))*sum((h-y).^2)+(lambda/(2*m))*theta_reg'*theta_reg;

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);

% ===================== explained as below ===================
%
% z = X*theta;
%
% m_1 = length(theta);
%
% J = 1/(2*m) * sum((z - y).^2) + lambda/(2*m) * sum((theta(2:m_1,:)).^2);
%
% m_2 = length(grad);
% grad(1) = 1/m * sum((z - y).*X(:,1));
% for i=2:m_2
%   grad(i) = 1/m * sum((z - y).*X(:,i)) + (lambda/m) * (theta(i,:));
% endfor



% =========================================================================

grad = grad(:);

end
