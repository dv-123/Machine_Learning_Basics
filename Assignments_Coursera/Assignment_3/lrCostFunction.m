function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y);                                                          % number of training examples

% initializing the variables

J = 0;
grad = zeros(size(theta));

% computing the regularized cost and gradient decent

s = size(theta);
s = s(1);

z = X*theta;
J = 1/m * sum(-y.*log(sigmoid(z)) - (1-y).*log(1-sigmoid(z))) + (lambda/(2*m))*sum(theta(2:s).^2) ;

b = sigmoid(z) - y;
grad(1) = 1/m * (X(:,1)'*b);
grad(2:s) = 1/m * (X(:,2:s)'*b)+ (lambda/m)*theta(2:s);


grad = grad(:);

end
