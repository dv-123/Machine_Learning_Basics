function [J, grad] = costFunctionReg(theta, X, y, lambda)

% Initialize some useful values
m = length(y);

% initialize the variables
J = 0;
grad = zeros(size(theta));

% computing the cost and gradients
z = X * theta;

s = size(theta);

%J = 1/m*(sum(-y.*log(sigmoid(z)) - (1-y).*log(1-sigmoid(z))));
J = 1/m*(sum(-y.*log(sigmoid(z)) - (1-y).*log(1-sigmoid(z)))) + (lambda/(2*m))*sum(theta(2:s).^2) ;
 
grad(1) = 1/m * sum((sigmoid(z) - y).*X(:,1));
for i = 2:s(1)
  grad(i) = 1/m * sum((sigmoid(z) - y).*X(:,i)) + (lambda/m)*theta(i);
endfor

end
