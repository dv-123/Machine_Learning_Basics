function [J, grad] = costFunction(theta, X, y)

m = length(y);                                                      % number of training examples

% initialising the variables
J = 0;
grad = zeros(size(theta));

computing cost and the gradient decent

z = X * theta;

J = 1/m*(sum(-y.*log(sigmoid(z)) - (1-y).*log(1-sigmoid(z))));

for i = 1:length(theta):
  grad(i) = 1/m*sum((sigmoid(z) - y).*X(:,i));
end

% grad(1) = 1/m*sum((sigmoid(z) - y).*X(:,1));
% grad(2) = 1/m*sum((sigmoid(z) - y).*X(:,2));
% grad(3) = 1/m*sum((sigmoid(z) - y).*X(:,3));

end
