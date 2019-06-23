function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));                                          % initializing the function

% computing the sigmoid function

g = (1+exp(-z)).^-1;

end
