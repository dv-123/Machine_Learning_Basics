function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

m = length(y);                                                          % number of training examples
J_history = zeros(num_iters, 1);                                        % initializeng J_history

for iter = 1:num_iters
    
    % initial computations
    hypo = X*theta;
    difference = hypo - y;
    
    % computing and saving the gradient
    delta1 = (1/m) * sum(difference.*X(:,1));
    delta2 = (1/m) * sum(difference.*X(:,2));
    delta3 = (1/m) * sum(difference.*X(:,3));
    new1 = alpha*delta1;
    new2 = alpha*delta2;
    new3 = alpha*delta3;
    theta(1) = theta(1) - new1;
    theta(2) = theta(2) - new2;
    theta(3) = theta(3) - new3;

    % Save the cost J in every iteration  
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
