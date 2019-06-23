function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  
  % initial calculations
  predict = X*theta;
  difference = predict - y;
  
  % computing the gradient
  delta1 = (1/m) * sum(difference.*X(:,1));
  delta2 = (1/m) * sum(difference.*X(:,2));
  new1 = alpha*delta1;
  new2 = alpha*delta2;
  theta(1) = theta(1) - new1;
  theta(2) = theta(2) - new2;
  
  % Save the cost J in every iteration
  J_history(iter) = computeCost(X, y, theta);
  
 endfor
  
 end 
