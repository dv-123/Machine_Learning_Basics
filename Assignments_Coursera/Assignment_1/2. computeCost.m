function J = computeCost( X, y, theta)

m = length(y);                                      % the number of training examples

J = 0;                                              % initialing the cost

predictions = X*theta;                              % craeting the predicted data with initailised random theeta
sqrErrors = (predictions-y).^2;                     % computing the squared error
J = 1/(2*m) * sum(sqrErrors);                       % computing the cost for current theeta

end
