function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% X = [ones(m,1) X];
% Xval = [ones(size(Xval,1)) Xval];

for i=1:m
  X_train = X(1:i,:);
  y_train = y(1:i);
  
  theta = trainLinearReg(X_train, y_train, lambda);
  
  % z_train = X_train*theta;
  % z_val = Xval*theta;
  
  error_train(i) = linearRegCostFunction(X_train, y_train, theta, 0);
  error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
  % taking lambda = 0;
  % error_train(i) = 1/(2*length(y)) * sum((z_train-y_train).^2) + (lambda/(2*length(y))) * sum((theta(2:length(theta),:)).^2); 
  % error_val(i) = 1/(2*length(yval)) * sum((z_val-yval).^2) + (lambda/(2*length(yval))) * sum((theta(2:length(theta),:)).^2);
endfor

end
