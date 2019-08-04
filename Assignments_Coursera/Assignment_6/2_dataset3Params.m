function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = 1;
sigma = 0.3;


list = [ 0.01,0.03,0.1,0.3,1,3,10,30 ];
m = length(list);

error = zeros(m,m);

for i = 1:m
  for j = 1:m
    C = list(i);
    sigma = list(j);
    
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    pred = svmPredict(model, Xval);
    
    error(i, j) = mean(double(pred ~= yval));
    
  endfor
endfor

minval = min(min(error));
[i, j] = find(error == minval);

C = list(i);
sigma = list(j);

end
