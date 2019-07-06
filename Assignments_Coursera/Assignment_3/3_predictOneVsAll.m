function p = predictOneVsAll(all_theta, X)

% important values

m = size(X, 1);
num_labels = size(all_theta, 1);

% initializing the variable
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% predicting the value
predict = sigmoid(X*all_theta');
[predict_max, index_max] = max(predict, [], 2);
p = index_max;

end
