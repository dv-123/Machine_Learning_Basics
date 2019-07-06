function p = predict(Theta1, Theta2, X)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% initializing the variables
p = zeros(size(X, 1), 1);

% adding ones to the feature vector matrix
X = [ones(m, 1) X];

% predicting using neural networks
predict1 = sigmoid(X*Theta1');
predict1 = [ones(m,1) predict1];
predict2 = sigmoid(predict1*Theta2');
[predict_max, index_max] = max(predict2, [], 2);
p = index_max;

end
