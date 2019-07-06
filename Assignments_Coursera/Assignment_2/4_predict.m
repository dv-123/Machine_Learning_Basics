function p = predict(theta, X)

m = size(X, 1);                                                            % Number of training examples

% initializing the variable
p = zeros(m, 1);

% predicting

prob = sigmoid(X*theta);

for i = 1:m
  if prob(i) >= 0.5
    p(i) = 1;
  endif
  if prob(i) < 0.5
    p(i) = 0;
  endif
end

end
