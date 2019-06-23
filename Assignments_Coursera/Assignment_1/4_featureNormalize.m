function [X_norm, mu, sigma] = featureNormalize(X)

%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

X_norm = X;                                                    % initializing X_norm
mu = zeros(1, size(X, 2));                                     % initializing mu
sigma = zeros(1, size(X, 2));                                  % initializing sigma

s = size(X);                                                   % initializing s

% computing and storing mean to mu
mean2 = sum(X(:,1))/s(1);                                      
mean3 = sum(X(:,2))/s(1);
mu = [mean2 mean3];

% subtracting the means from initial values
s2 = X(:,1) - mean2;
s3 = X(:,2) - mean3; 

% computing the standard deviation and storing in sigma
sd2 = (sum(s2.^2)/s(1))^1/2;
sd3 = (sum(s3.^2)/s(1))^1/2;
sigma = [sd2 sd3];

% comuting and storing the normalized value
X_norm = [s2./sd2 s3./sd3];

end
