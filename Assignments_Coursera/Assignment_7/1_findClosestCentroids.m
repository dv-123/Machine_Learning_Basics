function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

idx = zeros(size(X,1), 1);

for i = 1:length(idx)
  
  d = centroids - X(i,:);
  dist = ((d*d').*eye(K))*ones(K,1);
  [m,id] = min(dist);
  idx(i) = id;
  
endfor

%% for loop implementation

%for i = 1:length(idx)
%  closest = 0;
%  best_dist = 999999999;
%  for j = 1:K
%    d = X(i,:) - centroids(j,:);
%    dist = d*d';
%    if dist < best_dist
%      best_dist = dist;
%      closest = j;
%    endif
%  endfor
%  idx(i) = closest;
%endfor

end
