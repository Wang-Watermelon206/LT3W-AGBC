function [GBS, A, WJ, R, S] = Algorithm1_GBC(X)
n = size(X, 1);
k = max(2, min(round(log2(n)), 20));          
idx = knnsearch(X, X, 'K', k + 1);            
idx = idx(:, 2:end);                         

A = zeros(n, n);
for i = 1:n
    A(i, idx(i, :)) = 1;
end
A = max(A, A');                               
A(1:n + 1:end) = 0;                           
cover = false(n, 1);
GBS = {};
while any(~cover)
    seed = find(~cover, 1);                  
    cand = unique([seed, find(A(seed, :))]); 
    ball = cand(~cover(cand));               
    if isempty(ball), ball = seed; end
    GBS{end + 1} = ball;
    cover(ball) = true;
end

m = numel(GBS);
WJ = zeros(m, 1);
R  = zeros(m, 1);
S  = zeros(m, 1);
for i = 1:m
    g  = GBS{i};
    c  = mean(X(g, :), 1);                    
    d  = sqrt(sum((X(g, :) - c).^2, 2));     
    R(i) = max(d);                            
    if numel(g) > 1
        WJ(i) = 1 / (1 + mean(pdist(X(g, :)))); 
    else
        WJ(i) = 1;
    end
end
if m > 1
    rn = (R  - min(R))  ./ (max(R)  - min(R)  + eps);
    wn = (WJ - min(WJ)) ./ (max(WJ) - min(WJ) + eps);
else
    rn = 0; wn = 0;
end
S = 0.5 * wn + 0.5 * (1 - rn);   
end
