function [Y] = Algorithm3_GBClustering(GBS_star, C_B, A)


n = max([GBS_star{:}]);
Y = ones(n, 1);
m = numel(GBS_star);
if m <= 1, return; end

if size(C_B, 2) > 1
    D2  = pdist2(C_B, C_B);
    sig = mean(D2(D2 > 0));
    CS  = exp(-D2.^2 / (2 * sig^2 + eps));    % Gaussian kernel
else
    CS = zeros(m, m);
    for i = 1:m
        for j = i + 1:m
            CS(i, j) = 1 / (1 + abs(C_B(i) - C_B(j)));
            CS(j, i) = CS(i, j);
        end
    end
end
CS(1:m + 1:end) = 0;

G = graph(CS, 'upper');
T = minspantree(G);

w   = T.Edges.Weight;
thr = mean(w) + std(w);
bad = find(w > thr);
for i = 1:numel(bad)
    e = T.Edges.EndNodes(bad(i), :);
    CS(e(1), e(2)) = 0;
    CS(e(2), e(1)) = 0;
end

cc = conncomp(graph(CS));
for i = 1:m
    Y(GBS_star{i}) = cc(i);
end
end
