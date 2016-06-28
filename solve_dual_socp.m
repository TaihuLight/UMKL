load 'kernels.mat'
kernels = {unigram_kernel, bigram_kernel, trigram_kernel};
n = size(unigram_kernel, 1);
p = 10;
rho = 0.5;

[K, D] = eigs(kernels{1}, p);
for i = 1:size(K, 2)
    K(:,i) = K(:,i) * D(i,i);
end
for i = 2:size(kernels, 2)
    [V, D] = eigs(kernels{i}, p);
    for i = 1:size(V,2)
        V(:,i) = V(:,i) * D(i,i);
    end
    K = [K, V];
end

cvx_begin
    variable B(n, n)
    
    maximize ( trace(B) )
    
    norm(B, 'fro') <= 1;
    for i = 1:size(K, 2)
        norm(K(:,i)' * B, 'fro') <= rho;
    end
    
cvx_end