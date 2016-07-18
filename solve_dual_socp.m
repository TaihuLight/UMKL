%% Load data from Amazon reviews
load 'kernels.mat'
kernels = {unigram_kernel, bigram_kernel, trigram_kernel};
n = size(unigram_kernel, 1);
p = 10;
rho = 0.05;

[K, D] = eigs(kernels{1}, p);
for i = 1:size(K, 2)
    K(:,i) = K(:,i) * sqrt(D(i,i));
end
for i = 2:size(kernels, 2)
    [V, D] = eigs(kernels{i}, p);
    for i = 1:size(V,2)
        V(:,i) = V(:,i) * sqrt(D(i,i));
    end
    K = [K, V];
end
m = size(K, 2);

%% Load experimental random data
load('random_kernel.mat')
rho = 0.05;
n = size(K, 1);
m = size(K, 2);

%% Solve dual SOCP
cvx_begin
    variable B(n, n)
    
    maximize ( trace(B) )
    
    norm(B, 'fro') <= 1;
    for i = 1:size(K, 2)
        norm(K(:,i)' * B, 'fro') <= rho;
    end
    
cvx_end

%% Recover primal weights

C = (1/cvx_optval) * inv(B);

cvx_begin
    variable lambda(m+1, 1);
    
    minimize ( norm(combined_kernel_reg1(lambda, K, rho) - C) );
    lambda >= 0;
    sum(lambda) == 1;
    
cvx_end

bar(lambda)