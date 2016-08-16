%% Load data from Amazon reviews
load 'data/kernels.mat'
kernels = {unigram_kernel, bigram_kernel, trigram_kernel};
n = size(unigram_kernel, 1);
p = 10;
rho = 0.05;c

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
y = eye(n);

%% Load random kernel vectors
load('data/random_kernel.mat')
rho = 0.01;
n = size(K, 1);
m = size(K, 2);
y = eye(n);

%% Augment matrices if desired
sigma = rho/20;
K = [K; sigma*eye(m)];
y = [eye(n); zeros(m,n)];

%% Sovle bidual
cvx_begin
    variable U(m,n);
    
    t1 = 0;
    for i = 1:m
        t1 = t1 + norm(U(i,:));
    end
    
    t2 = y;
    for i = 1:m
        t2 = t2 - K(:,i) * U(i,:);
    end
    
    minimize ( rho * t1 + norm(t2) )
    
cvx_end

%% Recover primal weights
lambda = zeros(m+1,1);
lambda(1) = (1/cvx_optval) * norm(t2);
for i = 1:m
    lambda(i+1) = (rho/cvx_optval) * norm(U(i,:));
end

bar(lambda);


    