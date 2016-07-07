function Kernel = combined_kernel( lambda, K )
    n = size(K, 1);
    m = size(K, 2);
    Kernel = zeros(n, n);
    for i = 1:m
        Kernel = Kernel + lambda(i) * (K(:,i) * K(:,i)');
end

