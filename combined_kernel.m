function Kernel = combined_kernel( lambda, K, rho )
    n = size(K, 1);
    m = size(K, 2);
    Kernel = zeros(n, n);
    for i = 1:m
        Kernel = Kernel + rho^(-2) * lambda(i) * (K(:,i) * K(:,i)');
end

