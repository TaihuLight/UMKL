function Kernel = combined_kernel( lambda, K, rho )
    n = size(K, 1);
    m = size(K, 2);
    Kernel = rho^2 * lambda(1) * eye(n, n);
    for i = 1:m
        Kernel = Kernel + lambda(i+1) * (K(:,i) * K(:,i)');
end

