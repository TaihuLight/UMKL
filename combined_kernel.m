function Kernel = combined_kernel( lambda, K, rho )
    n = size(K, 1);
    m = size(lambda, 1) - 1;
    Kernel = lambda(1) * eye(n, n);
    for i = 1:m
        Kernel = Kernel + (rho^(-2)) * lambda(i+1) * (K(:,i) * K(:,i)');
end

