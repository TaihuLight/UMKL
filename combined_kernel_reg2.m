function Kernel = combined_kernel_reg2( mu, lambda, K, rho )
    n = size(K, 1);
    m = size(K, 2);
    Kernel = zeros(n, n);
    I = eye(n);
    
    for i = 1:n
        Kernel = Kernel + mu(i) * (I(:,i) * I(:,i)');
    end
    
    for i = 1:m
        Kernel = Kernel + rho^(-2) * lambda(i) * (K(:,i) * K(:,i)');
    end
end

