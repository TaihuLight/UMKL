% Experiment: solve the primal problem with perturbed basis vectors
n = 10;

K1 = eye(n) + normrnd(0, 0.1, n, n);
K2 = eye(n) + normrnd(0, 0.5, n, n);
K = [K1, K2];

rho = 0.05;
m = size(K, 2);

cvx_begin
    variable lambda(m+1, 1);
    
    minimize ( trace_inv(combined_kernel_reg1(lambda, K, rho)) )
    
    sum(lambda) == 1;
    lambda >= 0;
    
cvx_end

bar(lambda)