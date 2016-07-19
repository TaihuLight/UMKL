% Experiment: solve the primal problem with perturbed basis vectors
n = 10;

load('random_kernel.mat')
rho = 0.01;
m = size(K, 2);

cvx_begin
    variable lambda(m+1, 1);
    
    minimize ( trace_inv(combined_kernel_reg1(lambda, K, rho)) )
    
    sum(lambda) == 1;
    lambda >= 0;
    
cvx_end

bar(lambda)