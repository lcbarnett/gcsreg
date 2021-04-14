function P = parcov(V,x,y)

% Input
%
%     V        covariance matrix (positive-definite)
%     x        vector of indices of target multi-variable
%     y        vector of indices of conditioning multi-variable
%
% Output
%
%     P        partial covariance matrix
%
% Description
%
% Given a (symmetric, positive-definite) covariance matrix, calculate
% the (symmetric, positive-definite) partial covariance matrix.
%
%     V(x,x|y) = V(x,x) - V(x,y)*inv(V(y,y))*V(y,x)

[~,p] = chol(V);
if p == 0
	U = linsolve(chol(V(y,y),'lower'),V(y,x),struct('LT',true)); % 'linsolve' is more efficient than 'rdivide', given that we know the Cholesky factor is lower-triangular
	P = V(x,x)-U'*U;
else
	warning('Covariance matrix not positive-definite');
	P = V(x,x)-V(x,y)*(V(y,y)\V(y,x));
end
