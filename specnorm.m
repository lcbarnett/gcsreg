function [out1,out2] = specnorm(A,newrho)

% Input
%
%     A          VAR coefficients array
%     newrho     new VAR spectral norm
%
% Output
%
%     A          VAR coefficients array
%     rho        VAR spectral norm
%
% Description
%
%     rho = specnorm(A)
%
% returns the spectral norm (spectral radius) of the n x n x p array of
% VAR coefficients, where A(:,:,k) is the k-lag coefficients matrix.
%
%     [A,rho] = specnorm(A,newrho)
%
% decays the A(:,:,k) exponentially, so that the spectral radius is newrho.
% The original spectral radius is returned in rho.

[n,n1,p] = size(A);
assert(n1 == n,'VAR coefficients array has bad shape');
pn = p*n;
pn1 = (p-1)*n;
A1 = [reshape(A,n,pn); eye(pn1) zeros(pn1,n)]; % VAR "companion matrix"

% calculate spectral norm

rho = max(abs(eig(A1)));

if nargin < 2 || isempty(newrho)
    assert(nargout <= 1,'too many output parameters');
    out1 = rho; % spectral norm
else
    dfac = newrho/rho;
	f = 1;
	for k = 1:p
		f = dfac*f;
		A(:,:,k) = f*A(:,:,k);
	end
	out1 = A;   % spectral norm of A is now newrho
    out2 = rho; % original value of spectral norm
end
