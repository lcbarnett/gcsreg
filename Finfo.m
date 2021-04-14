function I = Finfo(A,V)

% Input
%
%     A        VAR coefficients array
%     V        residuals covariance matrix (positive-definite)
%
% Output
%
%     I        Fisher information matrix (VAR coefficients block)
%
% Description
%
% Calculate VAR coefficients block of Fisher information matrix for VAR likelihood function

[n,~,p] = size(A);
pn  = p*n;
p1n = (p-1)*n;

AA = [reshape(A,n,pn); eye(p1n) zeros(p1n,n)];
VV = [V zeros(n,p1n); zeros(p1n,pn)];
I = dlyap(AA,VV);
