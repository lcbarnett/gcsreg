function I = Finfo_bl(A,V,frange,fres)

% Input
%
%     A        VAR coefficients array
%     V        residuals covariance matrix (positive-definite)
%     frange   frequency range [fmin,fmax] (angular frequency in range [0,2pi])
%     fres     frequency resolution (number of frequency points in [fmin,fmax])
%
% Output
%
%     I        "bandlimited Fisher information" matrix (VAR coefficients block)
%
% Description
%
% Calculate VAR coefficients block of "bandlimited Fisher information" matrix for VAR likelihood function

[n,~,p] = size(A);
pn = p*n;

f = frange(1) + (frange(2)-frange(1))*((0:fres)/fres); % frange must be angular frequency!

L = chol(V,'lower');  % Cholesky factor of residuals covariance matrix

I = zeros(pn,pn,fres+1);
for i = 1:(fres+1)
	fi = f(i);
	Pi = eye(n);                    % inverse transfer function evaluated at f(i)
	for k = 1:p
		Pi = Pi - A(:,:,k)*exp(-1i*k*fi);
	end
	SLi = Pi\L;                     % Cholesky factor of CPSD evaluated at f(i)
	Zi  = exp(-1i*(0:(p-1))'*fi);   % frequency vector evaluated at f(i)
	I(:,:,i) = kron(Zi*Zi',SLi*SLi');
end

% Band-limit

I = trapz(real(I),3)/fres;
