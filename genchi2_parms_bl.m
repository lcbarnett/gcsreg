function [L,QA,QB] = genchi2_parms_bl(A,V,nx,frange,fres,warn_nonnull,tol)

% Input
%
%     A             VAR coefficients array
%     V             residuals covariance matrix (positive-definite)
%     nx            dimension of target variable
%     warn_nonnull  warn if A not in null space
%     frange        frequency range [fmin,fmax] (angular frequency in range [0,2pi])
%     fres          frequency resolution (number of frequency points in [fmin,fmax])
%     tol           numerical tolerance for imaginary and/or unstable eignevalues
%
% Output
%
%     L             eignevalues for generalised chi^2 distribution (sorted ascending)
%     QA            the generalised chi^2 'A' matrix
%     QB            the generalised chi^2 'B' matrix
%
% Description
%
% Calculates parameters for the asymptotic generalised chi^2 sampling distribution
% of the band-limited spectral single-regression Granger causality estimator. See:
%
%     A. J. Gutknecht and L. Barnett, Sampling distribution for single-regression
%     Granger causality estimators, arXiv, 2019: https://arxiv.org/abs/1911.09625

if nargin < 6 || isempty(warn_nonnull)
	warn_nonnull = true;
end

if nargin < 7 || isempty(tol)
	tol = sqrt(eps);
end

[n,n1,p] = size(A);
assert(n1 == n);
[n1,n2] = size(V);
assert(n1 == n && n2 == n);
assert (nx > 0 && nx < n);

x  = 1:nx;
y  = (nx+1):n;
yy = y'+(0:(p-1))*n;
yy = yy(:);

% Check if parameters are in null space; i.e., A(x,y,:) = 0.
% If not, project onto null space (with optional warning).

H0 = A(x,y,:) == 0;
if ~all(H0(:))
	if warn_nonnull
		warning('VAR parameters not in null space! Projecting onto null space');
	end
	A(x,y,:);
end

% Inverse Fisher information

I = inv(Finfo(A,V));
QB = I(yy,yy); % under H0

% Hessian assuming H0

QA = Finfo_bl(A(y,y,:),parcov(V,y,x),frange,fres);

% Cholesky factor (note: QA may only be positive-semidefinite!)

[RB,pb] = chol(QB);

% check positive-definite, calculate BA matrix (or equivalent)

if pb > 0
	warning('Generalised chi^2 ''B'' matrix not positive-definite');
	QBA = QB*QA;
else
	QBA = RB*QA*RB'; % ensures QBA positive-semidefinite
end

% Calculate (and if necessary adjust) eigenvalues

L = eig(QBA);
if any(abs(imag(L)) > tol)
	warning('Complex eigenvalues detected: max imag = %.6fi (setting imaginary part to zero)\n',max(abs(imag(L))));
end
L = real(L); % enforce real-valued eignevalues
if any(L < -tol)
	 warning('Negative eigenvalues detected: min = %.6f (setting to 0)\n',-max(-L));
end
L(L<eps) = 0;
L = sort(L);

% NOTE: Use the eigenvalues (rather than trace) to calculate mean and variance,
% since in near-singular situations (cf. warnings above), the adjusted
% eigenvalues still return usable results, in particular for Gamma fit.
%
% m = sum(L);      % gen chi^2 mean
% v = 2*sum(L.*L); % gen chi^2 variance
% d = p*(n-nx);    % "degrees of freedom"
