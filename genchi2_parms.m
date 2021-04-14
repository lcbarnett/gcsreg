function [L,QA,QB] = genchi2_parms(A,V,nx,warn_nonnull,tol)

% Input
%
%     A             VAR coefficients array
%     V             residuals covariance matrix (positive-definite)
%     nx            dimension of target variable
%     warn_nonnull  warn if A not in null space
%     tol           numerical tolerance for imaginary and/or unstable eignevalues

% Output
%
%     L             eignevalues for generalised chi^2 distribution (sorted ascending)
%     QA            the generalised chi^2 'A' matrix
%     QB            the generalised chi^2 'B' matrix

if nargin < 4 || isempty(warn_nonnull)
	warn_nonnull = true;
end

if nargin < 5 || isempty(tol)
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

QA = Finfo(A(y,y,:),parcov(V,y,x));

% Cholesky factors

[LA,pa] = chol(QA,'lower');
[RB,pb] = chol(QB);

% Check positive-definite, calculate BA matrix

if pa>0 || pb > 0
	if pa>0
		warning('Generalised chi^2 ''A'' matrix not positive-definite');
	end
	if pb>0
		warning('Generalised chi^2 ''B'' matrix not positive-definite');
	end
	QBA = QB*QA;
else
	QBAFAC = RB*LA;
	QBA = QBAFAC*QBAFAC'; % ensures QBA positive-definite
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
if any(L > 1+tol)
	warning('Eigenvalues outside unit disk detected: max = %.6f (setting to 1)\n',max(L));
end
L(L>1-eps) = 1;
L = sort(L);

% NOTE: Use the eigenvalues (rather than trace) to calculate mean and variance,
% since in near-singular situations (cf. warnings above), the adjusted
% eigenvalues still return usable results, in particular for Gamma fit.
%
% m = sum(L);      % gen chi^2 mean
% v = 2*sum(L.*L); % gen chi^2 variance
% d = p*(n-nx);    % "degrees of freedom"
