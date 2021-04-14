function [m,v,d,L,err,QA,QB] = genchi2_parameters(A,V,nx,fulldist)

if nargin < 4 || isempty(fulldist), fulldist = 1; end

[n,n1,p] = size(A);
assert(n1 == n);
[n1,n2] = size(V);
assert(n1 == n && n2 == n);
assert (nx > 0 && nx < n);

tol = sqrt(eps);

err.code = uint32(0);
err.msg  = '';

ny = n-nx;
x  = 1:nx;
y  = (nx+1):n;
yy = y'+(0:(p-1))*n; yy = yy(:);

% NOTE: You should(?) project A onto null space
%       before calling this: i.e., A(x,y,:) = 0

A0 = A(x,y,:);
if any(A0(:) ~= 0)
	%VAR parameters not in null space!\n');
end

% inverse Fisher information

I = IFINFO(A,V);
QB = I(yy,yy); % under H0

% Hessian assuming H0

QA = FINFO(A(y,y,:),parcov(V,y,x));

% Cholesky factors

[LA,pa] = chol(QA,'lower');
[RB,pb] = chol(QB);

% check positive-definite, calculate BA matrix (or equivalent)

if pa>0 || pb > 0
	if pa>0, err.code = bitset(err.code,1); err.msg = sprintf('%sA matrix not positive-definite\n',err.msg); end
	if pb>0, err.code = bitset(err.code,2); err.msg = sprintf('%sB matrix not positive-definite\n',err.msg); end
	M = QB*QA;
else
	% ensure M positive-definite, since we can
	ML = RB*LA;
	M = ML*ML';
end

% calculate (and if necessary adjust) eigenvalues

L = eig(M);
if any(abs(imag(L)) > tol)
	err.code = bitset(err.code,3);
	err.msg = sprintf('%scomplex eigenvalues detected: max imag = %.6fi (setting imaginary part to zero)\n',err.msg,maxabs(imag(L)));
end
L = real(L); % enforce real
if any(L < -tol)
	 err.code = bitset(err.code,4);
	 err.msg = sprintf('%snegative eigenvalues detected: min = %.6f (setting to 0)\n',err.msg,-max(-L));
end
L(L<eps) = 0;
if any(L > 1+tol)
	err.code = bitset(err.code,5);
	err.msg = sprintf('%seigenvalues outside unit disk detected: max = %.6f (setting to 1)\n',err.msg,max(L));
end
L(L>1-eps) = 1;

% NOTE: we use the eigenvalues (rather than trace) to calculate mean and
% variance, since in singular situations (cf. warnings above), the adjusted
% eigenvalues still return usable results, in particular for Gamma fit.

m = sum(L);      % gen chi^2 mean
v = 2*sum(L.*L); % gen chi^2 variance
d = p*ny;        % "degrees of freedom"

% m1 = trace(M);     % gen chi^2 mean
% v1 = 2*trace(M*M); % gen chi^2 variance
% [abs(m-m1) abs(v-v1)]

% adjust for Kronecker product with I_{xx}

if fulldist > 0
	m = nx*m;
	v = nx*v;
	d = nx*d;
	if nargout > 3 && fulldist > 1
		L = repmat(L,nx,1);
		if nargout > 4
			QA = kron(eye(nx),QA);
			QB = kron(eye(nx),QB);
		end
	end
end
