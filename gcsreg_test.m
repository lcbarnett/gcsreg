
if ~exist('nx',      'var'), nx       = 7;         end % target dimension
if ~exist('ny',      'var'), ny       = 10;        end % source dimension
if ~exist('p',       'var'), p        = 3;         end % VAR model order
if ~exist('rho',     'var'), rho      = 0.9;       end % VAR spectral norm
if ~exist('k',       'var'), k        = 0;         end % residuals correlation parameter (integer: bigger means *less* correlation)
if ~exist('blimits', 'var'), blimits  = [0.1,0.6]; end % frequency band limits (normalised to [0,1])
if ~exist('fres',    'var'), fres     = 1000;      end % frequency resolution (number of points in frequency band)

n = nx+ny;
x  = 1:nx;
y  = (nx+1):n;

frange = 2*pi*blimits;

% Generate random null-hypothesis VAR coefficients with specified spectral norm

A = randn(n,n,p);
A(x,y,:) = 0;
A = specnorm(A,rho);

% Generate a random positive-definite covariance matrix

X = randn(n,n+k);
V = X*X';

Ltd = genchi2_parms_td(A,V,nx);
Lbl = genchi2_parms_bl(A,V,nx,frange,fres);

d = p*ny;

figure(1);
clf;
plot((1:d)',[Ltd Lbl],'o');
xlim([0,d+1]);
title(sprintf('Eigenvalues for generalised \\chi^2 distribution (time domain)\n'));
legend({'broadband','band-limited'},'location','northwest');
xlabel('Number (sorted ascending)');
ylabel('Eigenvalue');
