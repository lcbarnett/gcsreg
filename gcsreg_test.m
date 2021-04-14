%-------------------------------------------------------------------------------
%
% Test routine for the eigenvalues of the generalised chi^2 asymptotic sampling
% distribution of broadband and band-limited single-regression Granger causality
% estimators. See:
%
%     A. J. Gutknecht and L. Barnett, Sampling distribution for single-regression
%     Granger causality estimators, arXiv, 2019: https://arxiv.org/abs/1911.09625
%
%-------------------------------------------------------------------------------
% Default parameters
%-------------------------------------------------------------------------------
if ~exist('nx',      'var'), nx       = 7;         end % target dimension
if ~exist('ny',      'var'), ny       = 10;        end % source dimension
if ~exist('p',       'var'), p        = 3;         end % VAR model order
if ~exist('rho',     'var'), rho      = 0.9;       end % VAR spectral norm
if ~exist('k',       'var'), k        = 0;         end % residuals correlation parameter (integer: bigger means *less* correlation)
if ~exist('blimits', 'var'), blimits  = [0.1,0.6]; end % frequency band limits (normalised to [0,1])
if ~exist('fres',    'var'), fres     = 1000;      end % frequency resolution (number of points in frequency band)
%-------------------------------------------------------------------------------

n = nx+ny;
x = 1:nx;
y = (nx+1):n;

frange = 2*pi*blimits; % frequency range in radians

d = p*ny;              % degrees of freedom

% Generate random null-hypothesis VAR coefficients with specified spectral norm

A = randn(n,n,p);    % random VAR coefficients array
A(x,y,:) = 0;        % enforce null condition
A = specnorm(A,rho); % set spectral radius to rho

% Generate a random positive-definite covariance matrix

X = randn(n,n+k);
V = X*X';

% Calculate eigenvalues for time-domain (broadband) generalised chi^2 distribution
% for the single-regression estimator for the Granger causality from y --> x

L = genchi2_parms(A,V,nx);

% Calculate eigenvalues for band-limited generalised chi^2 distribution

Lbl = genchi2_parms_bl(A,V,nx,frange,fres);

% Plot eigenvalues

figure(1);
clf;
plot((1:d)',[L Lbl],'o');
xlim([0,d+1]);
title(sprintf('Eigenvalues for generalised \\chi^2 distributions\n'));
yline(1);
legend({'broadband','band-limited'},'location','southeast');
xlabel('Eigenvalue number (sorted ascending)');
ylabel('Eigenvalue');
