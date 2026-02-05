% ========================================================================
% main_BinaryClassification.m
%
% AM 230 – Numerical Optimization
%
% This script implements Gradient Descent (GD) for a binary classification
% problem using logistic regression.
%
% The goal is to solve:
%
%   min_theta  f(theta)
%   where
%     f(theta) = (1/N) * sum_i [ log(1 + exp(s_i)) - y_i s_i ]
%                + (mu/2) * ||theta||^2,
%     s_i = w^T x_i + b,   theta = [w; b].
%
% ========================================================================
% This script:
%   1) Loads training data (X, y)
%   2) Defines the logistic regression objective via a "problem" struct
%   3) Runs gradient descent with a fixed step size
%   4) Plots convergence diagnostics (loss, grad norm, parameter norm)
%
% Expected folder structure (relative to this file):
%   ./data     : Project1_train.mat
%   ./models   : logistic_objective.m
%   ./solvers  : solve_gd.m
%   ./utils    : helper functions (e.g., plotting, line search, etc.)
%
% NOTE:
%   - If you want UNregularized logistic regression, set problem.mu = 0.
%   - If you want regularized logistic regression, choose mu > 0.


clear; close all; clc;

%% ------------------------------------------------------------------------
%  Path setup
% -------------------------------------------------------------------------
% Reset MATLAB path to factory defaults, then add ONLY this project
restoredefaultpath;
rehash toolboxcache;   % refresh function cache
% Add folders for data, models, and utility functions.
root = fileparts(mfilename('fullpath'));
addpath(root);
addpath(fullfile(root,'data'));
addpath(fullfile(root,'models'));
addpath(fullfile(root,'solvers'));
addpath(fullfile(root,'utils'));

%% ------------------------------------------------------------------------
%  Load training data
% -------------------------------------------------------------------------
Tr = load(fullfile(root,'data','Project1_train.mat'));
% X: N-by-d feature matrix, y: N-by-1 labels in {0,1}
X = Tr.X;     %  N x d data matrix (each row is a sample)
y = Tr.y(:);   % N x 1 labels in {0,1}

[N,d] = size(X);

%% ------------------------------------------------------------------------
%  Define the optimization problem
% -------------------------------------------------------------------------
% The "problem" structure collects everything defining the objective.
% This separation allows us to reuse the same solver code for different
% problems later in the course.

problem = struct();
problem.Xtr = X;
problem.ytr = y;
problem.mu  = 10^-4;                    % regularization parameter
problem.obj = @logistic_objective;  % objective handle
% problem.obj(theta, X, y, mu, mode) returns:
%   mode=0: f
%   mode=1: g
%   mode=2: [f,g]
%   mode=3: [g,H]

%% ------------------------------------------------------------------------
%  Gradient Descent (GD) settings
% -------------------------------------------------------------------------
% The "opts" structure contains algorithmic choices and stopping criteria.
% These are solver-related parameters, not part of the problem definition.

opts = struct();
opts.maxIter    = 500;     % maximum GD iterations
opts.tolGrad    = 1e-8;     % stopping tolerance on ||grad||
opts.printEvery = 500;      % print diagnostics every this many iterations

% Line search parameters (used only if useLineSearch = true)
opts.ls = struct( ...
    'c1',        1e-4, ...
    'c2',        0.9,  ...
    'alpha0',    1.0,  ...
    'alpha_max', 100,   ...
    'alpha_min', 1e-6,   ...
    'maxIter',   50,   ...
    'maxZoom',   50 );

%% ------------------------------------------------------------------------
%  Initialization
% -------------------------------------------------------------------------
theta0 = zeros(d+1,1); % theta = [w; b] is (d+1)-by-1 where w is d-by-1 and b is scalar

% Step size options
opts.useLineSearch = false;  % false: use fixed step size
opts.alpha_fixed   = 1;  % used only if useLineSearch = false

[theta, hist] = solve_gd(problem, theta0, opts);
% theta is the final parameter value.
% hist is a struct storing iteration history for diagnostics, e.g.
%   hist.iter        : iteration indices (0..k)
%   hist.f           : loss values f(theta_k)
%   hist.gn          : gradient norms ||grad f(theta_k)||_2
%   hist.theta_norm  : parameter norms ||theta_k||_2


figure;

subplot(3,1,1);
semilogy(hist.iter, hist.f, 'LineWidth', 2);
grid on; ylabel('L(\theta_k)'); title('Gradient Descent Diagnostics');

subplot(3,1,2);
semilogy(hist.iter, hist.gn, 'LineWidth', 2);
grid on; ylabel('||\nabla L(\theta_k)||_2');

subplot(3,1,3);
semilogy(hist.iter, hist.theta_norm, 'LineWidth', 2);
grid on; ylabel('||\theta_k||_2'); xlabel('Iteration k');

set(gca,'FontSize',13);


results = struct();
results.GD.theta = theta;
results.GD.hist  = hist;
save(fullfile(root,'results_GD.mat'),'results','problem','opts');


%% ------------------------------------------------------------------------
%  Verification: Calculate Training Accuracy
% -------------------------------------------------------------------------
% 1. Extract learned weights (w) and bias (b) from theta
w = theta(1:d);
b = theta(end);

fprintf('Final Model Parameters:\n');
fprintf('  Weight w1: %.4f\n', w(1));
fprintf('  Weight w2: %.4f\n', w(2));
fprintf('  Bias b:    %.4f\n', b);

% 2. Compute scores for all training samples: s = X*w + b
scores = X*w + b;

% 3. Convert scores to binary predictions (1 if s > 0, else 0)
y_pred = (scores > 0);

% 4. Compare predictions with actual labels to find accuracy
num_correct = sum(y_pred == y);
accuracy = (num_correct / N) * 100;

fprintf('\nVerification Results:\n');
fprintf('Training Accuracy: %.2f%%\n', accuracy);
if accuracy == 100
    fprintf('The data is perfectly separated!\n');
else
    fprintf('Number of misclassified samples: %d\n', N - num_correct);
end


