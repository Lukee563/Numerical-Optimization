function [f, g, H] = logistic_objective(theta, X, y, mu, mode)
%LOGISTIC_OBJECTIVE  Regularized logistic regression objective and derivatives.
%
% Model:
%   s_i = w^T x_i + b,   theta = [w; b]
%   X is N x d (rows are samples)
%
% Loss:
%   f(theta) = (1/N) * sum_i [ log(1+exp(s_i)) - y_i*s_i ]
%              + mu*||theta||^2
%
% Gradient:
%   g = (1/N) * [ X'*(sigma-y) ; sum(sigma-y) ] + 2*mu*theta
%
% Hessian:
%   H = (1/N) * [ X' D X,    X' D 1
%                 1' D X,    1' D 1 ] + 2*mu*I
%   where D = diag(sigma_i(1-sigma_i)).
%
% -------------------------------------------------------------------------
% MODE USAGE (IMPORTANT)
% -------------------------------------------------------------------------
% This function supports the optimization methods used in AM 230:
%   - Gradient descent (fixed step):          mode = 1
%   - Gradient descent + Wolfe line search:   mode = 2 (during line search)
%   - Pure Newton (alpha = 1):                mode = 3
%   - Damped Newton + Wolfe line search:      mode = 3 (direction) and mode = 2 (line search)
%
% Modes:
%   mode = 0 : LOSS ONLY
%       Used by Wolfe sufficient decrease test (function value only).
%   mode = 1 : GRADIENT ONLY
%       Used by GD updates and Wolfe curvature test.
%   mode = 2 : LOSS + GRADIENT
%       Used by Wolfe line search for efficiency.
%   mode = 3 : HESSIAN + GRADIENT
%       Used by Newton/damped-Newton to form p = -H^{-1} g.
%
% Inputs:
%   theta : (d+1) x 1 parameter vector, theta = [w; b]
%   X     : N x d data matrix (rows are samples)
%   y     : N x 1 labels in {0,1}
%   mu    : >= 0 regularization parameter
%   mode  : computation mode (0,1,2,3)
%
% Outputs:
%   f : scalar loss (empty unless requested by mode)
%   g : (d+1) x 1 gradient (empty unless requested by mode)
%   H : (d+1) x (d+1) Hessian (empty unless requested by mode)

    narginchk(4,5);
    if nargin < 5 || isempty(mode)
        mode = 2;
    end
    if ~ismember(mode, [0 1 2 3])
        error('Invalid mode. Use 0, 1, 2, or 3.');
    end

    theta = theta(:);
    y = y(:);

    [N, d] = size(X);
    if numel(y) ~= N
        error('Dimension mismatch: y must have length N.');
    end
    if numel(theta) ~= d+1
        error('Dimension mismatch: theta must have length d+1.');
    end
    if mu < 0
        error('mu must be nonnegative.');
    end

    % Split parameters
    w = theta(1:d);
    b = theta(d+1);

    % ---------------------------------------------------------------------
    % Shared quantities used by ALL modes 
    % ---------------------------------------------------------------------
    
    s = X*w + b;          % N x 1 scores
    t = exp(-abs(s));     % N x 1, safe (<= 1)
    
    % Numerical stability identities:
    %   log(1+exp(s)) = max(s,0) + log(1+exp(-|s|)),
    %   sigmoid(s) = 1/(1+exp(-s)) = { 1/(1+exp(-s)), s>=0 ; exp(s)/(1+exp(s)), s<0 }.
    %
    % We compute t = exp(-|s|), which satisfies 0 < t <= 1, so it never overflows.

    % Initialize outputs
    f = [];
    g = [];
    H = [];

    % Mode needs
    need_loss = (mode == 0) || (mode == 2);
    need_grad = (mode == 1) || (mode == 2) || (mode == 3);
    need_hess = (mode == 3);

    % ---- LOSS ----
    if need_loss
        softplus = max(s,0) + log1p(t);          % stable log(1+exp(s))
        f = (1/N) * sum(softplus - y .* s);
        if mu > 0
            f = f + mu * (theta' * theta);
        end
        if mode == 0
            return;
        end
    end

    % ---- SIGMOID (needed for grad/Hess) ----
    if need_grad
        idx = (s >= 0);
        sigma = zeros(N,1);
        sigma(idx)  = 1 ./ (1 + t(idx));           % s>=0: t=exp(-s)
        sigma(~idx) = t(~idx) ./ (1 + t(~idx));    % s<0:  t=exp(s)
    end

    % ---- GRADIENT ----
    if need_grad
        err = sigma - y;                           % N x 1
        grad_w = (1/N) * (X' * err);               % d x 1
        grad_b = (1/N) * sum(err);                 % scalar
        g = [grad_w; grad_b];                      % (d+1) x 1
        if mu > 0
            g = g + (2*mu) * theta;
        end
        if mode == 1
            return;
        end
    end

    % ---- HESSIAN ----
    if need_hess
        dvec = sigma .* (1 - sigma);               % N x 1, entries in (0,0.25]

        % Efficient block computation without forming D:
        % X'*(D*X) = X'*(X .* dvec)
        Xd  = X .* dvec;                           % N x d (row-wise scaling)
        Hww = (1/N) * (X' * Xd);                   % d x d
        Hwb = (1/N) * (X' * dvec);                 % d x 1
        Hbb = (1/N) * sum(dvec);                   % scalar

        H = [Hww, Hwb;
             Hwb', Hbb];

        if mu > 0
            H = H + (2*mu) * eye(d+1);
        end
    end
end
