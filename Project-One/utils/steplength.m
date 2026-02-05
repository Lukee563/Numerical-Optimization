function alpha = steplength(problem, theta, p, opts)

% Line search satisfying the (strong) Wolfe conditions (Nocedal–Wright).
%
%   alpha = steplength(problem, theta, p, opts)
%
% This routine implements the structure of Nocedal & Wright:
%   Algorithm 3.5 (Line Search Algorithm) + Algorithm 3.6 (Zoom)
% for the *strong Wolfe* conditions on the 1D function
%       phi(a) = f(theta + a p),
% where f is the loss function defined in problem.obj
% The routine has an internal safeguard alpha >= alpha_min.
%
% Strong Wolfe conditions:
%   (W1) Sufficient decrease (Armijo):
%         phi(a) <= phi(0) + c1 * a * phi'(0)
%   (W2) Strong curvature:
%         |phi'(a)| <= c2 * |phi'(0)|
%
% Required fields in "problem":
%   problem.Xtr : N x d training data
%   problem.ytr : N x 1 labels in {0,1}
%   problem.mu  : L2 regularization parameter
%   problem.obj : objective handle, define the cost, grad, and Hessian (optional)               
%                 [f,g,H] = obj(theta, X, y, mu, mode)
%                 Modes used here:
%                   mode=0 loss only
%                   mode=1 grad only
%                   mode=2 loss+grad 
%
% Inputs:
%   theta : current iterate (column vector)
%   p     : search direction (column vector); must satisfy g0'*p < 0
%   opts  : struct (optional):
%       c1        (default 1e-4)
%       c2        (default 0.9)       % for NCG use c2 < 0.5; for GD/Newton 0.9 is fine
%       alpha0    (default 1)
%       alpha_max (default 50)
%       alpha_min (1e-6)              % internal safeguard
%       maxIter   (default 50)        % Alg 3.5 outer iterations
%       maxZoom   (default 50)        % Alg 3.6 zoom iterations
%
% Output:
%   alpha : chosen step size; returns 0 if p is not a descent direction.

% ---------------- Defaults ----------------
if nargin < 4 || isempty(opts), opts = struct(); end
c1        = get_opt(opts, 'c1', 1e-4);
c2        = get_opt(opts, 'c2', 0.9);
alpha     = get_opt(opts, 'alpha0', 1.0);
alpha_max = get_opt(opts, 'alpha_max', 50.0);
alpha_min = get_opt(opts, 'alpha_min', 1e-6);
maxIter   = get_opt(opts, 'maxIter', 50);
maxZoom   = get_opt(opts, 'maxZoom', 50);

% ---------------- Problem components ----------------
X  = problem.Xtr;
y  = problem.ytr(:);
mu = problem.mu;
obj = problem.obj;

theta = theta(:);
p     = p(:);

% ---------------- Evaluate phi(0), phi'(0) ----------------
[phi0, g0] = obj(theta, X, y, mu, 2);  % loss + grad
dphi0 = g0' * p;

% Descent safeguard: if p is not a descent direction, line search is not applicable.
% Return alpha_min so the outer algorithm can still make progress (or detect issues).
if ~isfinite(phi0) || any(~isfinite(g0)) || dphi0 >= 0
    alpha = alpha_min;
    return;
end

alpha_prev = 0;
phi_prev   = phi0;

% ---------------- Algorithm 3.5: Line search ----------------
for i = 1:maxIter
    th  = theta + alpha*p;
    phi = obj(th, X, y, mu, 0);        % loss only

    % Robustness: treat invalid phi as "too large" and zoom on (prev, curr)
    if ~isfinite(phi)
        alpha = zoom_strong_wolfe(obj, X, y, mu, theta, p, ...
                                  alpha_prev, alpha, phi0, dphi0, c1, c2, maxZoom, alpha_min);
        alpha = enforce_alpha(alpha, alpha_min);
        return;
    end
    
    % Condition (Alg 3.5): Armijo failure OR (i>1 and phi(alpha_i) >= phi(alpha_{i-1}))
    if (phi > phi0 + c1*alpha*dphi0) || (i > 1 && phi >= phi_prev)
        alpha = zoom_strong_wolfe(obj, X, y, mu, theta, p, ...
                                  alpha_prev, alpha, phi0, dphi0, c1, c2, maxZoom, alpha_min);
        alpha = enforce_alpha(alpha, alpha_min);
        return;
    end

    % Derivative at current alpha
    [~, g] = obj(th, X, y, mu, 1);     % grad only
    dphi = g' * p;
   
    % If strong curvature condition satisfied, accept alpha_i
    if isfinite(dphi) && abs(dphi) <= c2 * abs(dphi0)
        alpha = enforce_alpha(alpha, alpha_min);
        return;
    end

    % If derivative becomes nonnegative -> zoom(curr, prev) (note the order)
    if ~isfinite(dphi) || dphi >= 0
        alpha = zoom_strong_wolfe(obj, X, y, mu, theta, p, ...
                                  alpha, alpha_prev, phi0, dphi0, c1, c2, maxZoom, alpha_min);
        alpha = enforce_alpha(alpha, alpha_min);
        return;
    end

    % Otherwise increase step
    alpha_prev = alpha;
    phi_prev   = phi;
    alpha      = min(alpha_max, 2*alpha);
end

% Best effort if maxIter reached
alpha = enforce_alpha(alpha, alpha_min);
end


% ======================================================================

function alpha = zoom_strong_wolfe(obj, X, y, mu, theta, p, ...
                                   alo, ahi, phi0, dphi0, c1, c2, maxZoom, alpha_min)
%ZOOM_STRONG_WOLFE  Algorithm 3.6 (Zoom) for strong Wolfe conditions.
% This implementation follows the logical structure of Nocedal–Wright Alg. 3.6.

phi_alo = obj(theta + alo*p, X, y, mu, 0);
if ~isfinite(phi_alo), phi_alo = phi0; end

for j = 1:maxZoom
    alpha = 0.5*(alo + ahi);

    % Interval collapsed numerically
    if alpha == alo || alpha == ahi
        alpha = alo;
        alpha = enforce_alpha(alpha, alpha_min);
        return;
    end

    th  = theta + alpha*p;
    phi = obj(th, X, y, mu, 0);

    if ~isfinite(phi) || (phi > phi0 + c1*alpha*dphi0) || (phi >= phi_alo)
        ahi = alpha;
    else
        [~, g] = obj(th, X, y, mu, 1);
        dphi = g' * p;

        if isfinite(dphi) && abs(dphi) <= c2 * abs(dphi0)
            alpha = enforce_alpha(alpha, alpha_min);
            return;
        end

        if ~isfinite(dphi) || dphi*(ahi - alo) >= 0
            ahi = alo;
        end

        alo = alpha;
        phi_alo = phi;
    end
end

% Best effort if maxZoom reached
alpha = enforce_alpha(alo, alpha_min);
end


function a = enforce_alpha(a, alpha_min)
% Ensure returned step is finite and not too small.
if ~isfinite(a) || a <= 0
    a = alpha_min;
else
    a = max(a, alpha_min);
end
end


function v = get_opt(opts, name, defaultValue)
if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
    v = opts.(name);
else
    v = defaultValue;
end
end
