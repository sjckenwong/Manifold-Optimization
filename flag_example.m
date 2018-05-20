% Problem: maximize f(X) = tr(X'AX), where A is a 60 x 60 symmetric 
% matrix and X is 60 x 12 matrix with orthnormal columns
%
% We can consider X \in Flag(4,7,12; R^60) as this function is also invariant
% to right multiplication by O(4)xO(3)xO(5) in particular.
%
% In the actual calculation, we complete X to be a 60 x 60 orthogonal
% matrix.
%
% We compare the solution with the sum of the 12 largest eigenvalues of A.

clear all
clc


A = randn(60);
A = 0.5 * (A+A'); % turn A symmetric

% dimensions of sequences of subspaces
n = [0,4,7,12];

% random initalization
[X0,~] = qr(randn(60));

X = X0;
fX = trace(X(:,1:12)'*A*X(:,1:12));


% stopping criteria
iter = 0; % number of iterations
norm_RG = 1; % norm of Riemannian gradient
dist = 1; % Grassmann distance between X_{i} and X_{i+1}

while iter < 1000 && norm_RG > 1e-6 && dist > 1e-5
    
    % Euclidean gradient
    EG = 2*A*X;
    % Riemannian gradient
    RG = zeros(60,12);
    
    % sum_{j=1}^{3} A_j*D_j'
    sumAjDj = zeros(60);
    for i = 1:3
        Ai = X(:,n(i)+1:n(i+1));
        Di = EG(:,n(i)+1:n(i+1));
        sumAjDj = sumAjDj + Ai*Di';
    end
    
    % compute Riemannian gradient using equation (9)
    for i = 1:3
        Ai = X(:,n(i)+1:n(i+1));
        Di = EG(:,n(i)+1:n(i+1));
        RG(:,n(i)+1:n(i+1)) = Di - (Ai*Ai'*Di + (sumAjDj - Ai*Di')*Ai);
    end
    
    B = zeros(60,60);
    B(:,1:12) = X'*RG;
    
    % turn B skew-symmetric
    B(1:12,13:60) = -B(13:60,1:12)';
   
    
    t = 1;
    beta = 0.5;
    Xnew = X*expm(t*B);
    [Xnew,~] = qr(Xnew); % reorthnormalize Xnew
    
    % backtracking line search
    % if Armijo?Goldstein condition is imposed, iteration stops too early
    % and can't converge to optimal
    while t > 1e-6 && trace(Xnew(:,1:12)'*A*Xnew(:,1:12)) < fX 
        % + 0.5*t*norm(RG)
        t = beta*t;
        Xnew = X*expm(t*B);
        [Xnew,~] = qr(Xnew);

    end
    
    
    norm_RG = norm(RG);
    dist = distance(X(:,1:12),Xnew(:,1:12));
    iter = iter+1;
    
    X = Xnew;
    fX = trace(X(:,1:12)'*A*X(:,1:12));
    fprintf('i = %d, step size = %f, dist(X_{i}, X_{i+1}) = %f, f(X): %f\n', iter, t, dist, fX)
    
end

fprintf('\n') 
eigs = eig(A);
eigs = sort(eigs,'descend');
fprintf('sum of 12 largest eigenvalues of A = %f\n',sum(eigs(1:12)))


