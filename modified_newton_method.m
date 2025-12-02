function [xk,fk,gradfk_norm,k,xseq,btseq] = modified_newton_method(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta)

% Modified Newton method for numerical optimization problems 
%   This functions is aimed to solve large scale numerical optimization
%   problems using the modified Newton method with backtracking techniques.

% Function handle for the armijo condition
farmijo = @(fk, alpha, c1_gradfk_pk) ...
    fk + alpha * c1_gradfk_pk;


% Initializations
xseq = zeros(length(x0), kmax);
btseq = zeros(1, kmax);
tol = 1e-6; % tol to check the positivness of Hk diagonal
maxit = 100; % max number of iteration to check the positivness of Hk diagonal


xk = x0;
fk = f(xk);
k = 0;
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);
tau_new = zeros(maxit+1,kmax); %matrix to sake the values of tau_k


while k < kmax && gradfk_norm >= tolgrad
    
    % Compute the hessian matrix
    Hk = hessf(xk);
    
    % Check if the hessian matrix is positive definite
    diagHk = diag(Hk);
    isPositive = all(diagHk>tol);

    if isPositive == true
        tau_k = 0;
    else
        tau_k = beta-min(diagHk); %check se va messo max(0,)

    end
     
    %Bk = Hk+tau_k*eye(size(Hk));

    % Once the parameter tau is found then the correction of the hessian 
    % can be built.
    % In order to check if the corrected matrix is positive definite, we 
    % attempt to perform the incomplete choleski factorization: 
    % if Bk is not positive definite, then you get an error.

    tauk = zeros(maxit+1,1);
    tauk(1)= tau_k;

    for j = 1:maxit
            Bk = Hk+tauk(j)*eye(size(Hk));
            [R,flag] = chol(Bk);
        % Chech if the correction is good enough (Is Bk positive definite?)
            if flag == 0
                disp('Bk is positive definite')
                break
            elseif flag == 1
        
                % If the Bk in not positive definite, then we need to correct it.
                % new value of tau_k
        
                tauk(j+1) = max(beta, 2*tauk(j));
            else
                disp('Error')
           
            end
    end

    % Once we obtain a matrix Bk which is positve definite we can solve the
    % following system with pcg method in order to find the 
    % descent direction.
     
    pk = pcg(Bk, -gradfk, [], [], R, R');

    % NOTE: there's no need to check if pk is a descent direction because
    % of remark2 
    
    
    % Reset the value of alpha
    alpha = 1;
    
    % Compute the candidate new xk
    xnew = xk + alpha * pk;
    % Compute the value of f in the candidate new xk
    fnew = f(xnew);
    
    c1_gradfk_pk = c1 * gradfk' * pk;
    bt = 0;

    % Backtracking strategy from here...: 
    % 2nd condition is the Armijo condition not satisfied
    
    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        % Reduce the value of alpha
        alpha = rho * alpha;
        % Update xnew and fnew w.r.t. the reduced alpha
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        
        % Increase the counter by one
        bt = bt + 1;
    end
    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        break
    end
    % ...to here
    
    % Update xk, fk, gradfk_norm
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    
    % Increase the step by one
    k = k + 1;
    
    % Store current xk in xseq
    xseq(:, k) = xk;
    % Store bt iterations in btseq
    btseq(k) = bt;
    %Store current tauk values
    tau_new(:, k) = tauk;
end


% "Cut" xseq and btseq to the correct size
xseq = xseq(:, 1:k);
btseq = btseq(1:k);
% "Add" x0 at the beginning of xseq (otherwise the first el. is x1)
xseq = [x0, xseq];


end