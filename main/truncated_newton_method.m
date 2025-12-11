function [xk,fk,gradfk_norm,k,xseq,btseq,pks,inner_iters] = truncated_newton_method(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,max_cg) 
% TRUNCATED_NEWTON_METHOD  
%   [xk,fk,gradfk_norm,k,xseq,btseq, pseq, inner_iters] =
%       truncated_newton_method(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,max_cg)
%
% Input arguments:
%   x0, f, gradf, hessf: initial guess, function, gradient of f, hessian
%   matrix of f
%   kmax: max outer iterations
%   max_xg: max conjugate gradient iterations, i.e. inner iterations
%   tolgrad: stopping tolerance on ||grad f||
%   c1, rho, btmax: Armijo/backtracking parameters
%   non so se ci vuole beta input del modified


    % Function handle for the armijo condition
    farmijo = @(fk, alpha, c1_gradfk_pk) ...
        fk + alpha * c1_gradfk_pk;
    
    
    
    xseq = zeros(length(x0), kmax);
    btseq = zeros(1, kmax);
    alphas = zeros(1, kmax);
    pks = zeros(length(x0), kmax);
    inner_iters = zeros(1, kmax); %vettore che tiene conto delle iterazioni interne di ogni interazione esterna k
    inner_it = 0; %indice che tiene conto delle iterazioni
    
    xk = x0;
    fk = f(xk);
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    k = 1;

      
    
    

    while k <= kmax && gradfk_norm >= tolgrad
        
        z = zeros(length(x0),1); %si deve resettare ad ogni iterazione k
        p_tn = []; % p_tn is the variable containing the final descent direction
        j = 0;
      
        Hk = hessf(xk);
        Bk = Hk;
        ck = -gradfk;
        eta_k = min(0.5, sqrt(gradfk_norm));
        rk = ck-Bk*z; % Residual of the system, at the first iteration z = 0 so rk = ck
        rk_old = rk'*rk;
        dk = rk; % d is the conjugate direction, at the first iteration z = 0 so dk = ck

        stop_inner = false; % Boolean variable used to understand whether the inner loop got to convergence, it has to go back to false at each iteration k

        while ~stop_inner && j < max_cg

            j = j+1;

            curv = dk'*Bk*dk;
            
            if curv > 0

                alpha_j = (rk'*rk)/(curv);
                z = z + alpha_j * dk;
                rk = rk - alpha_j * Bk * dk; % chat dice che va il - ma la pieraccini ha scritto +

                % check on convergence
                if norm(rk) <= eta_k * norm(ck)

                    p_tn = z;      
                    inner_it = j;      
                    stop_inner = true;

                break;

            end

                rk_new = rk'*rk;

                beta_j = rk_new/rk_old;
                dk = rk+beta_j*dk;
                rk_old = rk_new; %aggiornamento per il passo successivo
            else
                if j == 1 % sarebbe il caso j = 0 ma ho gia aggiornato j
                    p_tn = -gradfk;
                    
                else
                    p_tn = z;
                end
                inner_it = j;
                stop_inner = true;

                break
            end

            
        end

        inner_iters(k) = inner_it;
        pk = p_tn;
        
        % BACKTRACKING
        if norm(pk) == 0
            disp('Truncated Newton: null direction, stop.');
            return;
        end

        alpha = 1;
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        c1_gradfk_pk = c1 * (gradfk' * pk);
        bt = 0;

        while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
            alpha = rho * alpha;        % riduzione passo
            xnew = xk + alpha * pk;
            fnew = f(xnew);
            bt = bt + 1;
        end
        
        if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
            disp('Backtracking: massimo raggiunto (Truncated Newton)');
            break;            
        end
                
        % AGGIORNAMENTO VARIABILI
        xk = xnew;
        fk = fnew;
        gradfk = gradf(xk);
        gradfk_norm = norm(gradfk);
        k = k + 1;
        
        xseq(:, k) = xk;
        btseq(k) = bt;
        pks(:, k) = pk;
        alphas(k) = alpha;


    end %while loop on k

% Taglio matrici
xseq   = xseq(:, 1:k);
btseq  = btseq(1:k);
alphas = alphas(1:k);
pks    = pks(:, 1:k);

% Inserisco x0 come primo punto per l'animazione
xseq = [x0, xseq];




end %function end