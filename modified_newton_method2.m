function [xk,fk,gradfk_norm,k,xseq,btseq,tau_new,alphas,pks] = modified_newton_method2(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta)

farmijo = @(fk, alpha, c1_gradfk_pk) fk + alpha * c1_gradfk_pk;

tol = 1e-6;
maxit = 100;

xseq = zeros(length(x0), kmax);
btseq = zeros(1, kmax);
alphas = zeros(1, kmax);
pks = zeros(length(x0), kmax);
tau_new = zeros(maxit+1, kmax);

xk = x0;
fk = f(xk);
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);
k = 0;

while k < kmax && gradfk_norm >= tolgrad

    % Hessiana e correzione
    Hk = hessf(xk);
    diagHk = diag(Hk);
    isPositive = all(diagHk > tol);

    tau_k = 0;
    if ~isPositive
        tau_k = beta - min(diagHk);
    end

    tauk = zeros(maxit+1,1);
    tauk(1) = tau_k;

    for j = 1:maxit
        Bk = Hk + tauk(j)*eye(size(Hk));
        [R,flag] = chol(Bk);
        if flag == 0
            break
        else
            tauk(j+1) = max(beta, 2*tauk(j));
        end
    end

    % Direzione di Newton modificata
    pk = R\(R'\(-gradfk));

    % Backtracking
    alpha = 1;
    xnew = xk + alpha * pk;
    fnew = f(xnew);

    c1_gradfk_pk = c1 * gradfk' * pk;
    bt = 0;

    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        alpha = rho * alpha;
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        bt = bt + 1;
    end

    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        disp("Backtracking: massimo raggiunto");
        break;
    end

    % Aggiornamento
    k = k + 1;
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);

    % Salvataggi
    xseq(:, k)   = xk;
    btseq(k)     = bt;
    alphas(k)    = alpha;
    pks(:, k)    = pk;
    tau_new(:,k) = tauk;

end

% Taglio matrici
xseq   = xseq(:, 1:k);
btseq  = btseq(1:k);
alphas = alphas(1:k);
pks    = pks(:, 1:k);

% Inserisco x0 come primo punto per l'animazione
xseq = [x0, xseq];

end
