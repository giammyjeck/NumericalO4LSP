clc
close all
clear

load test_functions2.mat

beta = 1e-3;
[xknew,fk,gradnorm,k,xseq,btseq] = modified_newton_method( ...
        x0, f1, gradf1, Hessf1, kmax, tolgrad, c1, rho, btmax, beta);
xknew