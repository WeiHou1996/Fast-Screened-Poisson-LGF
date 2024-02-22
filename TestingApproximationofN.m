clc
clear all
close all


c2 = 0.01^2;
eps = 1e-14;

%x = [alpha, N]

errfunc = @(x) x(2)^2;


%bnd_nl = @(x) bnd(x(1), x(2));


A = [1 0;-1 0;];
b = [sqrt(c2), 0];
Aeq = [];
beq = [];
x0 = [sqrt(c2)/2, 10];
nonconF = @(x) bnd_nl(x, c2, eps);
%noncon = @nonconF
x = fmincon(errfunc, x0, A, b, Aeq, beq, [],[], nonconF)

%%
M_a = @(a) 1 / 2 / sqrt(c2 - a^2);
gamma_a = @(a) log(1 + a^2/2 + sqrt((1+a^2/2)^2 - 1));
ep = 0.01;
a_ad = (1 - ep)*sqrt(c2);
N_ad = (log(4*pi*M_a(a_ad)) - log(eps))/gamma_a(a_ad)

%%
options = optimoptions('fmincon','MaxFunctionEvaluations',1e6,'Algorithm','sqp');

C2_vec = logspace(-3, 0, 10);
N_c2_vec = length(C2_vec);
N_vec_true = zeros(1, N_c2_vec);
N_vec_app = zeros(1, N_c2_vec);
for i = 1:N_c2_vec
    c2 = C2_vec(i)^2
    eps = 1e-14;
    
    a_ad = (1 - ep)*sqrt(c2);

    M_a = @(a) 1 / 2 / sqrt(c2 - a^2);
    gamma_a = @(a) log(1 + a^2/2 + sqrt((1+a^2/2)^2 - 1));
    N_ad = (log(4*pi*M_a(a_ad)) - log(eps))/gamma_a(a_ad);

    nonconF = @(x) bnd_nl(x, c2, eps);
    %noncon = @nonconF
    A = [1 0;-1 0;];
    b = [sqrt(c2), 0];
    Aeq = [];
    beq = [];
    x0 = [sqrt(c2)/2, 1/sqrt(c2)];
    x = fmincon(errfunc, x0, A, b, Aeq, beq, [],[], nonconF, options);

    N_vec_true(i) = x(2);
    N_vec_app(i) = N_ad;
end
%%
loglog(C2_vec, N_vec_true,'o')
hold on
semilogx(C2_vec, N_vec_app)
ylim([min(N_vec_app), max(N_vec_app)])

%%
function [c, ceq] = bnd_nl(x, c2, eps)

M_a = @(a) 1 / 2 / sqrt(c2 - a^2);
gamma_a = @(a) log(1 + a^2/2 + sqrt((1+a^2/2)^2 - 1));
bnd = @(x) -(gamma_a(x(1))*x(2) - log(4*pi*M_a(x(1))) + log(eps));
ceq = [];
c(1) = bnd(x);
%c(1) = x(1) * x(2);
end
