% In this program, we implement the shooting bisection method to find
% b-solutions for the corresponding GP problem, solving the ODEs
% numerically using the 4th order Runge-Kutta method, in a given range of
% b-values. The numerics also save values of lambda(b) and the masses of
% the b-solutions (ground states)

clear all;
close all;

% Power p, dimension d, interval [rmin, rmax] of solution
% p = 1;
% d = 2/p + 2;
d = 4;
p = 2/(d-2);
alpha_p = p^2/(4*(1+p));
rmin = 10^(-5);
rmax = 8;

% 'tolmin' decides how close to zero the value f(rmax) has to be in order to
% accept the solution of the IVP as the b-solution, 'maxiter' is the
% maximum number of iteration when trying to find the b-solution, and 'last_val'
% is the current value of f(rmax), changing every iteration of the loop and initially set to be 1 
tolmin = 10^(-3);
maxiter = 10^4;
last_val = 1;

% Distribution of b-values
b0 = 10^1;
bmax = 10^5;
logbvals = linspace(log(b0),log(bmax),10^2);
bvals = exp(logbvals);
bvals = bvals(1:68);

% lambda(b) values stored in 'lamvals', mass of b-solutions stored in
% 'mass'
lamvals = zeros(1,length(bvals));
mass = zeros(1,length(bvals));

% System of equation corresponding ot the 2nd order GP equation with
% harmonic potential
f = @(f1,f2,r) f2;
% g = @(f1,f2,r,lambda) (1-d)./r.*f2 + r.^2.*f1 - lambda.*f1 - f1.^(1+2*p);
g = @(f1,f2,r,lambda) (1-d)./r.*f2 + r.^2.*f1 - lambda.*f1 - f1.*abs(f1).^(2*p);

% Step size h for the 4th order Runge-Kutta method
h = 10^(-6);

% Vector 'r' of values of the independent variable r, b-solutions stored in
% 'bsols'
r = rmin:h:rmax;
N = length(r);
bsols = zeros(length(bvals),N);

% Iterate over b-values in 'bvals'
tic;
for j = 1:length(bvals)
    disp(j)
    b = bvals(j);
    if mod(j,100)==0
        disp(j)
    end
    
    lam_min = 0;
    lam_max = d;
    lam = (lam_min + lam_max)/2;
    
    last_val = 1;
    iter = 1;
    while abs(last_val) > tolmin && iter <= maxiter
        % 4th order RK method used to solve the system f1'=f, f2'=g on the
        % interval [rmin, rmax] with step size 'h'. The method stops for a
        % given r_n \in [rmin, rmax] either if f1(r_{n+1})>f1(r_n), or if
        % f1(r_{n+1})<-tolmin. In the former case the values of lam and
        % lam_min get adjusted as: lam_min = lam, lam = (lam + lam_max)/2,
        % and in the latter case, as lam_max = lam, lam = (lam_min + lam)/2
        
        % Define empty vectors f1, f2 with initial values f1(1)=b, f2(1)=0
        f1 = zeros(1,N);
        f2 = zeros(1,N);
        f1(1) = b;
        f2(1) = 0;
        
        change_lam = 0;
        i = 1;
        
        while i<=(N-1) && ~change_lam
            
            k0 = h*f(f1(i),f2(i),r(i));
            l0 = h*g(f1(i),f2(i),r(i),lam);
            
            k1 = h*f(f1(i)+1/2*k0,f2(i)+l0/2,r(i)+h/2);
            l1 = h*g(f1(i)+1/2*k0,f2(i)+l0/2,r(i)+h/2,lam);
            
            k2 = h*f(f1(i)+k1/2,f2(i)+l1/2,r(i)+h/2);
            l2 = h*g(f1(i)+k1/2,f2(i)+l1/2,r(i)+h/2,lam);
            
            k3 = h*f(f1(i)+k2,f2(i)+l2,r(i)+h);
            l3 = h*g(f1(i)+k2,f2(i)+l2,r(i)+h,lam);
            
            f1(i+1) = f1(i) + 1/6*(k0 + 2*k1 + 2*k2 + k3);
            f2(i+1) = f2(i) + 1/6*(l0 + 2*l1 + 2*l2 + l3);
            
            if f1(i+1) > f1(i)
                lam_min = lam;
                lam = (lam + lam_max)/2;
                change_lam = 1;
            elseif f1(i+1)<-tolmin
                lam_max = lam;
                lam = (lam_min + lam)/2;
                change_lam = 1;
            end
            last_val = f1(i+1);
            i = i+1;
        end
        iter = iter + 1;
    end
    lamvals(j) = lam;
    mass(j) = trapz(r,f1);
    bsols(j,:) = f1;
end
toc;

% Estimate of the power alpha in the lambda(b)~b^alpha dependence
slope = (log(lamvals(end))-log(lamvals(end-10)))/(log(bvals(end))-logbvals(end-10));

% Estimate based on 10 points, skipping the ones that are equal
loglam = log(lamvals);
[loglam_dist, ind_loglam_dist] = unique(loglam);


figure;
plot(r,f1);
xlabel('r');
ylabel('f');

figure;
plot(bvals,lamvals,'.')
xlabel('b');
ylabel('lambda');

figure;
plot(lamvals,mass,'.');
xlabel('lambda');
ylabel('mass of u_b');

figure;
plot(log(bvals),log(lamvals),'.');
xlabel('log(b)');
ylabel('log(lambda(b))');
title(['p=' num2str(p) ' d=' num2str(d) ' slope=' num2str(slope)])

% Plot for 0 < p < 1/2
% figure;
% hold on;
% plot(log(bvals),log(lamvals),'.','LineWidth',2);
% plot(log(bvals),-2*p*log(bvals) + 6,'r','LineWidth',2);
% xlabel('$$\log(b)$$','Interpreter','Latex');
% ylabel('$$\log(\lambda(b))$$','Interpreter','Latex');
% title(['$$p=$$ ' num2str(p) ', $$d=$$ ' num2str(d) ', estimated slope=' num2str(slope)],'Interpreter','Latex')
% legend('$$\log(\lambda(b))$$', strcat(num2str(round(-2*p,2)),'$$\log(b)+\log(C_p)$$'),'Interpreter','Latex');
% box on;

% Plot for d=6.5, p=0.444
% figure;
% hold on;
% plot(log(bvals),log(lamvals),'.','LineWidth',2);
% plot(log(bvals),-2*p*log(bvals) + 6.375,'r','LineWidth',2);
% xlabel('$$\log(b)$$','Interpreter','Latex');
% ylabel('$$\log(\lambda(b))$$','Interpreter','Latex');
% title(['$$p=$$ ' num2str(p) ', $$d=$$ ' num2str(d) ', estimated slope=' num2str(round(b_s,2))],'Interpreter','Latex')
% legend('$$\log(\lambda(b))$$', strcat(num2str(round(-2*p,2)),'$$\log(b)+\log(C_p)$$'),'Interpreter','Latex');
% box on;

% Plot for 1/2 < p < 1
% figure;
% hold on;
% plot(log(bvals),log(lamvals),'.','LineWidth',2);
% plot(log(bvals),-2*(1-p)*log(bvals),'r','LineWidth',2);
% xlabel('$$\log(b)$$','Interpreter','Latex');
% ylabel('$$\log(\lambda(b))$$','Interpreter','Latex');
% title(['$$p=$$' num2str(p) ', $$d=$$' num2str(d) ', estimated slope=' num2str(slope)],'Interpreter','Latex')
% legend('$$\log(\lambda(b))$$','$$-1.2\log(b)$$','Interpreter','Latex');
% box on;

% Plot for p = 1
% figure;
% hold on;
% plot(bvals,lamvals,'.','LineWidth',2);
% plot(bvals,1.5./log(bvals),'r','LineWidth',2);
% xlabel('$$b$$','Interpreter','Latex');
% ylabel('$$\lambda(b)$$','Interpreter','Latex');
% title(['$$p=$$' num2str(p) ', $$d=$$' num2str(d)],'Interpreter','Latex')
% legend('$$\lambda(b)$$','$$\frac{C_p}{\log(b)}$$','Interpreter','Latex');
% box on;
