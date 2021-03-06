function [B] = PoissonSOR2D(A,d,F,dx,w)

% Approximates Poisson eq on a 2-dimensional mesh using G-S iteration
% A: 2D array such that boundary condition is specified on boundary points
% tol: Norm tolerance
% F: discretization of source f(x,y)
% dx: Change in x (assumed to be == dy)
% w: Relaxtion parameter

A=A+F;
B=A;
l=size(B);
% Need B!=A
for j=2:l(1)-1
    for k=2:l(2)-1
        B(j,k)=w*(A(j+1,k)+B(j-1,k)+A(j,k+1)+B(j,k-1)-dx^2*F(j,k));
    end
end

while ((abs(sum(sum(B))-sum(sum(A)))) > d)
    A=B;
    for j=2:l(1)-1
        for k=2:l(2)-1
            B(j,k)=w*(A(j+1,k)+B(j-1,k)+A(j,k+1)+B(j,k-1)-dx^2*F(j,k));
        end
    end
    imagesc(B);
    pause(.01);
    drawnow();
end