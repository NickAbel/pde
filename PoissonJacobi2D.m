function [B] = PoissonJacobi2D(A,d,F,dx)

% Approximates Poisson on a 2-dimensional mesh using Jacobi iteration
% A: 2D array such that boundary condition is specified on boundary points
% F: Discretization of source f(x,y)
% tol: Norm tolerance

A=A+F;
B=A;
l=size(B);
% Need B!=A
for j=2:l(1)-1
    for k=2:l(2)-1
        B(j,k)=.25*((A(j+1,k)+A(j-1,k)+A(j,k+1)+A(j,k-1)-F(j,k)*(dx)^2));
    end
end

while ((abs(sum(sum(B))/length(B)^2-sum(sum(A))/length(A)^2)))
    A=B;
    for j=2:l(1)-1
        for k=2:l(2)-1
            B(j,k)=(A(j+1,k)+A(j-1,k)+A(j,k+1)+A(j,k-1)-F(j,k)*(dx)^2)/4;
        end
    end
    imagesc(B);
    pause(.01);
    drawnow();
end