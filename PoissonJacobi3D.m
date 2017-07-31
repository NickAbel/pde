function [B] = PoissonJacobi3D(A,d,F,dx)

% Approximates Poisson eq on a 3-dimensional mesh using Jacobi iteration
% A: 2D array such that boundary condition is specified on boundary points
% tol: Norm tolerance
% F: Mesh of source f(x,y,z)
% dx: Change in x, assumed dy,dz are equal

A=A+F;
B=A;
S=size(B);
% Need B!=A
for j=2:S(1)-1
    for k=2:S(2)-1
        for l=2:S(3)-1
          B(j,k,l)=(1/6)*(A(j+1,k,l)+A(j-1,k,l)+A(j,k+1,l)+A(j,k-1,l)+A(j,k,l+1)+A(j,k,l-1)-dx^2*F(j,k,l));
        end
    end
end

while ((abs(sum(sum(sum(B)))-sum(sum(sum((A)))))) > d) % Replace with norm (seriously)
    A=B;
    for j=2:S(1)-1
        for k=2:S(2)-1
            for l=2:S(3)-1
                B(j,k,l)=(1/6)*(A(j+1,k,l)+A(j-1,k,l)+A(j,k+1,l)+A(j,k-1,l)+A(j,k,l+1)+A(j,k,l-1)-dx^2*F(j,k,l));
            end
        end
    end
    isosurface(B);
    pause(.01);
    drawnow();
end