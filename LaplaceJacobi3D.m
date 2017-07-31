function [B] = LaplaceJacobi3D(A,d)

% Approximates Laplace eq on a 3-dimensional mesh using Jacobi iteration
% A: 2D array such that boundary condition is specified on boundary points
% tol: Norm tolerance

B=A;
S=size(B);
% Need B!=A
for j=2:S(1)-1
    for k=2:S(2)-1
        for l=2:S(3)-1
          B(j,k,l)=(1/6)*(A(j+1,k,l)+A(j-1,k,l)+A(j,k+1,l)+A(j,k-1,l)+A(j,k,l+1)+A(j,k,l-1));
        end
    end
end

while ((abs(sum(sum(sum(B)))-sum(sum(sum((A)))))) > d) % Replace with norm (seriously)
    A=B;
    for j=2:S(1)-1
        for k=2:S(2)-1
            for l=2:S(3)-1
                B(j,k,l)=(1/6)*(A(j+1,k,l)+A(j-1,k,l)+A(j,k+1,l)+A(j,k-1,l)+A(j,k,l+1)+A(j,k,l-1));
            end
        end
    end
    isosurface(B);
    pause(.01);
    drawnow();
end