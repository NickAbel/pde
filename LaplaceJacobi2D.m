function [B] = LaplaceJacobi2D(A,d)

% Approximates the Laplacian on a 2-dimensional mesh using Jacobi iteration
% A: 2D array such that boundary condition is specified on boundary points
% tol: Norm tolerance

B=A;
l=size(B);
% Perform one iteration outside of the while statement so that B!=A
for j=2:l(1)-1
    for k=2:l(2)-1
        B(j,k)=.25*(A(j+1,k)+A(j-1,k)+A(j,k+1)+A(j,k-1));
    end
end

while ((abs(sum(sum(B))-sum(sum(A)))) > d)
    A=B;
    for j=2:l(1)-1
        for k=2:l(2)-1
            B(j,k)=.25*(A(j+1,k)+A(j-1,k)+A(j,k+1)+A(j,k-1));
        end
    end
    imagesc(B);
    pause(.1);
    drawnow();
end