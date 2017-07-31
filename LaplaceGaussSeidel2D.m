function [B] = LaplaceGaussSeidel2D(A,d)

% Approximates the Laplacian on a 2-dimensional mesh using G-S iteration
% A: 2D array such that boundary condition is specified on boundary points
% tol: Norm tolerance

B=A;
l=size(B);
% Need B!=A
for j=2:l(1)-1
    for k=2:l(2)-1
        B(j,k)=.25*(A(j+1,k)+B(j-1,k)+A(j,k+1)+B(j,k-1));
    end
end

while ((abs(sum(sum(B))/length(B)^2-sum(sum(A))/length(A)^2)) > d)
    A=B;
    for j=2:l(1)-1
        for k=2:l(2)-1
            B(j,k)=.25*(A(j+1,k)+B(j-1,k)+A(j,k+1)+B(j,k-1));
        end
    end
    imagesc(B);
    pause(.001);
    drawnow();
end