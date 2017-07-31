function [B] = LaplaceSOR2D(A,d,w)

% Approximates the Laplacian on a 2-dimensional mesh using SOR iteration
% A: 2D array such that boundary condition is specified on boundary points
% tol: Norm tolerance
% w: Relaxation parameter

B=A;
l=size(B);
% Perform one iteration outside of the while statement so that B!=A
for j=2:l(1)-1
    for k=2:l(2)-1
        B(j,k)=A(j,k)+w*(A(j+1,k)+A(j-1,k)+A(j,k+1)+A(j,k-1)-4*A(j,k));
    end
end

while ((abs(sum(sum(B))-sum(sum(A))) > d))
    A=B;
    for j=2:l(1)-1
        for k=2:l(2)-1
            B(j,k)=A(j,k)+w*(A(j+1,k)+B(j-1,k)+A(j,k+1)+B(j,k-1)-4*A(j,k));
        end
    end
    imagesc(B);
    pause(.1);
    drawnow();
end