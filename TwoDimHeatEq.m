function[A] = TwoDimHeatEq(s,A,m)

% Approximates the 2D heat equation using a forward difference in time, and
% a centered difference in x and y. Assumes dx=dy.
% s: The product k*dt/(dx)^2. Stability analysis reveals that this scheme
% is stable if s <= 1/4.
% A: The initial discretization in the form of a 2D array.
% m: Number of time-steps to perform.

n=size(A);

for l=1:m
    for i=2:n(1)-1
        for j=2:n(2)-1
            A(i,j)=s*(A(i+1,j)+A(i-1,j)+A(i,j+1)+A(i,j-1));
        end
    end
end