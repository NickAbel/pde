function[A] = TwoDimHeatEq(s,A,m)

% Approximates the 2D heat equation using a forward difference in time, and
% a centered difference in x and y. Assumes dx=dy.
% s: The product k*dt/(dx)^2. Stability analysis reveals that this scheme
% is stable if s <= 1/4.
% A: The initial discretization in the form of a 2D array.
% m: Number of time-steps to perform.


n=size(A);
B=A;
for i=1:m
  for j=2:n(1)-1
    for l=2:n(2)-1
      B(j,l)=s*(A(j+1,l)+A(j-1,l)+A(j,l+1)+A(j,l-1))+(1-4*s)*A(j,l);
    end
  end
  A=B;
  imagesc(A);
  pause(.1);
  drawnow();
end