function[b] = CrankNicolson(b,s,m)

% The one-dimensional heat equation, approximated using the Crank-Nicolson
% finite-difference scheme.
% b: u at initial time (column vector 1,...,n-1)
% s: The product k*dt/(2*(dx)^2). Crank-Nicolson remains stable for any dt,
% hence s can be as large as it is appropriate.
% m: Number of time-steps to perform.

n = length(b)+1;
r = 2+s;
A = zeros(n-1);
for i = 1:(n-1)
  A(i,i) = r;
  if (i > 1 && i < (n-1))
    A(i,i+1)=-1;
    A(i,i-1)=-1;
  elseif (i==1) 
    A(i,i+1)=-1;
  elseif (i==(n-1)) 
    A(i,i-1)=-1;
  end
end

for i=1:m
  b=s*(A\b);
  b(1)=0;
  b(n-1)=0;
end