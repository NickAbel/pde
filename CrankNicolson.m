function[b] = CrankNicolson(b,s,m)

% b: u at initial time (column vector 1,...,n-1)
% m: number of steps

n = length(b)+1;
r = 2+s;
A = zeros(n-1);
for (i = 1:(n-1))
  A(i,i) = r;
  if (i > 1 && i < (n-1))
    A(i,i+1)=A(i,i-1)=-1;
  elseif (i==1) 
    A(i,i+1)=-1;
  elseif (i==(n-1)) 
    A(i,i-1)=-1;
  end
end

for (i=1:m)
  b=s*(A\b);
  b(1)=b(n-1)=0;
end