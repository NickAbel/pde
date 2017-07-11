function[A] = OneDimHeatEq(I,B0,BL,S,N)

% This function takes as its input a 1xN vector I, representing the 1-D 
% rod at its initial condition. B0 and BL are numbers which represent 
% boundary conditions. S is the constant k*dt/(dx)^2, and N is the total 
% number of time steps. The vector A is returned, representing the 1-D rod 
% at time step N.

A=zeros(length(I),1);
A(1)=B0;
A(length(A))=BL;
for n=1:N
  for J=2:(length(A)-1)
    A(J)=I(J)+S*(I(J+1)+I(J-1)-2*I(J));
  end
  I=A;
end