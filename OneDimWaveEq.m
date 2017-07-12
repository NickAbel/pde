function [ump1] = OneDimWaveEq(f,g,c,T,dx)

% Approximates the wave equation in one dimension on a uniform string.
% f,g: Two vectors epreseting initial conditions are needed, for example
% if f(x)=u(x,0), g(x)=du/dt(x,0) are specified, we can march forward using
% this method.
% c: The ratio, tension / mass-density. In a uniform string, this is
% constant.
% T,dx: Total time steps, change in space respectively. Here, the Courant
% stability condition applies, so we ensure that c^2/(dx/dt)^2 <= 1 holds.

N=length(f);
t=0;
dt=dx/c;
umm1=g;
cfl=c*dt/dx;
s=c^2/(dx/dt)^2;
ump1=f;
um=f;

while (t<T)  
  % Uncomment lines below for absorbing boundary conditions
  % ump1(1)=um(2)+((cfl-1)/(cfl+1))*(ump1(2)-um(1));
  % ump1(N)=um(N-1)+((cfl-1)/(cfl+1))*(ump1(N-1)-um(N));
  
  t+=dt;
  umm1=um;
  um=ump1;
  
  for j=2:length(ump1)-1
    ump1(j)=2*um(j)-umm1(j)+s*(um(j-1)-2*um(j)+um(j+1));
  end
  
  % Uncomment, edit line below to include source on string
  % um(length(um)/2,1)=sin(t);
  
  plot(um);
  pause(.4);
  drawnow();
end