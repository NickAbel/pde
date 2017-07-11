function [un] = OneDimWaveEq(f,g,c,dt,dx,n)

% Approximates the wave equation in one dimension on a uniform string.
% f,g: Two vectors epreseting initial conditions are needed, for example
% if f(x)=u(x,0), g(x)=du/dt(x,0) are specified, we can march forward using
% this method.
% c: The ratio, tension over mass density. In a uniform string, this is
% constant.
% dt,dx: Change in time, change in space respectively. Here, the Courant
% stability condition applies, so ensure that c^2/(dx/dt)^2 <= 1 holds.
% n: Number of steps to iterate.

ui=g;
s=c^2/(dx/dt)^2;
uo=f;
for j=2:length(ui)-1
  ui(j)=-dt*g(j)+s*(f(j-1)-2*f(j)+f(j+1))+f(j);
end

un=zeros(length(ui),1);

for j=2:length(un)-1
  un(j)=2*dt*g(j)+ui(j);
end

for i=1:n
  ui=uo;
  uo=un;
  for j=2:length(un)-1
    un(j)=2*uo(j)-ui(j)+s*(uo(j-1)-2*uo(j)+uo(j+1));
  end
end