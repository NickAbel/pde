function[v] = OneDimHeatEq(u,T,dx,k)

% 1-D heat equation using forward time difference, centered space difference.
% Stability analysis reveals the scheme is stable if s=k*dt/(dx)^2<=1/2.

t=0;
dt=(dx)^2/2*k;
s=k*dt/(dx)^2;
v=zeros(length(u),1);

while (t<T)
  t=dt+t;
  for j=2:(length(u)-1)
    v(j)=u(j)+s*(u(j+1)+u(j-1)-2*u(j));
  end
  u=v;
  plot(v);
  pause(.2);
  drawnow();
end