function [un] = OneDimWaveEq(f,g,c,dt,dx,n)

ui=g;
s=c^2/(dx/dt)^2;
uo=f;
for (j=2:length(ui)-1)
  ui(j)=-dt*g(j)+s*[f(j-1)-2*f(j)+f(j+1)]+f(j);
end

un=zeros(length(ui),1);

for (j=2:length(un)-1)
  un(j)=2*dt*g(j)+ui(j);
end

for (i=1:n)
  ui=uo;
  uo=un;
  for (j=2:length(un)-1)
    un(j)=2*uo(j)-ui(j)+s*[uo(j-1)-2*uo(j)+uo(j+1)];
  end
end