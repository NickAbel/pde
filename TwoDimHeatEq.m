function[A] = TwoDimHeatEq(s,A)
n=size(A);
for (i=2:n(1)-1)
    for (j=2:n(2)-1)
        A(i,j)=s*[A(i+1,j)+A(i-1,j)+A(i,j+1)+A(i,j-1)];
    end
end
