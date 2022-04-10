img1 = imread('homooo\i1.jpg')
img2 = imread('homooo\i2.jpg')
no_points = 5
imshow(img1);
[xs,ys] = ginput(6)
imshow(img2);
[xd,yd] = ginput(6)
A = zeros(2*no_points,9);
for i = 1:no_points,
    A(2*i-1,:) = [xs(i),ys(i),1,0,0,0,-xs(i)*xd(i),-xd(i)*ys(i),-xd(i)];
    A(2*i,:) = [0,0,0,xs(i),ys(i),1,-xs(i)*yd(i),-yd(i)*ys(i),-yd(i)];
end;
if no_points == 4
    h = null(A)
else
    [u,s,v] = svd(A)
    h = v(:,9);
end;
h