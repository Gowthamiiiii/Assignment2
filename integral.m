I = imread('images\Image1.jpg');
imshow(I)
J = integralImage(I);
d = drawrectangle;
r = floor(d.Vertices(:,2)) + 1;
c = floor(d.Vertices(:,1)) + 1;
regionSum = J(r(1),c(1)) - J(r(2),c(2)) + J(r(3),c(3)) - J(r(4),c(4))
regionSum;