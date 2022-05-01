img1 = imread('homooo\i1.jpg')
img2 = imread('homooo\i2.jpg')
X = img1 - img2;
ssd = sum(X(:).^2);
ssd