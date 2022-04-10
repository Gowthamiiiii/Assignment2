I = rgb2gray(imread("gowthamipic.png"));
title("Gray Scale Image");
N = edge(I, 'Canny');
subplot(2, 4, 7),
imshow(N);
N;
type(N)
title("Canny");