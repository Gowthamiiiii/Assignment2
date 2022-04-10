I = rgb2gray(imread("gowthamipic.png"));
corners = detectHarrisFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(30));