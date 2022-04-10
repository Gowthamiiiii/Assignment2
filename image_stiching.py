import cv2
image_set=['i1.jpg','i2.jpg','i3.jpg']
imgs = []
for i in range(len(image_set)):
	imgs.append(cv2.imread(image_set[i]))
	imgs[i]=cv2.resize(imgs[i],(0,0),fx=0.4,fy=0.4)
cv2.imshow('1',imgs[0])
cv2.imshow('2',imgs[1])
cv2.imshow('3',imgs[2])
stitchy=cv2.Stitcher.create()
(stiching_done,output)=stitchy.stitch(imgs)
if stiching_done != cv2.STITCHER_OK:
	print("Try again!!!...")
else:
	print('Yayyy, successfull!!!')
cv2.imshow('final',output)
cv2.waitKey(0)
