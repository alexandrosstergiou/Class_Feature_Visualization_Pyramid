import cv2
from PIL import Image
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import copy
import os


def savetopng(tubes,path):

    # Iterate over tubes and apply visualisation transforms
    for frame in tubes:
        # Resizing
        image = cv2.resize(image, (256, 256))
        # Create 4 channel array (used for better frame overlaping)
        rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = 255

        rows,cols,ch = rgba.shape

        # Transforms
        pts1 = np.float32([[cols/15,rows/11],[cols/1.8,rows/10],[cols/8.5,rows/1.6]])
        pts2 = np.float32([[cols/5,rows/4],[cols/1.7,rows/4.7],[cols/5.5,rows/2.1]])
        M = cv2.getAffineTransform(pts1,pts2)
        dst1 = cv2.warpAffine(rgba,M,(cols,rows), borderValue=(255,255,255,0))

        filename = frame.split('/')[-1]

        # add to list of frames
        transformed_frames_filenames.append(filename)
        transformed_frames_list.append(dst1)

    if not os.path.exists(os.path.join(path,'frames')):
        os.makedirs(os.path.join(path,'frames'))

    # save frames to corresponding directory
    for i in range(0,len(transformed_frames_list)):

        cv2.imwrite(transformed_frames_filenames[i],os.path.join(path,'frames',transformed_frames_list[i]))


    # img2 is to be shifted by `shift` amount
    shift = (80, 0)

    # Saliency tube directory
    if not os.path.exists(os.path.join(path,'saliency_tubes')):
        os.makedirs(os.path.join(path,'saliency_tubes'))

    # Iterate over each frame(this will be the main frame to be visualised)
    for j in (range(len(transformed_frames_list))):
        for i,img in enumerate(reversed(transformed_frames_list)):

            #use ~.45 alpha for frames that are no main frame 'j'
            tmp = np.full((img.shape[0], img.shape[1], 4),(255,255,255,0))
            if (i != j):
                for ih in range(img.shape[0]):
                    for iw in range(img.shape[1]):
                        if(img[ih,iw,3]>0):
                            tmp[ih,iw] = np.array([img[ih,iw][0],img[ih,iw][1],img[ih,iw][2],115])
            else:
                tmp = copy.deepcopy(img)

            if (i==0):
                image = tmp
                continue
            new_h = image.shape[0] + shift[0]
            new_w = image.shape[1] + shift[1]
            new_image = np.full((new_h, new_w, 4),(255,255,255,0))

            new_image[shift[0]:image.shape[0]+shift[0],shift[1]:image.shape[1]+shift[1]] = image

            # Only transfer pixels that are not transparent (i.e. part of the frame rather than the image)
            for ih in range(tmp.shape[0]):
                for iw in range(tmp.shape[1]):
                    if(tmp[ih,iw,3]>0):
                        if (new_image[ih,iw,3] == 0):
                            new_image[ih,iw] = tmp[ih,iw]
                        else:
                            alpha = new_image[ih,iw,3] + tmp[ih,iw,3]
                            new_image[ih,iw] = new_image[ih,iw]*(new_image[ih,iw,3]/alpha) + tmp[ih,iw]*(tmp[ih,iw,3]/alpha)

            image = copy.deepcopy(new_image)
        cv2.imwrite(os.path.join(path,'saliency_tubes','result%d.png'%(len(transformed_frames_list)-j)),image)
