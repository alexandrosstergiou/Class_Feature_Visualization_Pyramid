import os
import copy
import torch
import numpy as np
import cv2
from scipy.ndimage import zoom


def define_vis_kernels(vis):
    if vis == 'all':
        layer_num = kernel_num = None
    elif 'top' in vis:
        kernel_num = int(vis.split('top')[-1].split('_')[0])
        layer_num = int(vis.split('in')[-1])
    else:
        print('Non regular visualisation format - visualizing all filters')
        layer_num = kernel_num = None
    return layer_num, kernel_num

'''
---  S T A R T  O F  F U N C T I O N  L A Y E R _ V I S U A L I S A T I O N S  ---

    [About]

        Main function used for computing class activations per layer. The activations that are above a certain threshold value for the video that is being studied are tracked throughout the network. Selected activations are concatenated to a single value given the kernels in which their pooled activations are sufficiently influential. Class activations are then applied on top of the video volume.

    [Args]

        - args: Parser object.
        - layers_dict: A dictionary containing nested dictionaries following the overall structure of the network. Kernel indices of layers correspond to keys and the values relate to the layer connections to kernels in the previous layer in which the activations were larger than a threshold value.
        - kernels: Tensor or list containing all the kernels of each layer in the network.
        - activations: Tensor or list containing all the activation maps of each layer of the network.
        - index: Integer for keeping track of how far (backwards) the function has backstepped into the network.
        - RGB_video: Array or List of frames.
        - tubes_dict: Dictionary of tuples containing the saliency tubes and the kernels that were connected to from the previous layer.

    [Returns]

        - [*layers_dict] : List containing all the keys of the current layer to be used by the parent call to the recursive function.
        - tubes_dict : Dictionary containing the computed saliency tubes along side the corresponding information as dictionary keys.


'''
def layer_visualisations(base_output_dir, layers_dict, kernels, activations, index, RGB_video, tubes_dict = {}):
    # Main Iteration
    for key,value in layers_dict.items():

        # Recursive step
        if isinstance(value,dict):
            layers_dict[key],tubes_dict = layer_visualisations(base_output_dir, value, kernels, activations, index+1,
                                                               RGB_video, tubes_dict)

        if isinstance(layers_dict[key],list):

            # get output activation map for layer
            layerout = torch.tensor(activations[-index-1])

            cam = torch.zeros([activations[-index-1].shape[0],
                               1,  # activations[-index-1].shape[1]
                               activations[-index-1].shape[2],
                               activations[-index-1].shape[3],
                               activations[-index-1].shape[4]], dtype=torch.float32).cuda()
            # main loop for selected kernels

            print('Creating Saliency Tubes for :',str('layer %d, kernel %d, w/ %d <child> kernels'%(index,key,len(layers_dict[key]))))

            # Apply padding - only to cases that there is a size mis-match

            for i in layers_dict[key]:
                try:
                    cam += layerout[0,i].unsqueeze(0)
                except Exception:
                    print('-- PREDICTIONS LAYER REACHED ---')

            # Resize CAM to frame level (batch,channels,frames,heigh,width) --> (frames, height, width)
            cam = cam.squeeze(0).squeeze(0)
            t, h, w = cam.shape
            _, clip_len, clip_height, clip_width, _ = RGB_video.shape

            # Tranfer both volumes to the CPU and convert them to numpy arrays
            cam = cam.cpu().numpy()
            cam = zoom(cam, (clip_len//t, clip_height//h, clip_width//w))

            # normalise
            cam -= np.min(cam)
            cam /= np.max(cam) - np.min(cam)

            # make dirs and filenames
            heatmap_dir = os.path.join(base_output_dir,
                                       str('layer_%d_kernel_%d_num_kernels_%d'%(index,key,len(layers_dict[key]))),
                                       "heat_tubes")

            # produce heatmap for every frame and activation map
            tubes = []
            for frame_num in range(cam.shape[0]):
                #Create colourmap
                heatmap = cv2.applyColorMap(np.uint8(255*cam[frame_num]), cv2.COLORMAP_JET)

                # Create frame with heatmap
                heatframe = heatmap//2 + RGB_video[0][frame_num]//2
                tubes.append(heatframe)

            # Append a tuple of the computed heatmap and the kernels used
            tubes_dict[heatmap_dir]= (tubes,layers_dict[key])

    print('End OF SALIENCY TUBE GENERATION IN DEPTH %d WITH KERNELS '%(index),[*layers_dict])
    return [*layers_dict],tubes_dict
'''
---  E N D  O F  F U N C T I O N  L A Y E R _ V I S U A L I S A T I O N S  ---
'''


'''
---  S T A R T  O F  F U N C T I O N  S A V E T O P N G  ---

    [About]

        Function for saving all computed saliency tubes in a stack-like format. Each stack consists of the video frames with their activation visualisations. An alpha value is applied to every but the frame of the specific iteration. This is to allow a better and clearer animated image.

    [Args]

        - tubes: Tuple holding the 4D saliency tubes created in an array type format and the corresponding kernels of the previous layer that were found to be influencial to this kernels given a threshold value.
        - path: String for the main filepath to save the data.

    [Returns]

        - None


'''
def savetopng(tubes, path):
    print('SAVING TUBES FOR :', path)
    # Save kernel indices that are visualised into file
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))
    file = open(os.path.join(path,'frames.txt'),'w')
    file.write(str(tubes[1]))
    file.close()

    transformed_frames_list = []
    transformed_frames_filenames = []

    # Iterate over tubes and apply visualisation transforms
    for frame,image in enumerate(tubes[0]):
        # Ensuring that only unsign integers will be used (as expected)
        image = image.astype(np.uint8)
        # Resizing
        image = cv2.resize(image, (256, 256))
        # Create 4 channel array (used for better frame overlaping)
        rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = 255

        rows, cols, ch = rgba.shape

        # Transforms
        pts1 = np.float32([[cols/15, rows/11], [cols/1.8, rows/10], [cols/8.5, rows/1.6]])
        pts2 = np.float32([[cols/5, rows/4], [cols/1.7, rows/4.7], [cols/5.5, rows/2.1]])
        M = cv2.getAffineTransform(pts1, pts2)
        dst1 = cv2.warpAffine(rgba, M, (cols, rows), borderValue=(255, 255, 255, 0))

        # add to list of frames
        transformed_frames_filenames.append(os.path.join(path, 'frames', 'frame_00%d.png' % frame))
        transformed_frames_list.append(dst1)

    if not os.path.exists(os.path.join(path, 'frames')):
        os.makedirs(os.path.join(path, 'frames'))

    # save frames to corresponding directory
    for i in range(0, len(transformed_frames_list)):
        cv2.imwrite(transformed_frames_filenames[i], transformed_frames_list[i])


    # img2 is to be shifted by `shift` amount
    shift = (80, 0)

    # Saliency tube directory
    if not os.path.exists(os.path.join(path,'saliency_tubes')):
        os.makedirs(os.path.join(path,'saliency_tubes'))

    # Iterate over each frame(this will be the main frame to be visualised)
    for j in (range(len(transformed_frames_list))):
        for i, img in enumerate(reversed(transformed_frames_list)):

            # use ~.45 alpha for frames that are no main frame 'j'
            tmp = np.full((img.shape[0], img.shape[1], 4), (255, 255, 255, 0))
            if i != j:
                # for ih in range(img.shape[0]):
                #     for iw in range(img.shape[1]):
                #         if(img[ih,iw,3]>0):
                #             tmp[ih,iw] = np.array([img[ih,iw][0],img[ih,iw][1],img[ih,iw][2],135])

                tmp[:, :, 0] = np.where(img[:, :, 3] > 0, img[:, :, 0], tmp[:, :, 0])
                tmp[:, :, 1] = np.where(img[:, :, 3] > 0, img[:, :, 1], tmp[:, :, 1])
                tmp[:, :, 2] = np.where(img[:, :, 3] > 0, img[:, :, 2], tmp[:, :, 2])
                tmp[:, :, 3] = np.where(img[:, :, 3] > 0, 135, 0)
            else:
                tmp = copy.deepcopy(img)

            if i == 0:
                image = tmp
                continue
            new_h = image.shape[0] + shift[0]
            new_w = image.shape[1] + shift[1]
            new_image = np.full((new_h, new_w, 4), (255, 255, 255, 0))

            new_image[shift[0]:image.shape[0]+shift[0], shift[1]:image.shape[1]+shift[1]] = image


            # alpha = new_image[:256,:256,3] + tmp[:,:,3]
            # for iii in range(4):
            #     new_image[:256,:256,iii] = np.where(tmp[:,:,3]>0,
            #                                 np.where(new_image[:256,:256,3]==0,
            #                                          tmp[:,:,iii],
            #                                          new_image[:256,:256,iii]*(new_image[:256,:256,3]/alpha) +
            #                                          tmp[:,:,iii]*(tmp[:,:,3]/alpha)
            #                                          ),
            #                                 new_image[:256,:256,iii])

            # Only transfer pixels that are not transparent (i.e. part of the frame rather than the image)
            for ih in range(tmp.shape[0]):
                for iw in range(tmp.shape[1]):
                    if tmp[ih, iw, 3] > 0:
                        if new_image[ih, iw, 3] == 0:
                            new_image[ih, iw] = tmp[ih, iw]
                        else:
                            alpha = new_image[ih, iw, 3] + tmp[ih, iw, 3]
                            new_image[ih, iw] = new_image[ih, iw]*(new_image[ih, iw, 3]/alpha) + \
                                tmp[ih, iw]*(tmp[ih, iw, 3]/alpha)

            image = copy.deepcopy(new_image)
        cv2.imwrite(os.path.join(path, 'saliency_tubes', 'result%d.png'%(len(transformed_frames_list)-j)), image)
'''
---  E N D  O F  F U N C T I O N  S A V E T O P N G  ---
'''


def kernel_heatmaps(layers_dict, kernels, activations, index, out_size, cams_list=list()):
    # Main Iteration
    for key, value in layers_dict.items():

        # Recursive step
        if isinstance(value, dict):
            layers_dict[key], cams_list = kernel_heatmaps(value, kernels, activations, index + 1, out_size, cams_list)

        if isinstance(layers_dict[key], list):

            # get output activation map for layer
            layerout = torch.tensor(activations[-index - 1])

            cam_shape = [activations[-index - 1].shape[0], 1, activations[-index - 1].shape[2],
                         activations[-index - 1].shape[3], activations[-index - 1].shape[4]]
            cam = torch.zeros(cam_shape, dtype=torch.float32).cuda()
            # main loop for selected kernels

            print('Creating Saliency Tubes for :',
                  str('layer %d, kernel %d, w/ %d <child> kernels' % (index, key, len(layers_dict[key]))))

            # Apply padding - only to cases that there is a size mis-match
            for i in layers_dict[key]:
                try:
                    cam += layerout[0, i].unsqueeze(0)
                except Exception:
                    print('-- PREDICTIONS LAYER REACHED ---')

            # Resize CAM to frame level (batch,channels,frames,heigh,width) --> (frames, height, width)
            cam = cam.squeeze(0).squeeze(0)
            t, h, w = cam.shape
            clip_len, clip_height, clip_width = out_size

            # Tranfer both volumes to the CPU and convert them to numpy arrays
            cam = cam.detach().cpu().numpy()

            cam = resize_cam(cam, x=clip_len//t, y=clip_height//h, z=clip_width//w)

            # cams_dict["{}_{}".format(index, key)] = cam
            cams_list.append(cam)

    print('End OF CAM GENERATION IN DEPTH %d WITH KERNELS ' % (index), [*layers_dict])
    return [*layers_dict], cams_list


def resize_cam(cam, x=2, y=32, z=32):
    # Resize CAM to frame level
    cam = zoom(cam, (x, y, z))
    # print("Min - max (pre standardization)")
    # for i in range(len(cam)):
    #     print("{}: {}, {}".format(i, np.min(cam[i]), np.max(cam[i])))
    cam -= np.min(cam)
    cam /= np.max(cam) - np.min(cam)

    return cam


def vis_cams_overlayed_per_branch(base_output_dir, cams, RGB_video):
    num_cams = len(cams)
    # one colormap per cam. At most 5 cams for visualization
    cam_colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_AUTUMN, cv2.COLORMAP_HOT, cv2.COLORMAP_COOL, cv2.COLORMAP_CIVIDIS]

    # make dirs and filenames
    cam_dir = os.path.join(base_output_dir,
                               str('overlay_cam_branch_{}'.format(num_cams)),
                               "heat_tubes")

    # first create the colormapped cams
    if num_cams > len(cam_colormaps): # colormaps are not enough to visualize all cams, just keep the 5 first cams
        cams = cams[:len(cam_colormaps)]

    cmap_cams = list()
    for i, cam in enumerate(cams):
        frames_cmap_cams = list()
        for j in range(cam.shape[0]):
            frame_cam = cam[j]
            heatmap = cv2.applyColorMap(np.uint8(255 * frame_cam), cam_colormaps[i]) / 255
            frames_cmap_cams.append(heatmap)
        cmap_cams.append(frames_cmap_cams)

    # aggregate the cams to the original frames

    for frame_num in range(RGB_video.shape[1]):  # usually 16
        heatframe = RGB_video[0][frame_num] // 3
        cam_heatframe = cmap_cams[0][frame_num]
        for frames_cmap_cams in cmap_cams[1:]:
             cam_heatframe += (255*frames_cmap_cams[frame_num]) // num_cams
        heatframe += 2*cam_heatframe//3
        cv2.imshow('frames_cam', heatframe.astype(np.uint8))
        cv2.waitKey(0)

    #
    # for frame_num in range(cams[0].shape[0]):
    #     # produce heatmap for every frame and activation map
    #     for i in range(len(cam_colormaps)):
    #         cam = cams[i]
    #         # Create colourmap
    #         heatmap = cv2.applyColorMap(np.uint8(255//5 * cam[frame_num]), cam_colormaps[i])
    #
    #         # Create frame with heatmap
    #         heatframe = heatmap // 2 + RGB_video[0][frame_num] // 3 # all_cams * 2/3 + original video * 1/3
    #     tubes.append(heatframe)
    #
    # # Append a tuple of the computed heatmap and the kernels used
    # tubes_dict[heatmap_dir] = (tubes, layers_dict[key])


def savetopng_norotation(tubes, path):
    path += "_orig"
    print('SAVING TUBES FOR :', path)
    # Save kernel indices that are visualised into file
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))
    file = open(os.path.join(path,'frames.txt'),'w')
    file.write(str(tubes[1]))
    file.close()

    if not os.path.exists(os.path.join(path, 'frames')):
        os.makedirs(os.path.join(path, 'frames'))

    # Iterate over tubes and just save without transforms
    for frame,image in enumerate(tubes[0]):
        # Ensuring that only unsign integers will be used (as expected)
        image = image.astype(np.uint8)
        cv2.imwrite(os.path.join(path, 'frames', 'frame_00%d.png' % frame), image)
