import os
import sqlite3
import copy
import cv2
import torch
import argparse
import numpy as np
from scipy.ndimage import zoom
from models.resnet import *
import torch.nn.functional as F

'''
---  S T A R T  O F  F U N C T I O N  B A C K P R O P _ K E R N E L _ I N D I C E S  ---

    [About]

        Function for finding the indices of specific kernels in previous layers that are above a certain threshold.
        The function iterates over all chosen kernels k_l and applied them to the spatio-temporal activation maps used
        as input (a_l1). The activation map is then pooled to a single vector shape of size (#channels).
        For that vector, the indices of the most used activations based on the applied kernel k_l can be found (given
        that they are larger than the defined threshold). What is returned, is a dictionary containing the index of each
        kernel k_l as keys and the indices of each corresponding activation a_l1[j] that is larger than the threshold.
        The indices of theses activations directly relate to the convolution operation (with kernel k_l1[j]), so the
        same process can be further applied to the previous layer.

    [Args]

        - indices_l: Dictionary of integers for the kernel indices chosen to be visualised in layer l.
        - k_l: Tensor containing all the kernels of layer (l).
        - a_l1: Tensor containing the input activation maps (a_l1) for layer (l).
        - thres: Float. Determined the threshold value chosen for visualising only specific kernels.
        - topn: Integer only visualise top n kernels with the highest activations.

    [Returns]

        - indices_l1: Dictionary of integers containing the kernel k_l indices as keys with the indices of the found
         activations (>= thres) as values.

'''


def backprop_kernel_indices(indices_l, k_l, a_l1, thres, topn=None):
    #Initialisation
    indices_l1 = {}

    # Special architectures that include either; cross-channel convolutions, seperate branches or fibres.
    if (len(k_l)>1):
        # Get shape(s) of all kenrls in layer
        kernels_shapes = [k.shape[0] for k in k_l]
        # Create corresponding kernel shapes
        tmp=0
        kernels_in_activations = [tmp+k for k in kernels_shapes]
        kernels_in_activations = [0] + kernels_in_activations

    # Start by computing activation map a^k_l(i)_l1 for each k_l(i), where i in indices
    for i in indices_l:

        # Normal convolutions over entire activation maps
        if (len(k_l)==1):
            kernel_indx = 0

        # Convolutions with various channels sizes
        else:
            tmp = 0
            kernel_indx = [id for id,ki in enumerate(kernels_shapes) if (kernels_in_activations[id+1]-i) > 0]
            kernel_indx = kernel_indx[-1]

        # pointwise multiplication for activation a(l1) and the ith kernel in k_l
        #print("Kernel shape: ",k_l[i].shape)
        #print("Activation_shape: ",a_l1[0].shape)

        # Downsample the spatio-temporal dimension to create a global representation of the activation map and kernel.
        _, dk, hk, wk = list(k_l[kernel_indx][i].size())
        _, da, ha, wa = list(a_l1[0].size())
        kernel = F.avg_pool3d(k_l[kernel_indx][i], (dk, hk, wk)).squeeze(-1).squeeze(-1).squeeze(-1)
        act_map = F.avg_pool3d(a_l1[0], (da, ha, wa)).squeeze(-1).squeeze(-1).squeeze(-1)

        # If group convolutions, increase the kernel size
        groups = act_map.shape[0] // kernel.shape[0]

        # Inflate kernels to represent all kernels in the same feature space as the input
        if groups > 1:
            kernel = kernel.repeat(groups)
            # Kernel inflation form [out_channels, in_channels/groups] -> [out_channels, in_channels]
            kernel = torch.cat([k.repeat(groups) for k in kernel],0)

        # Select activations for corresponding kernel channels - special architectures
        if (len(k_l)>1):
            act_map = act_map[kernels_in_activations[kernel_indx]:kernels_in_activations[kernel_indx+1]]



        pooled = torch.mul(kernel, act_map)

        pooled = pooled/pooled.sum(0).expand_as(pooled)

        base = torch.min(pooled)
        pooled_range = torch.max(pooled) - base
        pooled = torch.FloatTensor([(x-base)/pooled_range for x in pooled])

        # Iterate over the pooled volume and find the indices that have a value larger than the threshold
        if topn is None:
            indices_l1_i = [j for j, feat in enumerate(pooled) if feat >= thres]
        else:
            # Get all values above threshold
            values = [value for value in enumerate(pooled) if value >= thres]
            # accending order sort
            values.sort()
            # select n values
            values = values[-topn:]
            # find top n value indices in pooled tensor
            indices_l1_i = [j for j, feat in enumerate(pooled) if feat in values]


        # Append indices to dictionary
        indices_l1[i]=indices_l1_i

    return indices_l1
'''
---  E N D  O F  F U N C T I O N  B A C K P R O P _ K E R N E L _ I N D I C E S  ---
'''






'''
---  S T A R T  O F  F U N C T I O N  G E N E R A T E _ I N D I C E S  ---

    [About]

        Function for tunneling through the network saving the indices of kernels that produce activations larger than
        a threshold in a parend-child manner. For iterating through defined model depth max_depth the function
         takes a recursive form.

    [Args]

        - layers_dict: A dictionary containing nested dictionaries following the overall structure of the network.
            Kernel indices of layers correspond to keys and the values relate to the layer connections to kernels in
            the previous layer in which the activations were larger than a threshold value.
        - kernels: Tensor or list containing all the kernels of each layer in the network.
        - activations: Tensor or list containing all the activation maps of each layer of the network.
        - threshold: Float value determining the connections that should be visualised.
        - index: Integer for keeping track of how far (backwards) the function has backstepped into the network.
        - max_depth: Integer for the maximum depth to backstep to.
        - vis_depth: Integer for the layer in which only selected kerensl are to be visualised.
        - vis_num_kernels: Integer for the top k kernels to be visualised/backstepped in layer vis_depth

    [Returns]

        - layers_dict: A dictionary updated with all the connection paths between {net_depth,...,net_depth-max_depth}
            in pairs of (int,dir) per layer.


'''


def generate_indices(layers_dict, kernels, activations, threshold, index, max_depth, vis_depth, vis_num_kernels):
    print('Backstepping to depth -%d of maximum -%d'%(index,max_depth))

    # Function termination after maximum depth is reached
    if (index>=max_depth):
        print('END OF BRANCH DISCOVERY')
        return layers_dict
    # Iteration
    for key in layers_dict.keys():
        print('layerindex:',index,'key:',key,'keys dict:',layers_dict.keys(),'\n')
        # Unexplored connection
        if not isinstance(layers_dict[key],dict):
            # For layer that only specific kernels are to be visualised
            if (index == vis_depth):
                layers_dict[key] = backprop_kernel_indices(layers_dict[key], kernels[-index], activations[-index-1], threshold, vis_num_kernels)
            else:
                layers_dict[key] = backprop_kernel_indices(layers_dict[key], kernels[-index], activations[-index-1], threshold)

        if (index>max_depth):
            return layers_dict
        # Recursive step
        if isinstance(layers_dict[key],dict):
            print('Backstepping...')
            layers_dict[key] = generate_indices(layers_dict[key], kernels, activations, threshold, index+1, max_depth, vis_depth, vis_num_kernels)

    return layers_dict
'''
---  E N D  O F  F U N C T I O N  G E N E R A T E _ I N D I C E S  ---
'''







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
def layer_visualisations(args, layers_dict, kernels, activations, index, RGB_video, tubes_dict = {}):
    # Main Iteration
    for key,value in layers_dict.items():

        # Recursive step
        if isinstance(value,dict):
            layers_dict[key],tubes_dict = layer_visualisations(args, value, kernels, activations, index+1, RGB_vid, tubes_dict)

        if isinstance(layers_dict[key],list):

            # get output activation map for layer
            layerout = torch.tensor(activations[-index-1])

            cam = torch.zeros([activations[-index-1].shape[0],1,activations[-index-1].shape[2],activations[-index-1].shape[3],activations[-index-1].shape[4]], dtype = torch.float32).cuda()
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
            heatmap_dir = os.path.join(args.base_output_dir, str('layer_%d_kernel_%d_num_kernels_%d'%(index,key,len(layers_dict[key]))), "heat_tubes")

            # produce heatmap for every frame and activation map
            tubes = []
            for frame_num in range(cam.shape[0]):
                #Create colourmap
                heatmap = cv2.applyColorMap(np.uint8(255*cam[frame_num]), cv2.COLORMAP_JET)

                # Create frame with heatmap
                heatframe = heatmap//2 + RGB_vid[0][frame_num]//2
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
def savetopng(tubes,path):
    print('SAVING TUBES FOR :',path)
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

        rows,cols,ch = rgba.shape

        # Transforms
        pts1 = np.float32([[cols/15,rows/11],[cols/1.8,rows/10],[cols/8.5,rows/1.6]])
        pts2 = np.float32([[cols/5,rows/4],[cols/1.7,rows/4.7],[cols/5.5,rows/2.1]])
        M = cv2.getAffineTransform(pts1,pts2)
        dst1 = cv2.warpAffine(rgba,M,(cols,rows), borderValue=(255,255,255,0))

        # add to list of frames
        transformed_frames_filenames.append(os.path.join(path,'frames','frame_00%d.png'%(frame)))
        transformed_frames_list.append(dst1)

    if not os.path.exists(os.path.join(path,'frames')):
        os.makedirs(os.path.join(path,'frames'))

    # save frames to corresponding directory
    for i in range(0,len(transformed_frames_list)):
        cv2.imwrite(transformed_frames_filenames[i],transformed_frames_list[i])


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

            if (i==0):
                image = tmp
                continue
            new_h = image.shape[0] + shift[0]
            new_w = image.shape[1] + shift[1]
            new_image = np.full((new_h, new_w, 4),(255,255,255,0))

            new_image[shift[0]:image.shape[0]+shift[0],shift[1]:image.shape[1]+shift[1]] = image


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
                    if(tmp[ih,iw,3]>0):
                        if (new_image[ih,iw,3] == 0):
                            new_image[ih,iw] = tmp[ih,iw]
                        else:
                            alpha = new_image[ih,iw,3] + tmp[ih,iw,3]
                            new_image[ih,iw] = new_image[ih,iw]*(new_image[ih,iw,3]/alpha) + tmp[ih,iw]*(tmp[ih,iw,3]/alpha)

            image = copy.deepcopy(new_image)
        cv2.imwrite(os.path.join(path,'saliency_tubes','result%d.png'%(len(transformed_frames_list)-j)),image)
'''
---  E N D  O F  F U N C T I O N  S A V E T O P N G  ---
'''


def load_images2(**kwargs):
    load_from = r"D:\Datasets\egocentric\GTEA\cropped_clipsframes\P02-R03-BaconAndEggs\P02-R03-BaconAndEggs-1378142-1381629-F033067-F033168"
    revert_lr = False
    revert_ud = False
    sampler = kwargs.get('sampler')

    if load_from == 'sample_line':
        sample = get_sample_line()
        orig_imgs, final_imgs, uid, idxs = _load_images_from_sample(sample, revert_lr, revert_ud, sampler)
    elif load_from == 'video_path':
        video_path = get_video_path()
        orig_imgs, final_imgs, uid, idxs = _load_images_from_clip(video_path, revert_lr, revert_ud, sampler)
    else:
        raise NotImplementedError('Unknown image input type')

    return orig_imgs, _make_torch_images(final_imgs), uid, idxs


def center_crop(data, tw=256, th=256):
    h, w, c = data.shape
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
    return cropped_data

def load_images(frame_dir, selected_frames):
    images = np.zeros((16, 224, 224, 3))
    orig_imgs = np.zeros_like(images)

    # Establish connection to .db

    con = sqlite3.connect(os.path.join(frame_dir, 'frames.db'))
    cur = con.cursor()
    if cur is None:
        print("cursor is empty")
        raise Exception

    paths = []
    dir = frame_dir.split('/')[-1]
    dir_parts = os.path.split(frame_dir)
    # Get framespaths to load from database
    for index in selected_frames:
        # paths.append(os.path.join(str(dir),'frame_%05d'%index))
        paths.append("{}/{}".format(dir_parts[-1],'frame_%05d'%index))

    # for each element in database
    for i, frame_name in enumerate(paths):
        row = cur.execute('SELECT Objid, frames FROM Images WHERE ObjId=?', (frame_name,))
        for ObjId, item in row:
            #--- Decode blob
            nparr  = np.fromstring(item, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        cropped_img = center_crop(img_np)
        scaled_img = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LINEAR)
        final_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
        images[i] = final_img
        orig_imgs[i] = scaled_img

    cur.close()
    con.close()

    torch_imgs = torch.from_numpy(images.transpose(3,0,1,2))
    torch_imgs = torch_imgs.float() / 255.0
    mean_3d = [124 / 255, 117 / 255, 104 / 255]
    std_3d = [0.229, 0.224, 0.225]
    for t, m, s in zip(torch_imgs, mean_3d, std_3d):
        t.sub_(m).div_(s)
    return np.expand_dims(orig_imgs, 0), torch_imgs.unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description='mfnet-base-parser')
    parser.add_argument("--num_classes", type=int) # 400
    parser.add_argument("--model_weights", type=str)
    parser.add_argument("--frame_dir", type=str)
    parser.add_argument("--frames_start", type=int) # end - start = 16 frames
    parser.add_argument("--frames_end", type=int) # duration = 16
    parser.add_argument("--label", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--backprop_depth", type=int, default=3)
    parser.add_argument("--tubes_vis_method", type=str, default='concat')
    parser.add_argument("--visualisation_method", type=str, default='all')
    parser.add_argument("--base_output_dir", type=str, default=r"visualisations")
    return parser.parse_args()

args = parse_args()
# Frame duration definition
duration = int(args.frames_end - args.frames_start)
# Create a list of frames based on start and end time
selected_frames = [i for i in range(args.frames_start,args.frames_end)]
# Load clip
RGB_vid, vid = load_images(args.frame_dir, selected_frames)

# load network structure
model_ft = resnet152(sample_size=224,sample_duration=duration,num_classes=args.num_classes)
# Create parallel model for multi-gpus
model_ft = torch.nn.DataParallel(model_ft).cuda()
# Load checkpoint
checkpoint = torch.load(args.model_weights, map_location={'cuda:1':'cuda:0'})
model_ft.load_state_dict(checkpoint['state_dict'])
# Set to evaluation mode
model_ft.eval()
print('\n MODEL LOADED SUCESSFULLY... \n')

# get class prediction, all regularisation predictions and last convolution layer activation map
with torch.no_grad():
    predictions, kernels, activations = model_ft(torch.tensor(vid).cuda())

print('\n PREDICTIONS CALCULATED... \n')

# Get Linear layer weights
class_weights = kernels[-1]
class_weights = class_weights[0].detach().cpu().numpy().transpose()

# Minmax normalisation
base = class_weights.min()
weights_range = class_weights.max() - base
class_weights = np.asarray([(x-base)/weights_range for x in class_weights])

# Get class weights that are larger than threshold
kernel_indeces = [index for index,weight in enumerate(class_weights[:,args.label]) if (weight >= args.threshold)]

layers_weights_dict = {}
layers_weights_dict[args.label] = kernel_indeces

# Define kernels to be visualised
vis = args.visualisation_method
if (vis == 'all'):
    layer_num = kernel_num = None
elif('top' in vis):
    kernel_num = int(vis.split('top')[-1].split('_')[0])
    layer_num = int(vis.split('in')[-1])
else:
    print('Non regular visualisation format - visualizing all filters')
    layer_num = kernel_num = None


for i,k in enumerate(kernels[:-1]):
    print()
    print('Kernels shapes',[ki.shape for ki in k])
    print('Activations shape',activations[i].shape)


# Create dictionary for layer indices
k_indices_dict = generate_indices(layers_dict=layers_weights_dict, kernels=kernels[:-1], activations=activations[:-1], threshold=args.threshold, index=1, max_depth=args.backprop_depth, vis_depth=layer_num,vis_num_kernels=kernel_num)

print(k_indices_dict)

# Save to JSON file
import json

with open('class_dependency_graph.json', 'w') as fp:
    json.dump(k_indices_dict, fp)

# Call to get all saliency tubes and store them to a dictionary
_, tubes_dict = layer_visualisations(args=args, layers_dict=layers_weights_dict, kernels=kernels[:-1], activations=activations[:-1], index=1, RGB_video=RGB_vid)

print([*tubes_dict])

import time
start = time.time()
for filename,tube in tubes_dict.items():
    savetopng(tube,filename)
end = time.time()
print(end-start)
