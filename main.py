import os
import sqlite3
import cv2
import torch
import argparse
import numpy as np
from scipy.ndimage import zoom
from resnet import resnet152
import torch.nn.functional as F
from sklearn.preprocessing import normalize


from saliency_tube_visualisation import savetopng





'''
---  S T A R T  O F  F U N C T I O N  B A C K P R O P _ K E R N E L _ I N D I C E S  ---

    [About]

        Function for finding the indices of specific kernels in previous layers that are above a certain threshold. The function iterates over all chosen kernels k_l and applied them to the spatio-temporal activation maps used as input (a_l1). The activation map is then pooled to a single vector shape of size (#channels). For that vector, the indices of the most used activations based on the applied kernel k_l can be found (given that they are larger than the defined threshold). What is returned, is a dictionary containing the index of each kernel k_l as keys and the indices of each corresponding activation a_l1[j] that is larger than the threshold. The indices of theses activations directly relate to the convolution operation (with kernel k_l1[j]), so the same process can be further applied to the previous layer.

    [Args]

        - indices_l: Dictionary of integers for the kernel indices chosen to be visualised in layer l.
        - k_l: Tensor containing all the kernels of layer (l).
        - a_l1: Tensor containing the input activation maps (a_l1) for layer (l).
        - thres: Float. Determined the threshold value chosen for visualising only specific kernels.
        - topn: Integer only visualise top n kernels with the highest activations.

    [Returns]

        - indices_l1: Dictionary of integers containing the kernel k_l indices as keys with the indices of the found activations (>= thres) as values.

'''
def backprop_kernel_indices(indices_l, k_l, a_l1, thres, topn=None):
    #Initialisation
    indices_l1 = {}

    # Start by computing activation map a^k_l(i)_l1 for each k_l(i), where i in indices
    for i in indices_l:

        # pointwise multiplication for activation a(l1) and the ith kernel in k_l
        #print('Kernel shape: ',k_l[i].shape)
        #print('Activation_shape: ',a_l1[0].shape)

        # Downsample the spatio-temporal dimension to create a global representation of the activation map and kernel.
        _, dk, hk, wk = list(k_l[i].size())
        _, da, ha, wa = list(a_l1[0].size())
        kernel = F.avg_pool3d(k_l[i], (dk,hk,wk)).squeeze(-1).squeeze(-1).squeeze(-1)
        act_map = F.avg_pool3d(a_l1[0], (da,ha,wa)).squeeze(-1).squeeze(-1).squeeze(-1)

        pooled =  torch.mul(kernel, act_map)

        pooled = pooled/pooled.sum(0).expand_as(pooled)

        base = torch.min(pooled)
        pooled_range = torch.max(pooled) - base
        pooled = torch.FloatTensor([(x-base)/pooled_range for x in pooled])

        # Iterate over the pooled volume and find the indices that have a value larger than the threshold
        if (topn is None):
            indices_l1_i = [j for j,feat in enumerate(pooled) if (feat>=thres)]
        else:
            # Get all values above threshold
            values = [value  for value in enumerate(pooled) if (value>=thres)]
            # accending order sort
            values.sort()
            # select n values
            values = values[-topn:]
            # find top n value indices in pooled tensor
            indices_l1_i = [j for j,feat in enumerate(pooled) if (feat in values)]


        # Append indices to dictionary
        indices_l1[i]=indices_l1_i

    return indices_l1
'''
---  E N D  O F  F U N C T I O N  B A C K P R O P _ K E R N E L _ I N D I C E S  ---
'''






'''
---  S T A R T  O F  F U N C T I O N  G E N E R A T E _ I N D I C E S  ---

    [About]

        Function for tunneling through the network saving the indices of kernels that produce activations larger than a threshold in a parend-child manner. For iterating through defined model depth max_depth the function takes a recursive form.

    [Args]

        - layers_dict: A dictionary containing nested dictionaries following the overall structure of the network. Kernel indices of layers correspond to keys and the values relate to the layer connections to kernels in the previous layer in which the activations were larger than a threshold value.
        - kernels: Tensor or list containing all the kernels of each layer in the network.
        - activations: Tensor or list containing all the activation maps of each layer of the network.
        - threshold: Float value determining the connections that should be visualised.
        - index: Integer for keeping track of how far (backwards) the function has backstepped into the network.
        - max_depth: Integer for the maximum depth to backstep to.
        - vis_depth: Integer for the layer in which only selected kerensl are to be visualised.
        - vis_num_kernels: Integer for the top k kernels to be visualised/backstepped in layer vis_depth

    [Returns]

        - layers_dict: A dictionary updated with all the connection paths between {net_depth,...,net_depth-max_depth} in pairs of (int,dir) per layer.


'''
def generate_indices(layers_dict, kernels, activations, threshold, index, max_depth, vis_depth, vis_num_kernels):
    print('Backstepping to depth -%d of maximum -%d'%(index,max_depth))
    print(index)
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


    [Args]

        - layers_dict: A dictionary containing nested dictionaries following the overall structure of the network. Kernel indices of layers correspond to keys and the values relate to the layer connections to kernels in the previous layer in which the activations were larger than a threshold value.
        - kernels: Tensor or list containing all the kernels of each layer in the network.
        - activations: Tensor or list containing all the activation maps of each layer of the network.
        - index: Integer for keeping track of how far (backwards) the function has backstepped into the network.
        - clip_len: Integer for the temporal dimension of the clip.
        - clip_height: Integer corresponding to the height of each frame.
        - clip_width: Integer corresponding to the width of each frame.

    [Returns]

        - layers_dict: A dictionary updated with all the connection paths between {net_depth,...,net_depth-max_depth} in pairs of (int,dir) per layer.


'''
def layer_visualisations(args, layers_dict, kernels, activations, index, RGB_video, tubes_dict = {}):
    # Main Iteration
    for key,value in layers_dict.items():

        # Recursive step
        if isinstance(value,dict):
            layers_dict[key],tubes_dict = layer_visualisations(args, value, kernels, activations, index+1, RGB_vid, tubes_dict)

        if not isinstance(layers_dict[key],dict):

            # Kernel visualisation happens here
            layerout = torch.tensor(activations[-index-1])
            pred_weights =  torch.tensor(kernels[-index])

            cam = torch.zeros([activations[-index-1].shape[0],1,activations[-index-1].shape[2],activations[-index-1].shape[3],activations[-index-1].shape[4]], dtype = torch.float32).cuda()
            # main loop for selected kernels

            print('Creating Saliency Tubes for :',str('layer_%d_num_kernels_%d'%(index,len(layers_dict))),' with kernels: ',layers_dict)

            # Apply padding - only to cases that there is a size mis-match
            _, _, kd, kh, kw = pred_weights.shape
            _, _, ad, ah, aw = layerout.shape

            needs_pad = True

            if (ad < kd or ah < kh or aw < kw):
                p3d = (1, 1, 1, 1, 1, 1)
                layerout = F.pad(layerout, p3d, "constant", 0)
                print('Padded input with pad :',p3d,layerout.shape)



            for i in layers_dict[key]:
                #print(i,type(i))
                #print(layerout.shape)
                #print(pred_weights[i, :, :, :].unsqueeze(0).shape)
                # Compute cam for every kernel
                cam += F.conv3d(layerout,pred_weights[i, :, :,  :, :].unsqueeze(0))




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
            heatmap_dir = os.path.join(args.base_output_dir, str('layer_%d_num_kernels_%d'%(index,len(layers_dict))), "heatmap")


            # produce heatmap for every frame and activation map
            tubes = []
            for frame_num in range(cam.shape[0]):
            #   Create colourmap
                heatmap = cv2.applyColorMap(np.uint8(255*cam[frame_num]), cv2.COLORMAP_COOL)

                super_threshold_indices = heatmap < 255 * .1
                heatmap[super_threshold_indices] = 0


                # Create frame with heatmap
                heatframe = heatmap//2 + RGB_vid[0][frame_num]//2
                tubes.append(heatframe)

            # Append a tuple of the computed heatmap and the kernels used
            tubes_dict[heatmap_dir]= (tubes,layers_dict[key])


        return [*layers_dict],tubes_dict
'''
---  E N D  O F  F U N C T I O N  L A Y E R _ V I S U A L I S A T I O N S  ---
'''









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

    paths = []
    dir = frame_dir.split('/')[-1]
    # Get framespaths to load from database
    for index in selected_frames:
        paths.append(os.path.join(str(dir),'frame_%05d'%index))

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
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--model_weights", type=str)
    parser.add_argument("--frame_dir", type=str)
    parser.add_argument("--frames_start", type=int)
    parser.add_argument("--frames_end", type=int)
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
predictions, kernels, activations = model_ft(torch.tensor(vid).cuda())

print('\n PREDICTIONS CALCULATED... \n')

# Get Linear layer weights
class_weights = model_ft.module.fc.weight.data.detach().cpu().numpy().transpose()

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
    print('Kernels shape',k.shape)
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
