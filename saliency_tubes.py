import argparse
import torch
import json
import numpy as np
from utils.calc_utils import generate_indices
from utils.vis_utils import define_vis_kernels, layer_visualisations, savetopng
from utils.load_utils import load_images, load_network_structure, prepare_model


def parse_args():
    parser = argparse.ArgumentParser(description='saliency-tubes-base-parser')
    parser.add_argument("--model_name", type=str, choices=['resnet50', 'resnet101', 'resnet152', 'resnet200', 'mfnet'])
    parser.add_argument("--num_classes", type=int)  # 400
    parser.add_argument("--model_weights", type=str)
    parser.add_argument("--frame_dir", type=str)
    parser.add_argument("--frames_start", type=int)  # end - start = 16 frames
    parser.add_argument("--frames_end", type=int)  # duration = 16
    parser.add_argument("--fname_convention", type=str, default='frame_{:010d}.jpg')
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
# Load clip
RGB_vid, vid = load_images(args.frame_dir, args.frames_start, args.frames_end, args.fname_convention)

# load network structure
model_ft, classification_layer_name = load_network_structure(model_name=args.model_name, num_classes=args.num_classes,
                                                             sample_size=224, sample_duration=duration)
model_ft = prepare_model(model_ft, args.model_weights)
print('\n MODEL LOADED SUCESSFULLY... \n')

# get class prediction, all regularisation predictions and last convolution layer activation map
with torch.no_grad():
    predictions, kernels, activations = model_ft(torch.tensor(vid).cuda())
print('\n PREDICTIONS CALCULATED... \n')


# Get Linear layer weights
# cl_l_ind = list(dict(model_ft.module.named_children()).keys()).index(classification_layer_name)
# class_weights = model_ft.module[cl_l_ind].weight.data.detach().cpu().numpy().transpose()
class_weights = kernels[-1].data.detach().cpu().numpy().transpose()

# Minmax normalisation
base = class_weights.min()
weights_range = class_weights.max() - base
class_weights = np.asarray([(x-base)/weights_range for x in class_weights])

# Get class weights that are larger than threshold
kernel_indeces = [index for index,weight in enumerate(class_weights[:, args.label]) if (weight >= args.threshold)]

layers_weights_dict = dict()
layers_weights_dict[args.label] = kernel_indeces

# Define kernels to be visualised
layer_num, kernel_num = define_vis_kernels(args.visualisation_method)

for i, k in enumerate(kernels[:-1]):
    print()
    print('Kernels shape', k.shape)
    print('Activations shape', activations[i].shape)


# Create dictionary for layer indices
k_indices_dict = generate_indices(layers_dict=layers_weights_dict, kernels=kernels[:-1], activations=activations[:-1],
                                  threshold=args.threshold, index=1, max_depth=args.backprop_depth, vis_depth=layer_num,
                                  vis_num_kernels=kernel_num)

print(k_indices_dict)

# Save to JSON file
with open('class_dependency_graph.json', 'w') as fp:
    json.dump(k_indices_dict, fp)

# Call to get all saliency tubes and store them to a dictionary
_, tubes_dict = layer_visualisations(base_output_dir=args.base_output_dir, layers_dict=layers_weights_dict,
                                     kernels=kernels[:-1], activations=activations[:-1], index=1, RGB_video=RGB_vid)

print([*tubes_dict])

for filename,tube in tubes_dict.items():
    savetopng(tube, filename)
