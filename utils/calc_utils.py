import torch
from torch.functional import F


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

    # Start by computing activation map a^k_l(i)_l1 for each k_l(i), where i in indices
    for i in indices_l:

        # pointwise multiplication for activation a(l1) and the ith kernel in k_l
        #print('Kernel shape: ',k_l[i].shape)
        #print('Activation_shape: ',a_l1[0].shape)

        # Downsample the spatio-temporal dimension to create a global representation of the activation map and kernel.
        _, dk, hk, wk = list(k_l[i].size())
        _, da, ha, wa = list(a_l1[0].size())
        kernel = F.avg_pool3d(k_l[i], (dk, hk, wk)).squeeze(-1).squeeze(-1).squeeze(-1)
        act_map = F.avg_pool3d(a_l1[0], (da, ha, wa)).squeeze(-1).squeeze(-1).squeeze(-1)
        groups = act_map.shape[0] // kernel.shape[0]
        if groups > 1:
            kernel = kernel.repeat(groups)

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
    print(index)
    # Function termination after maximum depth is reached
    if index >= max_depth:
        print('END OF BRANCH DISCOVERY')
        return layers_dict
    # Iteration
    for key in layers_dict.keys():
        print('layerindex:', index, 'key:', key, 'keys dict:', layers_dict.keys(), '\n')
        # Unexplored connection
        if not isinstance(layers_dict[key],dict):
            # For layer that only specific kernels are to be visualised
            if index == vis_depth:
                layers_dict[key] = backprop_kernel_indices(layers_dict[key], kernels[-index], activations[-index-1],
                                                           threshold, vis_num_kernels)
            else:
                layers_dict[key] = backprop_kernel_indices(layers_dict[key], kernels[-index], activations[-index-1],
                                                           threshold)

        if index > max_depth:
            return layers_dict
        # Recursive step
        if isinstance(layers_dict[key], dict):
            print('Backstepping...')
            layers_dict[key] = generate_indices(layers_dict[key], kernels, activations, threshold, index+1, max_depth,
                                                vis_depth, vis_num_kernels)

    return layers_dict
'''
---  E N D  O F  F U N C T I O N  G E N E R A T E _ I N D I C E S  ---
'''
