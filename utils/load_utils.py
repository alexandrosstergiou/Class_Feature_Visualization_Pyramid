import os
import sqlite3
import numpy as np
import cv2
import torch
from models.resnet import *
from models.mfnet_3d import MFNET_3D

# Image loading utils #


def _make_torch_images(images):
    torch_images = torch.from_numpy(images.transpose(3, 0, 1, 2))
    torch_images = torch_images.float() / 255.0
    mean_3d = [124 / 255, 117 / 255, 104 / 255]
    std_3d = [0.229, 0.224, 0.225]
    for t, m, s in zip(torch_images, mean_3d, std_3d):
        t.sub_(m).div_(s)
    return torch_images.unsqueeze(0)


def _center_crop(data, tw=224, th=224):
    h, w, c = data.shape
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
    return cropped_data


def _per_frame_transform(next_image):
    scaled_img = cv2.resize(next_image, (256, 256), interpolation=cv2.INTER_LINEAR)  # resize to 256x256
    cropped_img = _center_crop(scaled_img)  # center crop 224x224
    final_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    return cropped_img, final_img


def load_images(frame_dir, frames_start, frames_end, fname_convention):
    # choose type of frame loader
    files = os.listdir(frame_dir)
    if 'frames.db' in files:
        # Create a list of frames based on start and end time
        selected_frames = [i for i in range(frames_start, frames_end)]
        rgb_vid, vid = _load_images_db(frame_dir, selected_frames=selected_frames)
    else:
        frame_indices = range(frames_start, frames_end)
        selected_frames = [os.path.join(frame_dir, fname_convention.format(x)) for x in frame_indices]
        rgb_vid, vid = _load_images_from_frames(frame_dir, selected_frames)

    return rgb_vid, vid


def _load_images_from_frames(frame_dir, selected_frames):
    final_images = np.zeros((len(selected_frames), 224, 224, 3))
    orig_images = np.zeros_like(final_images)
    for i, frame_name in enumerate(selected_frames):
        im_name = os.path.join(frame_dir, frame_name)
        next_image = cv2.imread(im_name, cv2.IMREAD_COLOR)
        cropped_img, final_img = _per_frame_transform(next_image)
        final_images[i] = final_img
        orig_images[i] = cropped_img

    return np.expand_dims(orig_images, 0), _make_torch_images(final_images)


def _load_images_db(frame_dir, selected_frames):
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


        cropped_img = _center_crop(img_np)
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


# Network loading utils #

def _load_resnet(model_name, num_classes, duration, sample_size=224):
    if model_name == 'resnet50':
        model_ft = resnet50(sample_size=sample_size, sample_duration=duration, num_classes=num_classes)
    elif model_name == 'resnet101':
        model_ft = resnet101(sample_size=sample_size, sample_duration=duration, num_classes=num_classes)
    elif model_name == 'resnet152':
        model_ft = resnet152(sample_size=sample_size, sample_duration=duration, num_classes=num_classes)
    elif model_name == 'resnet200':
        model_ft = resnet200(sample_size=sample_size, sample_duration=duration, num_classes=num_classes)
    else:  # dummy case, else is never accessed
        model_ft = None
    return model_ft


def _load_mfnet(num_classes):
    model_ft = MFNET_3D(num_classes)
    return model_ft


def load_network_structure(model_name, num_classes, sample_size, sample_duration):
    if model_name in ['resnet50', 'resnet101', 'resnet152', 'resnet200']:
        model_ft = _load_resnet(model_name, num_classes, sample_duration, sample_size)
        classification_layer_name = 'fc'
    elif model_name == 'mfnet':
        model_ft = _load_mfnet(num_classes)
        classification_layer_name = 'classifier'
    else:
        raise Exception('Unsupported model structure: {}'.format(model_name))
    return model_ft, classification_layer_name


# load on data parallel and cuda, load weights, import weights to network
def prepare_model(model_ft, weights_path):
    # Create parallel model for multi-gpus
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location={'cuda:1': 'cuda:0'})
    model_ft.load_state_dict(checkpoint['state_dict'], strict=False)
    # Set to evaluation mode
    model_ft.eval()
    return model_ft

