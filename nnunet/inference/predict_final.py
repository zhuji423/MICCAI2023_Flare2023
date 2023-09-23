import time
load_time = time.time()
import torch.nn.functional as F
softmax_helper = lambda x: F.softmax(x, 1)
import glob
import argparse
from engine.fine_network import get_fine_model
from engine.coarse_network import get_coarse_model
from engine.organ_network import get_organ_model
from engine.preprocess import crop_from_list_of_files, resample_patient, resample_patient_coarse
from engine.postprocess import save_segmentation_nifti_from_softmax
from engine.utils import *
from engine.utils import _compute_steps_for_sliding_window

import cc3d
import fastremap


def keep_largest_connected_object(class_probabilities, label):
    labels_in = class_probabilities == label
    labels_out = cc3d.connected_components(labels_in, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    unvalid = [i[0] for i in candidates[1:]]
    seg_map = np.in1d(labels_out, unvalid).reshape(labels_in.shape)
    return seg_map

def keep_largest_connected_area(class_probabilities, class_):
    seg_map = np.ones_like(class_probabilities)
    for i in class_:
        seg_map -= keep_largest_connected_object(class_probabilities, i)
    return class_probabilities * seg_map

def predict_3D(x: np.ndarray, do_mirroring: bool = True, mirror_axes: Tuple[int, ...] = (0, 1, 2),
               use_sliding_window: bool = True,
               step_size: float = 0.5, patch_size: Tuple[int, ...] = [96, 128, 160],
               regions_class_order: Tuple[int, ...] = None,
               use_gaussian: bool = True, pad_border_mode: str = "constant",
               pad_kwargs: dict = None, all_in_gpu: bool = False,
               verbose: bool = False, mixed_precision: bool = True, ):
    if pad_kwargs is None:
        pad_kwargs = {'constant_values': 0}

    # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
    # code that uses this convention

    assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

    with torch.no_grad():
        if use_sliding_window:
            c, d, h, w = x.shape
            if max(h, w) > 360:
                ratio = 360.0 / max(h, w)
                x = torch.nn.functional.interpolate(torch.Tensor(x)[None], mode='trilinear', scale_factor=ratio,
                                                    align_corners=True)[0]

            res = _internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                    regions_class_order, use_gaussian, pad_border_mode,
                                                    pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                    verbose=verbose)

    return res

def _internal_predict_3D_3Dconv_tiled(x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                      patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                      pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                      verbose: bool):
    # better safe than sorry
    assert len(x.shape) == 4, "x must be (c, x, y, z)"

    if verbose: print("step_size:", step_size)
    if verbose: print("do mirror:", do_mirroring)

    assert patch_size is not None, "patch_size cannot be None for tiled prediction"

    # for sliding window inference the image must at least be as large as the patch size. It does not matter
    # whether the shape is divisible by 2**num_pool as long as the patch size is
    data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
    data_shape = data.shape  # still c, x, y, z

    # compute the steps for sliding window
    steps = _compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
    num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

    if verbose:
        print("data shape:", data_shape)
        print("patch size:", patch_size)
        print("steps (x, y, and z):", steps)
        print("number of tiles:", num_tiles)

    # we only need to compute that once. It can take a while to compute this due to the large sigma in
    # gaussian_filter
    data = data.half()
    add_for_nb_of_preds = torch.ones(patch_size, dtype=torch.float32)
    if patch_size == [64, 160, 224]:
        aggregated_results = torch.zeros([14] + list(data.shape[1:]), dtype=torch.float32)
        aggregated_nb_of_predictions = torch.zeros([14] + list(data.shape[1:]), dtype=torch.float32)
    else:
        aggregated_results = torch.zeros([15] + list(data.shape[1:]), dtype=torch.float32)
        aggregated_nb_of_predictions = torch.zeros([15] + list(data.shape[1:]), dtype=torch.float32)

    forward = time.time()
    for x in steps[0]:
        lb_x = x
        ub_x = x + patch_size[0]
        for y in steps[1]:
            lb_y = y
            ub_y = y + patch_size[1]
            for z in steps[2]:
                lb_z = z
                ub_z = z + patch_size[2]

                predicted_patch = _internal_maybe_mirror_and_pred_3D(model,
                                                                     data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z],
                                                                     mirror_axes, do_mirroring,
                                                                     None)[0]

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds
    #print('forward time', time.time() - forward)
    torch.cuda.empty_cache()

    # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
    slicer = tuple(
        [slice(0, aggregated_results.shape[i]) for i in
         range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
    aggregated_results = aggregated_results[slicer]
    aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

    # computing the class_probabilities by dividing the aggregated result with result_numsamples
    class_probabilities = aggregated_results / aggregated_nb_of_predictions

    #print('prediction before Resample:', class_probabilities.shape)
    class_probabilities[class_probabilities < 0.10] = 0
    return class_probabilities

def _internal_maybe_mirror_and_pred_3D(model, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                       do_mirroring: bool = True,
                                       mult: np.ndarray or torch.tensor = None):
    assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

    x = x.cuda()
    x = softmax_helper(model(x)).cpu()
    return x

def sample_resizeAndArgmax(class_probabilities, size_after_cropping):  # Numpy to Numpy
    if size_after_cropping[0] > 900:
        class_probabilities = resize_and_argmax(class_probabilities, list(size_after_cropping), step=4)
    elif size_after_cropping[0] > 550:
        class_probabilities = resize_and_argmax(class_probabilities, list(size_after_cropping), step=3)
    elif size_after_cropping[0] > 380:
        class_probabilities = resize_and_argmax(class_probabilities, list(size_after_cropping), step=2)
    else:

        class_probabilities = class_probabilities.cuda()
        class_probabilities = \
        torch.nn.functional.interpolate(class_probabilities[None], mode='trilinear', size=(list(size_after_cropping)),
                                        align_corners=True)[0]
        class_probabilities = torch.argmax(class_probabilities, 0).cpu().numpy()
    return class_probabilities

def get_bbox1_from_mask(mask, outside_value=0):
    mask_shape = mask.shape
    mask_voxel_coords = torch.where(mask != outside_value)
    minzidx = max(int(torch.min(mask_voxel_coords[0])) - int(20 / space[0]), 0)
    maxzidx = min(int(torch.max(mask_voxel_coords[0])) + int(20 / space[0]), mask_shape[0])

    minxidx = max(int(torch.min(mask_voxel_coords[1])) - int(20 / space[1]), 0)
    maxxidx = min(int(torch.max(mask_voxel_coords[1])) + int(20 / space[1]), mask_shape[1])

    minyidx = max(int(torch.min(mask_voxel_coords[2])) - int(20 / space[2]), 0)
    maxyidx = min(int(torch.max(mask_voxel_coords[2])) + int(20 / space[2]), mask_shape[2])
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

def mask_to_bbox(image, mask1, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    image = np.zeros(tuple(image.shape))
    image[resizer] = mask1
    return image

parser = argparse.ArgumentParser(description='full functional execute script of flare seg module.')
parser.add_argument('-i', '--input_path', type=str, default=r'../Flare2023/validation/', help='input path')
parser.add_argument('-o', '--output_path', type=str, default=r'../Flare2023/validation_out/', help='output path')

args = parser.parse_args()


#####--------------------------------------------#####--------------------------------------------####

paths = glob.glob(args.input_path + '/*_0000.nii.gz')

for input_file_nii_gz in paths:
    start = time.time()
    import os
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    output_filename = args.output_path + '/' + input_file_nii_gz.split('\\')[-1].replace('_0000.nii.gz', '.nii.gz')
    crop_data, crop_data_clip, properties = crop_from_list_of_files([input_file_nii_gz])

    # stage 1
    ####--------------------------------------------#####--------------------------------------------####
    start1 = time.time()
    lower_bound = -325.0
    upper_bound = 266.0
    stage1_data_resample = np.clip(crop_data_clip, lower_bound, upper_bound)
    stage1_data_resample = (stage1_data_resample - 59.284187) / 83.43879

    stage1_data_resample = resample_patient_coarse(stage1_data_resample, np.array(properties["original_spacing"]),
                                   np.array([2.3009669, 1.26433326, 1.26433326]),
                                   3, 1, force_separate_z=False, order_z_data=0, order_z_seg=0)  # 1,z,x,y
    torch.cuda.empty_cache()

    model = get_coarse_model(path='engine/coarse_model.pth')
    stage1_class_probabilities = predict_3D(stage1_data_resample, patch_size=[64, 160, 224], step_size=1)  # return torch-array

    stage1_class_probabilities = slice_argmax(stage1_class_probabilities)  # return np-array
    sample_centroid = get_centroid(stage1_class_probabilities, center_class=1)
    if sample_centroid != None:
        stage1_class_probabilities = del_fp(stage1_class_probabilities, sample_centroid,
                                            score=170 / 3)  # return np-array

    stage1_class_probabilities = keep_largest_connected_area(stage1_class_probabilities,
                                                       [1, 2, 3, 4, 5, 6, 9, 10, 11, 13])
    stage1_class_probabilities[stage1_class_probabilities == 7] = 0
    stage1_class_probabilities[stage1_class_probabilities == 8] = 0
    stage1_class_probabilities[stage1_class_probabilities == 12] = 0
    stage1_class_probabilities = torch.from_numpy(stage1_class_probabilities).half().cuda()
    stage1_class_probabilities = torch.nn.functional.interpolate(stage1_class_probabilities[None][None], mode='nearest',
                                                                 size=(list(properties['size_after_cropping'])))[0][
        0].cpu()
    torch.cuda.empty_cache()
    print('stage1 time: ', time.time() - start1)
    space = properties['original_spacing']

    # stage 2
    #####--------------------------------------------#####--------------------------------------------####
    start2 = time.time()
    mask = stage1_class_probabilities > 0
    bbox = get_bbox1_from_mask(mask)
    stage2_data_ori = crop_to_bbox(crop_data_clip[0], bbox)
    stage2_data_shape = stage2_data_ori.shape
    lower_bound = -325.0
    upper_bound = 273
    stage2_data = np.clip(stage2_data_ori, lower_bound, upper_bound)
    stage2_data = (stage2_data - 86.59582) / 81.018715

    stage2_data_resample = resample_patient_coarse(stage2_data[None], np.array(properties["original_spacing"]),
                                   np.array([1.7632679173989854, 1.4395425277931828, 1.4395425277931828]),
                                   3, 1, force_separate_z=False, order_z_data=0, order_z_seg=0)  # 1,z,x,y
    torch.cuda.empty_cache()
    model = get_fine_model(path='engine/tumor_model_1.pth')
    stage2_class_probabilities = predict_3D(stage2_data_resample, patch_size=[96, 160, 192], step_size=0.5)
    stage2_class_probabilities = sample_resizeAndArgmax(stage2_class_probabilities, stage2_data_shape)
    torch.cuda.empty_cache()
    stage2_class_probabilities = keep_largest_connected_area(stage2_class_probabilities,[1,2,3,4,5,6,9,10,11,13])
    class_probabilities = mask_to_bbox(stage1_class_probabilities, stage2_class_probabilities, bbox)
    print('stage2 time: ', time.time() - start2)
    # stage 3
    #####--------------------------------------------#####--------------------------------------------####
    start3 = time.time()
    stage3_data = stage2_data
    stage3_data_shape = stage3_data.shape
    stage3_data = resample_patient_coarse(stage3_data[None], np.array(properties["original_spacing"]),
                                       np.array([1.7632679173989854, 1.4395425277931828, 1.4395425277931828]),
                                       3, 1, force_separate_z=False, order_z_data=0, order_z_seg=0)  # 1,z,x,y
    torch.cuda.empty_cache()
    model = get_fine_model(path='engine/tumor_model_2.pth')
    # model = get_fine_model(path=args.best_path)
    stage3_class_probabilities = predict_3D(stage3_data, patch_size=[96, 128, 192], step_size=0.5)
    stage3_class_probabilities = sample_resizeAndArgmax(stage3_class_probabilities, stage3_data_shape)

    torch.cuda.empty_cache()
    #stage3_class_probabilities = keep_largest_connected_area(stage3_class_probabilities,[1,2,3,4,5,6,9,10,11,13])
    stage3_class_probabilities = mask_to_bbox(stage1_class_probabilities, stage3_class_probabilities, bbox)  #0.9026 0.9562 0.4582 0.3852

    temp_cls_probabilities = class_probabilities

    indices = np.where(stage3_class_probabilities == 14)
    indices1 = np.where(temp_cls_probabilities == 14)
    temp_cls_probabilities[indices] = 14

    indices2 = np.where(temp_cls_probabilities == 14)
    for idx in range(len(indices[0])):
        x, y, z = indices2[0][idx], indices2[1][idx], indices2[2][idx]
        if stage3_class_probabilities[x, y, z] != 14:
            temp_cls_probabilities[x, y, z] = stage3_class_probabilities[x, y, z]
            with open('log.txt', 'a') as f:
                f.write( f'{input_file_nii_gz}:enter organ save\n')

    class_probabilities = temp_cls_probabilities
    print('stage3 time: ', time.time() - start3)
    indices4 = np.where(class_probabilities == 14)
    print('---------------input name---------:', input_file_nii_gz)
    print('---------------costa time----------: ', time.time() - start)
    #stage 4
    ####--------------------------------------------#####--------------------------------------------####
    stage4_data = stage2_data_ori
    stage4_data_shape = stage4_data.shape

    lower_bound = -325.0
    upper_bound = 267.0
    stage4_data = np.clip(stage4_data, lower_bound, upper_bound)
    stage4_data = (stage4_data - 70.34444) / 82.539795

    stage4_data = resample_patient_coarse(stage4_data[None], np.array(properties["original_spacing"]),
                                   np.array([2.5, 1.25834116, 1.25834116]),
                                   3, 1, force_separate_z=False, order_z_data=0, order_z_seg=0)  # 1,z,x,y
    torch.cuda.empty_cache()
    model = get_organ_model(path='engine/organ_model.pth')
    stage4_class_probabilities = predict_3D(stage4_data, patch_size=[64, 160, 224], step_size=0.5)  # return torch-array
    stage4_class_probabilities = sample_resizeAndArgmax(stage4_class_probabilities, stage4_data_shape)
    torch.cuda.empty_cache()
    stage4_class_probabilities = keep_largest_connected_area(stage4_class_probabilities,[1,2,3,4,5,6,9,10,11,13])
    stage4_class_probabilities = mask_to_bbox(stage1_class_probabilities, stage4_class_probabilities, bbox)
    class_probabilities[class_probabilities < 14] = 0
    class_probabilities = class_probabilities + stage4_class_probabilities
    class_probabilities[class_probabilities > 13] = 14
    print(output_filename)
    save_segmentation_nifti_from_softmax(class_probabilities, output_filename, properties, 1, None, None, None, None,
                                                 None, None, 0)
    print('---------------input name---------:', input_file_nii_gz)
    print('---------------costa time----------: ', time.time() - start)
print('fina time:', time.time() - load_time) # 总时间：1421s