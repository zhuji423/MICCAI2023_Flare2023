import numpy as np
import torch
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, List
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
import copy
import time
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3

def get_do_separate_z(spacing, anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def create_nonzero_mask(data):

    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"

    for c in range(data.shape[0]):

        this_mask = data[c] >= -325
        nonzero_mask =  this_mask

    return torch.from_numpy(nonzero_mask)


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = torch.where(mask != outside_value)
    minzidx = int(torch.min(mask_voxel_coords[0]))
    maxzidx = int(torch.max(mask_voxel_coords[0])) + 1
    minxidx = int(torch.min(mask_voxel_coords[1]))
    maxxidx = int(torch.max(mask_voxel_coords[1])) + 1
    minyidx = int(torch.min(mask_voxel_coords[2]))
    maxyidx = int(torch.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return copy.deepcopy(image[resizer])


def get_case_identifier(case):
    case_identifier = case[0].split("/")[-1].split(".nii.gz")[0][:-5]
    return case_identifier


def get_case_identifier_from_npz(case):
    case_identifier = case.split("/")[-1][:-4]
    return case_identifier


# def load_case_from_list_of_files(data_files, seg_file=None):
#     assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
#     properties = OrderedDict()
#     data_itk = [sitk.ReadImage(f) for f in data_files]
#
#     properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
#     properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
#     properties["list_of_data_files"] = data_files
#     properties["seg_file"] = seg_file
#
#     properties["itk_origin"] = data_itk[0].GetOrigin()
#     properties["itk_spacing"] = data_itk[0].GetSpacing()
#     properties["itk_direction"] = data_itk[0].GetDirection()
#
#     data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
#     ##################zhc############################
#     data_npy = np.squeeze(data_npy)# 降维
#     data_npy = np.clip(data_npy, -325, 325)
#     data_npy = data_npy.reshape(1,data_npy.shape[0], data_npy.shape[1], data_npy.shape[2]) # 在第一维增加
#     ##################zhc############################
# #     if seg_file is not None:
# #         seg_itk = sitk.ReadImage(seg_file)
# #         seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
# #     else:
#     seg_npy = None
#     return data_npy.astype(np.float32), seg_npy, properties

def load_case_from_list_of_files(data_files, seg_file=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    data_npy_copy = np.copy(data_npy)
    ###############################文物反
    data_npy_copy = np.squeeze(data_npy_copy)
    data_npy_copy = np.clip(data_npy_copy, -325, 325)
    data_npy_copy = data_npy_copy.reshape(1,data_npy_copy.shape[0], data_npy_copy.shape[1], data_npy_copy.shape[2])
    ######################


#     if seg_file is not None:
#         seg_itk = sitk.ReadImage(seg_file)
#         seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
#     else:
    seg_npy = None
    return data_npy.astype(np.float32), data_npy_copy.astype(np.float32), seg_npy, properties

def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    data = crop_to_bbox(data[0], bbox)[None]

    return data, bbox
def crop(data, properties, seg=None):
    shape_before = data.shape
    data, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
    shape_after = data.shape
    # print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
    #       np.array(properties["original_spacing"]), "\n")

    properties["crop_bbox"] = bbox

    data = np.clip(data,-325,325)
    data[np.isnan(data)] = 0
    properties["size_after_cropping"] = data[0].shape
    return data, properties


# def crop_from_list_of_files(data_files, seg_file=None):
#     #start = time.time()
#     data, seg, properties = load_case_from_list_of_files(data_files, seg_file)
#     #print('load_case_from_list_of_files:',time.time()-start)
#
#     return crop(data, properties, seg)
def crop_from_list_of_files(data_files, seg_file=None):
    #start = time.time()
    data, data_clip, seg, properties = load_case_from_list_of_files(data_files, seg_file)
    #print('load_case_from_list_of_files:',time.time()-start)
    crop_data_clip, properties = crop(data_clip, properties, seg)
    return data, crop_data_clip, properties



def resample_patient_coarse(data, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    """
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """

    shape = np.array(data[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)
    data = torch.from_numpy(data[None]).cuda()
    data_reshaped = (F.interpolate(data,size=list(new_shape),mode='trilinear', align_corners=True)[0]).cpu()
    del data
    return data_reshaped


from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize
from scipy.ndimage import map_coordinates
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing.pool import Pool
def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    """
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """

    shape = np.array(data[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z,
                                             order_z=order_z_data)

    return data_reshaped
def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param cval:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order,
                                                       **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None])
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data