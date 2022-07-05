from SeeingThroughFog.tools.DatasetViewer.lib.read import load_velodyne_scan
from SeeingThroughFog.tools.DatasetViewer.lib.read import load_calib_data
from SeeingThroughFog.tools.ProjectionTools.Lidar2RGB.lib.utils import filter, \
    find_closest_neighbors, find_missing_points, transform_coordinates
from SeeingThroughFog.tools.ProjectionTools.Lidar2RGB.lib.visi import plot_spherical_scatter_plot, plot_image_projection, get_pc_projection
from SeeingThroughFog.tools.DatasetStatisticsTools.lib_stats.util import read_split
from SeeingThroughFog.tools.DatasetViewer.lib.read import load_radar_points

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json

import os
import argparse


def parsArgs():
    parser = argparse.ArgumentParser(description='Lidar 2d projection tool')
    parser.add_argument('--root', '-r', help='Enter the root folder', default='data/dense/')
    parser.add_argument('--split_dir', type=str, help='definde split directory', default='./SeeingThroughFog/splits')
    parser.add_argument('--label_file', '-l', help='label file', default='all')
    parser.add_argument('--frame', '-f', help='frame to project to', default='RGB2Gatedv2')

    parser.add_argument('--lidar_type', '-t', help='Enter the root folder', default='lidar_hdl64',
                        choices=['lidar_hdl64', 'lidar_vlp32'])
    parser.add_argument('--calib_root', '-c', help='Path to callibration files', default='./SeeingThroughFog/tools/DatasetViewer/calibs')

    args = parser.parse_args()

    return args

def create_img(img_coordinates, pts_3D, r, cfg, radar=False):
    # Set variables and resize points to target size
    lidar_scale_factor = cfg['lidar_scale_factor']  # store with 1mm accuracy
    shift = cfg['shift']
    target_img_size = r.dsize# (640, 360)

    # Creating float image:
    width = target_img_size[0]
    height = target_img_size[1]
    background = 0

    img_coordinates = img_coordinates.astype(dtype=np.int32) #.T ?
    image = lidar_scale_factor * shift * np.ones((width, height, 3)).astype(
        np.uint16)

    values = (pts_3D + shift) * lidar_scale_factor
    if not radar:
        image[img_coordinates[:, 0], img_coordinates[:, 1], :] = values
    elif radar:
        for point, value in zip(img_coordinates, values):
            if image[point[0], point[1], 0] == lidar_scale_factor * shift or image[
                point[0], point[1], 0] > value[1]:
                image[point[0], :, 0] = value[0] # height y
                image[point[0], :, 1] = value[1] # depth z
                image[point[0], :, 2] = value[2] # velocity v


    return image.transpose([1, 0, 2]).squeeze()

    # info['lidar_img'][cam].update({'width': width, 'height': height,
    #                                'background': background, 'img_scale_factor': scale_factor})
    # info['lidar_img'][cam]['rih'].update({'pixel_scale_factor': lidar_scale_factor, 'shift':shift, 'empty_channels': None})
    # info['lidar_img'][cam]['xz0'].update({'pixel_scale_factor': lidar_scale_factor, 'shift':shift, 'empty_channels': [2]})

def update_statistics(statistics, image, cfg):
    lidar_scale_factor = cfg['lidar_scale_factor']
    shift = cfg['shift']

    image = image.astype(np.float32)
    image = image / lidar_scale_factor
    image = image - shift
    empty_value = 0
    num_channels = image.shape[2]

    mask = (image[:,:,0]!=empty_value)
    masked_image = np.zeros((sum(sum(mask)),num_channels))
    for i in range(num_channels):
        masked_image[:,i] = image[:,:,i][mask]
    # masked_image = image[mask]

    # Mean pixel value for processed image.
    statistics['means_per_image_masked'].append(np.mean(masked_image, axis=0))
    statistics['means_per_image_full'].append(np.mean(np.mean(np.array(image), axis=1), axis=0))
    # % Variances of pixel values in each channel separately.
    # % NOTE: Variance is computed using the unadjusted formula where the original
    # % number of samples appears in the denominator, which corresponds to a
    # % biased estimator. However, the total number of pixels in each image is
    # % large enough for the resulting bias to be negligible.
    variances = np.zeros([num_channels])
    for i in range(num_channels):
        variances[i] = np.var(masked_image[:,i])
    statistics['variances_per_image_masked'].append(variances)

    variances = np.zeros([num_channels])
    for i in range(num_channels):
        variances[i] = np.var(image[:,:,i])
    statistics['variances_per_image_full'].append(variances)
    # statistics['variances_per_image'][i, 1] = np.var(np.array(image_2))
    # statistics['variances_per_image'][i, 2] = np.var(np.array(image_3))

    return statistics

def disp_statistics_results(statistics, prefix='', empty_list=False, no_file_list=False):
    resutls = {'legend': ['range', 'intensity', 'height(-y)', 'x', 'height(z)', 'empty channel']}
    # % Final aggregation across the entire dataset.
    statistics['means_per_image_masked'] = np.array(statistics['means_per_image_masked'])
    statistics['variances_per_image_masked'] = np.array(statistics['variances_per_image_masked'])
    mean_dataset = np.mean(statistics['means_per_image_masked'], axis=0)
    variance_of_means = np.var(statistics['means_per_image_masked'],  axis=0)
    mean_of_variances = np.mean(statistics['variances_per_image_masked'], axis=0)
    variance_dataset = variance_of_means + mean_of_variances
    std_dataset = np.sqrt(variance_dataset)
    resutls['mean_dataset_masked'] = list(mean_dataset)
    resutls['std_dataset_masked'] = list(std_dataset)

    print(prefix+' mean_dataset masked: ', mean_dataset)
    print(prefix+' std_dataset masked: ', std_dataset)
    print(prefix+' mean_dataset masked mmdet: ', mean_dataset*255)
    print(prefix+' std_dataset masked mmdet: ', std_dataset*255)

    # On full image:
    mean_dataset = np.mean(statistics['means_per_image_full'], axis=0)
    variance_of_means = np.var(statistics['means_per_image_full'],  axis=0)
    mean_of_variances = np.mean(statistics['variances_per_image_full'], axis=0)
    variance_dataset = variance_of_means + mean_of_variances
    std_dataset = np.sqrt(variance_dataset)
    resutls['mean_dataset_full'] = list(mean_dataset)
    resutls['std_dataset_full'] = list(std_dataset)

    print(prefix+' mean_dataset full: ', mean_dataset)
    print(prefix+' std_dataset full: ', std_dataset)
    print(prefix+' mean_dataset full mmdet: ', mean_dataset*255)
    print(prefix+' std_dataset full mmdet: ', std_dataset*255)

    if empty_list:
        resutls['no_points_list'] = statistics['no_points_list']
        print('number of images without radar points:', len(statistics['no_points_list']))
    if no_file_list:
        resutls['no_file_list'] = statistics['no_file_list']
        print('number of images without lidar file:', len(statistics['no_file_list']))

    with open('./'+prefix+'_channel_statistics.json', 'a',encoding="utf-8") as file:
        json.dump(str(resutls), file)

interesting_samples = [
    '2018-02-06_14-25-51_00400',
    '2019-09-11_16-39-41_01770',
    '2018-02-12_07-16-32_00100',
    '2018-10-29_16-42-03_00560',
]

echos = [
    ['last', 'strongest'],
]



if __name__ == '__main__':

    args = parsArgs()

    lidar_statistics = dict(
        means_per_image_masked = [], #np.zeros([len(nusc.sample), 3])
        variances_per_image_masked = [], #np.zeros([len(nusc.sample), 3])
        means_per_image_full = [],
        variances_per_image_full = [],
        no_file_list = [])
    radar_statistics = dict(
        means_per_image_masked = [],
        variances_per_image_masked = [],
        means_per_image_full = [],
        variances_per_image_full = [],
        no_points_list = [])

    cfg = dict(
        lidar_scale_factor = 100,   # store with 1mm accuracy
        shift = 200)
    empty_dsize = (1280, 768)

    velodyne_to_camera, camera_to_velodyne, P, R, vtc, radar_to_camera, zero_to_camera = load_calib_data(
        args.calib_root, name_camera_calib='calib_cam_stereo_left.json', tf_tree='calib_tf_tree_full.json',
        velodyne_name='lidar_hdl64_s3_roof' if args.lidar_type == 'lidar_hdl64' else 'lidar_vlp32_roof')
    rtc = np.matmul(np.matmul(P, R), radar_to_camera)

    seleced_files = read_split(args.split_dir, args.label_file + '.txt')

    for sample in tqdm.tqdm(seleced_files):
        lidar_proj_file_path = os.path.join(args.root, f'lidar_samples_{args.frame}/yzi',
                                            args.lidar_type + '_' + echos[0][1])
        lidar_proj_file = os.path.join(lidar_proj_file_path, sample + '.png')
        radar_proj_file_path = os.path.join(args.root, f'radar_samples_{args.frame}/yzv')
        radar_proj_file = os.path.join(radar_proj_file_path, sample + '.png')
        if os.path.exists(lidar_proj_file) and os.path.exists(radar_proj_file):
            continue

        # Lidar
        velo_file_strongest = os.path.join(args.root, args.lidar_type + '_' + echos[0][1],
                                           sample + '.bin')
        if os.path.exists(velo_file_strongest):
            lidar_data_strongest = load_velodyne_scan(velo_file_strongest)

            # filters point below distance threshold
            lidar_data_strongest = filter(lidar_data_strongest, 1.5)

            img_coordinates, pts_3D_yzi, r = get_pc_projection(lidar_data_strongest, vtc, velodyne_to_camera,
                                                               frame=args.frame)
            image = create_img(img_coordinates, pts_3D_yzi, r, cfg)

            lidar_statistics = update_statistics(lidar_statistics, image, cfg)
        else:
            raise Exception(f'Lidar file is missing {velo_file_strongest}')
            # lidar_statistics['no_file_list'].append(sample)
            # image = cfg['lidar_scale_factor'] * cfg['shift'] * np.ones((empty_dsize[1], empty_dsize[0], 3)).astype(np.uint16)

        lidar_proj_file_path = os.path.join(args.root, f'lidar_samples/yzi', args.lidar_type + '_' + echos[0][1])
        os.makedirs(lidar_proj_file_path, exist_ok = True)
        lidar_proj_file = os.path.join(lidar_proj_file_path,sample + '.png')
        cv2.imwrite(lidar_proj_file, image)

        # Radar
        radar_file = os.path.join(args.root, 'radar_targets',
                                  sample + '.json')
        if os.path.exists(radar_file):
            radar_data = load_radar_points(radar_file)
            if len(radar_data) == 0:
                radar_statistics['no_points_list'].append(sample)
                image = cfg['lidar_scale_factor'] * cfg['shift'] * np.ones((r.dsize[1], r.dsize[0], 3)).astype(np.uint16)
            else:
                img_coordinates, pts_3D_yzv, r = get_pc_projection(radar_data, rtc, radar_to_camera,
                                                                frame=args.frame)
                image = create_img(img_coordinates, pts_3D_yzv, r, cfg, radar=True)

                radar_statistics = update_statistics(radar_statistics, image, cfg)
        else:
            raise Exception(f'Radar file is missing {radar_file}')

        radar_proj_file_path = os.path.join(args.root, f'radar_samples/yzv') #riv
        os.makedirs(radar_proj_file_path, exist_ok=True)
        radar_proj_file = os.path.join(radar_proj_file_path, sample + '.png')
        cv2.imwrite(radar_proj_file, image)

    # disp_statistics_results(lidar_statistics, prefix= args.lidar_type + '_' + echos[0][1], no_file_list=True)
    # disp_statistics_results(radar_statistics, prefix='radar', empty_list=True)
    print('Finished radar and lidar projection')
