# Copyright (c) OpenMMLab. All rights reserved.
# python setup.py develop
# select the right interpreter
import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
# from nuscenes.nuscenes import NuScenesExplorer
from .nuscenes_explorer import NuScenesExplorer
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
import PIL.Image as Image
import cv2 

from mmdet.core.bbox.box_np_ops import points_cam2img

NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')


def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    # night_cnt = 0
    # rain_cnt = 0
    # both_cnt = 0
    # all_cnt = 0
    # for scene in nusc.scene:
    #     if 'night' in scene['description'].lower():
    #         night_cnt += 1
    #     if 'rain' in scene['description'].lower():
    #         rain_cnt += 1
    #         print(scene['description'])
    #     if 'rain' in scene['description'].lower() and 'night' in scene['description'].lower():
    #         both_cnt += 1
    #     all_cnt += 1
    # print('Nbr rain scenes: {}, nbr night scenes: {}, nbr both: {}, out of a total of {} scenes'.format(rain_cnt, night_cnt, both_cnt, all_cnt))
    for scene in nusc.scene:        
        # if 'rain' in scene['description'].lower() and 'after rain' not in scene['description'].lower():
        #     pass
        # else:
        #     continue
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

def get_lidar_img_info(cam_path):
    sting_list = cam_path.split('/')
    if '.jpg' in sting_list[5]:
        sting_list[5] = sting_list[5].replace('.jpg','.png')
    else: 
        raise Exception('Expecting jpg image format')
    sting_list[3]='lidar_samples/xz0'
    lidar_path_xz0 = '/'.join(sting_list)
    sting_list[3]='lidar_samples/rih'
    lidar_path_rih = '/'.join(sting_list)
    lidar_img_info= dict(rih=dict(file_name = lidar_path_rih), xz0=dict(file_name = lidar_path_xz0))
    return lidar_img_info

def get_radar_img_info(cam_path):
    sting_list = cam_path.split('/')
    if '.jpg' in sting_list[5]:
        sting_list[5] = sting_list[5].replace('.jpg','.png')
    else: 
        raise Exception('Expecting jpg image format')
    sting_list[3]='radar_samples/xz0'
    radar_path_xz0 = '/'.join(sting_list)
    sting_list[3]='radar_samples/riv'
    radar_path_riv = '/'.join(sting_list)
    radar_img_info= dict(riv=dict(file_name = radar_path_riv), xz0=dict(file_name = radar_path_xz0))
    return radar_img_info

def update_statistics(statistics, image, lidar_scale_factor, shift):
    image = image.astype(np.float32)
    image = image / lidar_scale_factor
    image = image - shift
    empty_value = 0
    # if empty_value != image[0,0,0]:
    #     raise Exception('Empty_value and does not match first pixel')
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

    return statistics

def disp_statistics_results(statistics, prefix=''):
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

    import json
    with open('../output/'+prefix+'_channel_statistics.json', 'a',encoding="utf-8") as file:
        json.dump(str(resutls), file)


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    x_end = 0
    y_end = 0
    # lidar_statistics = dict(
    #     means_per_image_masked = [], #np.zeros([len(nusc.sample), 3])
    #     variances_per_image_masked = [], #np.zeros([len(nusc.sample), 3])
    #     means_per_image_full = [], 
    #     variances_per_image_full = []) 
    # radar_statistics = dict(
    #     means_per_image_masked = [],
    #     variances_per_image_masked = [],
    #     means_per_image_full = [], 
    #     variances_per_image_full = []) 
    for sample in mmcv.track_iter_progress(nusc.sample):
        # if sample['scene_token'] in val_scenes:
        #     pass  
        # else:
        #     continue  
        # if x_end > 5 and y_end > 5:
        #     continue
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar_img': dict(),
            'radar_img': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        lidar_img_folder_paths = ['./data/nuscenes/lidar_samples/rih/','./data/nuscenes/lidar_samples/xz0/',
                                './data/nuscenes/radar_samples/riv/','./data/nuscenes/radar_samples/xz0/']
        for folder_path in lidar_img_folder_paths:
            for cam in camera_types:
                if not os.path.exists(folder_path + cam):
                    os.makedirs(folder_path + cam)
        explorer = NuScenesExplorer(nusc)
        for cam in camera_types:            
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

            lidar_img_info = get_lidar_img_info(cam_path)
            info['lidar_img'].update({cam: lidar_img_info})

            # radar
            radar_img_info = get_radar_img_info(cam_path)
            info['radar_img'].update({cam: radar_img_info})

            # Here we just grab the front camera and the point sensor.
            pointsensor_channel='LIDAR_TOP'
            sample_record = nusc.get('sample', sample['token'])
            pointsensor_token = sample_record['data'][pointsensor_channel]
            camera_token = sample_record['data'][cam]            
            points, _, points_3d_plus_features = explorer.map_pointcloud_to_image(pointsensor_token, 
                                                                camera_token,
                                                                render_intensity=True,
                                                                show_lidarseg=False,
                                                                filter_lidarseg_labels=None,
                                                                lidarseg_preds_bin_path=None,
                                                                show_panoptic=False,
                                                                return_3d_points = True,
                                                                with_img = False)
            distances = np.linalg.norm(points_3d_plus_features[:3, :], axis=0)
            intensities = points_3d_plus_features[3, :]
            points_3d = points_3d_plus_features[:3, :]
            # Set variables and resize points to target size
            lidar_scale_factor = 100  # store with 1mm accuracy
            shift = 200
            target_img_size = (640, 360)
            scale_factor = 2.5
            points_reszied = points[:2,:] /scale_factor    

            # Creating float image:
            width = target_img_size[0]
            height = target_img_size[1]
            background = lidar_scale_factor*shift
            img_empty = lidar_scale_factor*shift*np.ones((height, width, 6)).astype(np.uint16) # 6 channels so it it easier to store and save

            points_reszied_int = np.rint(points_reszied).T       
            points_reszied_int[:,0] = np.clip(points_reszied_int[:,0], 0, width-1)
            points_reszied_int[:,1] = np.clip(points_reszied_int[:,1], 0, height-1)
            points_reszied_int = points_reszied_int.astype(np.int32)  
            for point, distance, intensity, point_3d in zip(points_reszied_int, distances, intensities, points_3d.T):
                if img_empty[point[1],point[0],0] == lidar_scale_factor*shift or img_empty[point[1],point[0],0] > distance:
                    img_empty[point[1],point[0],0] = ((distance+shift)*lidar_scale_factor).astype(np.uint16)
                    img_empty[point[1],point[0],1] = ((intensity+shift)*lidar_scale_factor).astype(np.uint16)
                    img_empty[point[1],point[0],2] = ((-point_3d[1]+shift)*lidar_scale_factor).astype(np.uint16) # height
                    img_empty[point[1],point[0],3:5] = ((point_3d[[0,2]]+shift)*lidar_scale_factor).astype(np.uint16)

            # lidar_statistics = update_statistics(lidar_statistics, img_empty, lidar_scale_factor, shift)

            cv2.imwrite(lidar_img_info['rih']['file_name'], img_empty[:,:,:3])
            cv2.imwrite(lidar_img_info['xz0']['file_name'], img_empty[:,:,3:])

            info['lidar_img'][cam].update({'width': width, 'height': height, 
                                            'background': background, 'img_scale_factor': scale_factor})
            info['lidar_img'][cam]['rih'].update({'pixel_scale_factor': lidar_scale_factor, 'shift':shift, 'empty_channels': None}) 
            info['lidar_img'][cam]['xz0'].update({'pixel_scale_factor': lidar_scale_factor, 'shift':shift, 'empty_channels': [2]}) 

            # Radar  
            barcode = False   
            velocity_type = 'abs' # 'radial'     
            collected_points = []
            collected_distances = []    
            collected_intensities = []
            collected_points_3d = []
            collected_velocities = []
            collected_radar_xyz_endpoint = []            
            for pointsensor_channel in ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']:
                sample_record = nusc.get('sample', sample['token'])
                pointsensor_token = sample_record['data'][pointsensor_channel]
                camera_token = sample_record['data'][cam]            
                points, _, points_3d_plus_features, radar_xyz_endpoint = explorer.map_pointcloud_to_image(pointsensor_token, 
                                                                    camera_token,
                                                                    render_intensity=False,
                                                                    show_lidarseg=False,
                                                                    filter_lidarseg_labels=None,
                                                                    lidarseg_preds_bin_path=None,
                                                                    show_panoptic=False,
                                                                    return_radar = True,
                                                                    return_3d_points = True,
                                                                    with_img = False)
                distances = np.linalg.norm(points_3d_plus_features[[0,2], :], axis=0)
                if velocity_type == 'abs':    
                    # Ego motion compensated velocities
                    velocities = np.linalg.norm(np.array([points_3d_plus_features[8,:], points_3d_plus_features[9,:]]), axis=0) 
                elif velocity_type == 'radial':
                    # Calculate vrad comp
                    radial = np.array([points_3d_plus_features[0,:], points_3d_plus_features[1,:]]) # Calculate the distance vector
                    radial = radial / np.linalg.norm(radial, axis=0, keepdims=True)# Normalize these vectors
                    v = np.array([points_3d_plus_features[8,:], points_3d_plus_features[9,:]]) # Create the speed vector
                    velocities = np.sum(v*radial, axis=0, keepdims=True)[0,:] # Project the speed component onto this vector
                intensities = points_3d_plus_features[5, :] # RCS
                points_3d = points_3d_plus_features[:3, :]
                # Collect
                collected_points.append(points)
                collected_distances.append(distances)
                collected_intensities.append(intensities)
                collected_points_3d.append(points_3d) 
                collected_velocities.append(velocities)
                collected_radar_xyz_endpoint.append(radar_xyz_endpoint)
            points = np.concatenate(collected_points, axis =1)
            distances = np.concatenate(collected_distances, axis =0)
            intensities = np.concatenate(collected_intensities, axis =0)
            points_3d = np.concatenate(collected_points_3d, axis =1)
            velocities = np.concatenate(collected_velocities, axis =0)  
            radar_xyz_endpoint = np.concatenate(collected_radar_xyz_endpoint, axis =1)         
            
            # img_empty = np.array(im.resize(target_img_size))
            img_empty = lidar_scale_factor*shift*np.ones((height, width, 6)).astype(np.uint16) # 6 channels so it it easier to store and save

            radar_xyz_endpoint_reszied = radar_xyz_endpoint[:2,:] /scale_factor
            radar_xyz_endpoint_reszied = np.rint(radar_xyz_endpoint_reszied).T        
            radar_xyz_endpoint_reszied[:,0] = np.clip(radar_xyz_endpoint_reszied[:,0], 0, width-1)
            radar_xyz_endpoint_reszied[:,1] = np.clip(radar_xyz_endpoint_reszied[:,1], 0, height-1)
            radar_xyz_endpoint_reszied = radar_xyz_endpoint_reszied.astype(np.int32)  

            points_reszied = points[:2,:] /scale_factor    
            points_reszied_int = np.rint(points_reszied).T       
            points_reszied_int[:,0] = np.clip(points_reszied_int[:,0], 0, width-1)
            points_reszied_int[:,1] = np.clip(points_reszied_int[:,1], 0, height-1)  
            points_reszied_int = points_reszied_int.astype(np.int32)  
            for point, distance, intensity, point_3d, endpoint, velocity in zip(points_reszied_int, distances, intensities, points_3d.T, radar_xyz_endpoint_reszied, velocities):
                if img_empty[point[1],point[0],0] == lidar_scale_factor*shift or img_empty[point[1],point[0],0] > distance:
                    if img_empty[endpoint[1],point[0],0] == lidar_scale_factor*shift or img_empty[endpoint[1],point[0],0] > distance:
                        if barcode:
                            point[1] = img_empty.shape[0]
                            endpoint[1] = 0
                        if point[1] > endpoint[1]:
                            img_empty[endpoint[1]:point[1],point[0],0] = (np.ones((point[1]-endpoint[1]))*(distance+shift)*lidar_scale_factor).astype(np.uint16)
                            img_empty[endpoint[1]:point[1],point[0],1] = (np.ones((point[1]-endpoint[1]))*(intensity+shift)*lidar_scale_factor).astype(np.uint16)
                            img_empty[endpoint[1]:point[1],point[0],2] = (np.ones((point[1]-endpoint[1]))*(velocity+shift)*lidar_scale_factor).astype(np.uint16) # height
                            img_empty[endpoint[1]:point[1],point[0],3] = (np.ones((point[1]-endpoint[1]))*(point_3d[0]+shift)*lidar_scale_factor).astype(np.uint16)
                            img_empty[endpoint[1]:point[1],point[0],4] = (np.ones((point[1]-endpoint[1]))*(point_3d[2]+shift)*lidar_scale_factor).astype(np.uint16)
                            # img_empty[endpoint[1]:point[1],point[0],3:5] = (np.ones((point[1]-endpoint[1],2))*(point_3d[[0,2]]+shift)*lidar_scale_factor).astype(np.uint16)
                        else:
                            print('skipping radar point: point[1] < endpoint[1]: ', point[1], endpoint[1])

            # radar_statistics = update_statistics(radar_statistics, img_empty, lidar_scale_factor, shift)

            # cv2.imwrite('../output/radar_test_pillars_3m_1_3m_height_'+str(x_end)+cam+'.png', img_empty[:,:,::-1])
            cv2.imwrite(radar_img_info['riv']['file_name'], img_empty[:,:,:3])
            cv2.imwrite(radar_img_info['xz0']['file_name'], img_empty[:,:,3:])

            info['radar_img'][cam].update({'width': width, 'height': height, 
                                            'background': background, 'img_scale_factor': scale_factor})
            info['radar_img'][cam]['riv'].update({'pixel_scale_factor': lidar_scale_factor, 'shift':shift, 'empty_channels': None}) 
            info['radar_img'][cam]['xz0'].update({'pixel_scale_factor': lidar_scale_factor, 'shift':shift, 'empty_channels': [2]}) 

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
            x_end += 1
        else:
            val_nusc_infos.append(info)
            y_end += 1

    # disp_statistics_results(lidar_statistics, prefix='lidar')    
    # disp_statistics_results(radar_statistics, prefix='radar')   

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], lidar_projections=[], radar_projections=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['2', '3', '4'], # orig: ['', '1', '2', '3', '4']
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1 # I should add the lidar & radar here            

            lidar_img_info = info['lidar_img'][cam]
            lidar_img_info.update(
                    id=cam_info['sample_data_token']+'l',
                    token=info['token'])
            lidar_img_info['rih']['file_name'] = lidar_img_info['rih']['file_name'].split('data/nuscenes/')[-1]
            lidar_img_info['xz0']['file_name'] =lidar_img_info['xz0']['file_name'].split('data/nuscenes/')[-1]
            coco_2d_dict['lidar_projections'].append(lidar_img_info)

            # radar
            radar_img_info = info['radar_img'][cam]
            radar_img_info.update(
                    id=cam_info['sample_data_token']+'r',
                    token=info['token'])
            radar_img_info['riv']['file_name'] = radar_img_info['riv']['file_name'].split('data/nuscenes/')[-1]
            radar_img_info['xz0']['file_name'] =radar_img_info['xz0']['file_name'].split('data/nuscenes/')[-1]
            coco_2d_dict['radar_projections'].append(radar_img_info)

            
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NameMapping:
        return None
    cat_name = NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0
    coco_rec['visibility_token'] = repro_rec['visibility_token'] # store the visibility

    return coco_rec
