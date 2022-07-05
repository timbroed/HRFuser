# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import os

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class LoadGatedImageFromFile:
    """Load an gated image from file.
    """

    def __init__(self,
                 to_float32=True,
                 only_acc=False,
                 color_type='unchanged',
                 gated_folders = ['gated_full_rect', 'gated_full_acc_rect'],
                 file_client_args=dict(backend='disk'),
                 pad=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.only_acc = only_acc
        self.gated_folders = gated_folders
        self.pad = pad

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        img_name = results['img_info']['filename'].split('/')[1]
        
        if results['img_prefix'] is not None:
            if not self.only_acc:
                filename = osp.join(results['img_prefix'],self.gated_folders[0],img_name)
                if not os.path.exists(filename):
                    filename = osp.join(results['img_prefix'],self.gated_folders[1],img_name)
            else:
                filename = osp.join(results['img_prefix'],self.gated_folders[1],img_name)
        else:
            filename = osp.join(self.gated_folders[0],img_name)
            if not self.only_acc:
                filename = osp.join(self.gated_folders[0],img_name)
                if not os.path.exists(filename):
                    filename = osp.join(self.gated_folders[1],img_name)
            else:
                filename = osp.join(self.gated_folders[1],img_name)

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        
        if self.pad:
            img = mmcv.impad(img, padding=self.pad)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        results['gated_filename'] = filename
        results['gated_img'] = img
        results['gated_img_shape'] = img.shape
        results['gated_ori_shape'] = img.shape
        results['img_fields'].append('gated_img')
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'only_acc={self.only_acc}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args}), '                    
                    f'pad={self.pad})')
        return repr_str

@PIPELINES.register_module()
class LoadStackedGatedImageFromFile:

    def __init__(self,
                 to_float32=True,
                 only_acc=False,
                 color_type='unchanged',
                 gated_folders = ['gated0_rect', 'gated1_rect', 'gated2_rect'],
                 file_client_args=dict(backend='disk'),
                 pad=None,
                 expected_shape=(720, 1280)):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.only_acc = only_acc
        self.gated_folders = gated_folders
        self.pad = pad
        self.expected_shape = expected_shape

    def __call__(self, results):

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        img_name = results['img_info']['filename'].split('/')[1]
        
        if results['img_prefix'] is not None:
            filenames = [osp.join(results['img_prefix'], gated_folder, img_name)
                for gated_folder in self.gated_folders]
        else:
            filenames = [osp.join(gated_folder, img_name)
                for gated_folder in self.gated_folders]
        
        # Load and combine depth and intensity images
        img = []
        for filename in filenames:
            if os.path.exists(filename):
                img_bytes = self.file_client.get(filename)
                loaded_img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
                if not (loaded_img.shape == self.expected_shape):
                    raise Exception ('Unexpected gated image shape')
            else:
                loaded_img = np.zeros(self.expected_shape)
                print(f'Missing gated image: {filename}')
            if len(loaded_img.shape) == 2:
                loaded_img = np.expand_dims(loaded_img, axis=2)
            img.append(loaded_img)

        if len(self.gated_folders) > 1:
            img = np.concatenate(img, axis=2)

        if self.to_float32:
            img = img.astype(np.float32)

        if self.pad:
            img = mmcv.impad(img, padding=self.pad)

        results['gated_filenames'] = filenames
        results['gated_img'] = img
        results['gated_img_shape'] = img.shape
        results['gated_ori_shape'] = img.shape
        results['img_fields'].append('gated_img')
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'only_acc={self.only_acc}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args}, '
                    f'color_type={self.color_type}, '
                    f'pad={self.pad}, '
                    f'expected_shape={self.expected_shape})')
        return repr_str

@PIPELINES.register_module()
class LoadProjectedSensorImageFile:
    """Load projected sensor images.
        The loading is adapted from LoadMultiChannelImageFromFiles
    """
    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk'),
                 sensor_type='LIDAR', # lidar or radar
                 channels=['rih'], # Lidar channels 'rih' or 'xz0'
                 with_mask=False,
                 delete_channels=None,
                 expected_shape=(360, 640, 3)):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.sensor_type = sensor_type
        self.channels = channels
        self.with_mask = with_mask
        self.delete_channels = delete_channels
        self.expected_shape = expected_shape
        if sensor_type == 'lidar':
            self.prefix = 'lidar_prefix'  
            self.sensor_info = 'lidar_info'
        elif sensor_type == 'radar':
            self.prefix = 'radar_prefix'  
            self.sensor_info = 'radar_info'
        else:
            raise Exception('Only sensor types radar and lidar are supported')

    def __call__(self, results):
        
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results[self.prefix] is not None:
            filenames = [
                osp.join(results[self.prefix], results[self.sensor_info][channel]['file_name'])
                for channel in self.channels]
        else:
            filenames = [results[self.sensor_info][channel]['file_name']
                for channel in self.channels
            ]
        
        # Load and combine depth and intensity images
        img = []
        for name, channel in zip(filenames, self.channels):
            # Load image
            img_bytes = self.file_client.get(name)
            loaded_img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
            if not (loaded_img.shape == self.expected_shape):
                print(f'\nUnexpected shape {loaded_img.shape} in image {name}')
                if loaded_img.min() == loaded_img.max():
                    print('Empty image -> trying to continue with swaped axis')
                    loaded_img = np.swapaxes(loaded_img, 0, 1)
                else:
                    raise Exception ('Unexpected image shape')

            # Delete empty channels
            if results[self.sensor_info][channel]['empty_channels']:
                previous_channel = 99
                for del_channel in results[self.sensor_info][channel]['empty_channels'][::-1]:
                    if del_channel >= previous_channel:
                        raise Exception('Channels have to be in ascending order') 
                    loaded_img = np.delete(loaded_img, del_channel, axis=2) 
                    previous_channel = del_channel
            if self.delete_channels:
                for del_channel in self.delete_channels:
                    loaded_img = np.delete(loaded_img, del_channel, axis=2)

            if self.to_float32:
                loaded_img = loaded_img.astype(np.float32) 
            else:
                raise Exception('Loaded sensor images need ot be run in float32 mode')

            # Get  
            loaded_img /= results[self.sensor_info][channel]['pixel_scale_factor']
            loaded_img -= results[self.sensor_info][channel]['shift']            
            if len(self.channels) > 1:        
                img.append(loaded_img)
            else:
                img = loaded_img
        if len(self.channels) > 1:
            img = np.concatenate(img, axis=2)

        # Mask for image normalisation of only the 
        if self.with_mask:
            mask = (loaded_img[:,:,0] != 0)
            results[self.with_mask] = mask

        results[self.sensor_type+'_filenames'] = filenames    
        results[self.sensor_type+'_img'] = img
        results[self.sensor_type+'_img_shape'] = img.shape
        results[self.sensor_type+'_ori_shape'] = img.shape
        results['img_fields'].append(self.sensor_type+'_img')
        return results
    
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args}, '
                    f'sensor_type={self.sensor_type}, '
                    f'channels={self.channels}, '
                    f'with_mask={self.with_mask}, '
                    f'delete_channels={self.delete_channels}, '
                    f'expected_shape={self.expected_shape})')
        return repr_str

@PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles:
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']

        img = []
        for name in filename:
            img_bytes = self.file_client.get(name)
            img.append(mmcv.imfrombytes(img_bytes, flag=self.color_type))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
       with_visibility (bool): Wehther to laod the visibilities. Default: False
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 with_visibility=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.with_visibility = with_visibility
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results
    
    def _load_visibility(self, results):
        """Private function to load visibility annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_visibilities'] = results['ann_info']['visibilities'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_visibility:
            results = self._load_visibility(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadPanopticAnnotations(LoadAnnotations):
    """Load multiple types of panoptic annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 with_seg=True,
                 with_visibility=False,
                 file_client_args=dict(backend='disk')):
        if rgb2id is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        super(LoadPanopticAnnotations,
              self).__init__(with_bbox, with_label, with_mask, with_seg, with_visibility, True,
                             file_client_args)

    def _load_masks_and_semantic_segs(self, results):
        """Private function to load mask and semantic segmentation annotations.

        In gt_semantic_seg, the foreground label is from `0` to
        `num_things - 1`, the background label is from `num_things` to
        `num_things + num_stuff - 1`, 255 means the ignored label (`VOID`).

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask and semantic segmentation
                annotations. `BitmapMasks` is used for mask annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        pan_png = rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for mask_info in results['ann_info']['masks']:
            mask = (pan_png == mask_info['id'])
            gt_seg = np.where(mask, mask_info['category'], gt_seg)

            # The legal thing masks
            if mask_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))

        if self.with_mask:
            h, w = results['img_info']['height'], results['img_info']['width']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

        if self.with_seg:
            results['gt_semantic_seg'] = gt_seg
            results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types panoptic annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask or self.with_seg:
            # The tasks completed by '_load_masks' and '_load_semantic_segs'
            # in LoadAnnotations are merged to one function.
            results = self._load_masks_and_semantic_segs(results)

        return results


@PIPELINES.register_module()
class LoadProposals:
    """Load proposal pipeline.

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(num_max_proposals={self.num_max_proposals})'


@PIPELINES.register_module()
class FilterAnnotations:
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Default: True        
        min_visibility (int): Minimum visibility value.
    """

    def __init__(self, min_gt_bbox_wh=None, keep_empty=True, min_visibility=None):
        # TODO: add more filter options
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.keep_empty = keep_empty
        self.min_visibility = min_visibility

    def __call__(self, results):
        if self.min_gt_bbox_wh is not None:
            assert 'gt_bboxes' in results
            gt_bboxes = results['gt_bboxes']
            if gt_bboxes.shape[0] == 0:
                return results
            w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
            if not keep.any():
                if self.keep_empty:
                    return None
                else:
                    return results
            else:
                keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
                for key in keys:
                    if key in results:
                        results[key] = results[key][keep]
                return results
        if self.min_visibility is not None:
            assert 'gt_visibilities' in results
            keep = (results['gt_visibilities'] >= self.min_visibility)
            keys = ('gt_bboxes', 'gt_labels', 'gt_visibilities')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results            

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(min_gt_bbox_wh={self.min_gt_bbox_wh},' \
               f'always_keep={self.always_keep})'\
               f'min_visibility={self.min_visibility})'