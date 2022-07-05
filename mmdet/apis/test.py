# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time
import copy

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results, bbox2result


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    first_run_dir=None,
                    epoch_nr=None,
                    undo_gt_crop=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # if i % 100 == 0:
        if first_run_dir:
            stored_settings = (show, out_dir, show_score_thr)
            out_dir = first_run_dir + "/debug_val_images/"
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            undo_gt_crop=False
            show_gt_bbox_results = False
            post_fix = '_former'
            import numpy as np
            reduzed_classes=False
            crop_to_horizon = False
            offset_w=50#0#394,
            offset_h=300#204#296,
            crop_size=(432, 1500)#(492, 1599)
            cat_ids_del = []#[2,4,8,9]#[0, 1, 3, 5, 6, 7]
            
            # Get ground trouth results
            gt_bbox_results = None
            gt_ann_info = dataset.get_ann_info(i)
            gt_bboxes = copy.deepcopy(gt_ann_info['bboxes'])
            gt_labels = copy.deepcopy(gt_ann_info['labels'])

            if undo_gt_crop:
                import numpy as np
                # set parameters for cropping of gt
                offset_w = 296
                offset_h = 394
                img_shape = (384, 1248)
                thresh_in_frame = 0.1

                pre_width = gt_bboxes[:, 2] - gt_bboxes[:, 0]
                pre_height = gt_bboxes[:, 3] - gt_bboxes[:, 1]
                pre_area = pre_width * pre_height

                # Crop bbox
                bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                    dtype=np.float32)
                bboxes = gt_bboxes - bbox_offset
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1]-1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0]-1)

                # calculate the percentage of the cropped bbox compared to the original one
                post_width = bboxes[:, 2] - bboxes[:, 0]
                post_height = bboxes[:, 3] - bboxes[:, 1]
                post_area = post_width * post_height
                in_frame = post_area / pre_area

                # Filter Boxes
                valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                    bboxes[:, 3] > bboxes[:, 1]) & (
                    in_frame >= thresh_in_frame)
                gt_bboxes = bboxes[valid_inds]
                gt_labels = gt_labels[valid_inds]

            gt_bbox_results = [bbox2result(gt_bboxes, gt_labels, len(dataset.CLASSES))]
            
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                if (img_meta['ori_shape'] != img_meta['img_shape']) and (img_meta['scale_factor'] == 1).all():
                    print('scipping img resize, this is only wanted if a crop, but no resize was done')
                else:
                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                    out_file_gt = out_file
                    if first_run_dir:
                        out_file = out_file.split('.')
                        out_file[-2] = out_file[-2] + '_e' + str(epoch_nr)
                        out_file = '.'.join(out_file)
                else:
                    out_file = None

                if crop_to_horizon:
                    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
                    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
                    bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h, 0],
                                        dtype=np.float32)

                    # crop the image
                    img_show = img_show[crop_y1:crop_y2, crop_x1:crop_x2, ...]

                    # crop the result box
                    for j in range(len(result[i])):
                        if result[i][j].size > 0:
                            bboxes = result[i][j] - bbox_offset
                            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, crop_size[1])
                            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, crop_size[0])  
                            result[i][j] = bboxes
                    # crop the gt box
                    for j in range(len(gt_bbox_results[0])):
                        if gt_bbox_results[0][j].size > 0:
                            bboxes = gt_bbox_results[0][j] - bbox_offset[:4]
                            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, crop_size[1])
                            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, crop_size[0])  
                            gt_bbox_results[0][j] = bboxes
                if reduzed_classes:
                    for cat_id in cat_ids_del:
                        result[i][cat_id] = np.empty(shape=(0,5))
                if post_fix:
                    out_file = out_file.split('.')
                    out_file[-2] += post_fix
                    out_file = '.'.join(out_file)
                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)
                if show_gt_bbox_results:
                    # Alter ouput file name
                    out_file_gt = out_file_gt.split('.')
                    out_file_gt[-2] += '_gt'
                    out_file_gt = '.'.join(out_file_gt)
                    # Show ground trouth results
                    if reduzed_classes:
                        for cat_id in cat_ids_del:
                            if gt_bbox_results[0][cat_id].size > 0:
                                gt_bbox_results[0][cat_id] = np.empty(shape=(0,4))
                    model.module.show_result(
                        img_show,
                        gt_bbox_results[0],
                        show=show,
                        out_file=out_file_gt,
                        score_thr=show_score_thr,
                        disp_gt=True)
            if first_run_dir:
                (show, out_dir, show_score_thr) = stored_settings
                first_run_dir = False

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
