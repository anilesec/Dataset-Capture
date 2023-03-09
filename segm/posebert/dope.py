# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import nms
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

from torchvision.models import resnet
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_keypoints, resize_boxes

import cv2
import numpy as np

parts = ['body', 'hand', 'face']
num_joints = {'body': 13, 'hand': 21, 'face': 84}


class Dope_Transform(GeneralizedRCNNTransform):

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(self.__class__, self).__init__(min_size, max_size, image_mean, image_std)

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            for k in ['pose2d', 'body_pose2d', 'hand_pose2d', 'face_pose2d']:
                if k in pred and pred[k] is not None:
                    pose2d = pred[k]
                    pose2d = resize_keypoints(pose2d, im_s, o_im_s)
                    result[i][k] = pose2d
        return result


class Dope_RCNN(GeneralizedRCNN):

    def __init__(self, backbone,
                 dope_roi_pool, dope_head, dope_predictor,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # others
                 num_anchor_poses={'body': 20, 'hand': 10, 'face': 10},
                 pose2d_reg_weights={part: 5.0 for part in parts},
                 pose3d_reg_weights={part: 5.0 for part in parts},
                 ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(dope_roi_pool, (MultiScaleRoIAlign, type(None)))

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        dope_heads = Dope_RoIHeads(dope_roi_pool, dope_head, dope_predictor, num_anchor_poses,
                                   pose2d_reg_weights=pose2d_reg_weights, pose3d_reg_weights=pose3d_reg_weights)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = Dope_Transform(min_size, max_size, image_mean, image_std)

        super(Dope_RCNN, self).__init__(backbone, rpn, dope_heads, transform)


class Dope_Predictor(nn.Module):

    def __init__(self, in_channels, dict_num_classes, dict_num_posereg):
        super(self.__class__, self).__init__()
        self.body_cls_score = nn.Linear(in_channels, dict_num_classes['body'])
        self.body_pose_pred = nn.Linear(in_channels, dict_num_posereg['body'])
        self.hand_cls_score = nn.Linear(in_channels, dict_num_classes['hand'])
        self.hand_pose_pred = nn.Linear(in_channels, dict_num_posereg['hand'])
        self.face_cls_score = nn.Linear(in_channels, dict_num_classes['face'])
        self.face_pose_pred = nn.Linear(in_channels, dict_num_posereg['face'])

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = {}
        pose_deltas = {}
        scores['body'] = self.body_cls_score(x)
        pose_deltas['body'] = self.body_pose_pred(x)
        scores['hand'] = self.hand_cls_score(x)
        pose_deltas['hand'] = self.hand_pose_pred(x)
        scores['face'] = self.face_cls_score(x)
        pose_deltas['face'] = self.face_pose_pred(x)
        return scores, pose_deltas


class Dope_RoIHeads(RoIHeads):

    def __init__(self,
                 dope_roi_pool,
                 dope_head,
                 dope_predictor,
                 num_anchor_poses,
                 pose2d_reg_weights,
                 pose3d_reg_weights):

        fg_iou_thresh = 0.5
        bg_iou_thresh = 0.5
        batch_size_per_image = 512
        positive_fraction = 0.25
        bbox_reg_weights = [0.0] * 4
        score_thresh = 0.0
        nms_thresh = 1.0
        detections_per_img = 99999999
        super(self.__class__, self).__init__(None, None, None, fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
                                             positive_fraction, bbox_reg_weights, score_thresh, nms_thresh,
                                             detections_per_img, mask_roi_pool=None, mask_head=None,
                                             mask_predictor=None, keypoint_roi_pool=None, keypoint_head=None,
                                             keypoint_predictor=None)
        for k in parts:
            self.register_buffer(k + '_anchor_poses', torch.empty((num_anchor_poses[k], num_joints[k], 5)))
        self.dope_roi_pool = dope_roi_pool
        self.dope_head = dope_head
        self.dope_predictor = dope_predictor
        self.J = num_joints
        self.pose2d_reg_weights = pose2d_reg_weights
        self.pose3d_reg_weights = pose3d_reg_weights

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[torch.Tensor])
            proposals (List[torch.Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        # roi_pool
        if features['0'].dtype == torch.float16:  # UGLY: dope_roi_pool is not yet compatible with half
            features = {'0': features['0'].float()}
            if proposals[0].dtype == torch.float16:
                hproposals = [p.float() for p in proposals]
            else:
                hproposals = proposals
            dope_features = self.dope_roi_pool(features, hproposals, image_shapes)
            dope_features = dope_features.half()
        else:
            dope_features = self.dope_roi_pool(features, proposals, image_shapes)

        # head
        dope_features = self.dope_head(dope_features)

        # predictor
        class_logits, dope_regression = self.dope_predictor(dope_features)

        # process results
        result = []
        losses = {}
        if self.training:
            raise NotImplementedError
        else:
            boxes, scores, poses2d, poses3d = self.postprocess_dope(class_logits, dope_regression, proposals,
                                                                    image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                res = {'boxes': boxes[i]}
                for k in parts:
                    res[k + '_scores'] = scores[k][i]
                    res[k + '_pose2d'] = poses2d[k][i]
                    res[k + '_pose3d'] = poses3d[k][i]
                result.append(res)

        return result, losses

    def postprocess_dope(self, class_logits, dope_regression, proposals, image_shapes):
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        num_images = len(proposals)
        pred_scores = {}
        all_poses_2d = {}
        all_poses_3d = {}
        for k in parts:
            # anchor poses
            anchor_poses = getattr(self, k + '_anchor_poses')
            nboxes, num_classes = class_logits[k].size()
            # scores
            sc = F.softmax(class_logits[k], -1)
            pred_scores[k] = sc.split(boxes_per_image, 0)
            # poses
            all_poses_2d[k] = []
            all_poses_3d[k] = []
            dope_regression[k] = dope_regression[k].view(nboxes, num_classes - 1, self.J[k] * 5)
            dope_regression_per_image = dope_regression[k].split(boxes_per_image, 0)
            for img_id in range(num_images):
                dope_reg = dope_regression_per_image[img_id]
                boxes = proposals[img_id]
                # 2d
                offset = boxes[:, 0:2]
                scale = boxes[:, 2:4] - boxes[:, 0:2]
                box_resized_anchors = offset[:, None, None, :] + anchor_poses[None, :, :, :2] * scale[:, None, None, :]
                dope_reg_2d = dope_reg[:, :, :2 * self.J[k]].reshape(boxes.size(0), num_classes - 1, self.J[k], 2) / \
                              self.pose2d_reg_weights[k]
                pose2d = box_resized_anchors + dope_reg_2d * scale[:, None, None, :]
                all_poses_2d[k].append(pose2d)
                # 3d
                anchor3d = anchor_poses[None, :, :, -3:]
                dope_reg_3d = dope_reg[:, :, -3 * self.J[k]:].reshape(boxes.size(0), num_classes - 1, self.J[k], 3) / \
                              self.pose3d_reg_weights[k]
                pose3d = anchor3d + dope_reg_3d
                all_poses_3d[k].append(pose3d)
        return proposals, pred_scores, all_poses_2d, all_poses_3d


def dope_resnet50(**dope_kwargs):
    backbone_name = 'resnet50'
    from torchvision.ops import misc as misc_nn_ops
    class FrozenBatchNorm2dWithHalf(misc_nn_ops.FrozenBatchNorm2d):
        def forward(self, x):
            if x.dtype == torch.float16:  # UGLY: seems that it does not work with half otherwise, so let's just use the standard bn function or half
                return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False)
            else:
                return super(self.__class__, self).forward(x)

    backbone = resnet.__dict__[backbone_name](pretrained=False, norm_layer=FrozenBatchNorm2dWithHalf)

    # build the main blocks
    class ResNetBody(nn.Module):
        def __init__(self, backbone):
            super(self.__class__, self).__init__()
            self.resnet_backbone = backbone
            self.out_channels = 1024

        def forward(self, x):
            x = self.resnet_backbone.conv1(x)
            x = self.resnet_backbone.bn1(x)
            x = self.resnet_backbone.relu(x)
            x = self.resnet_backbone.maxpool(x)
            x = self.resnet_backbone.layer1(x)
            x = self.resnet_backbone.layer2(x)
            x = self.resnet_backbone.layer3(x)
            return x

    resnet_body = ResNetBody(backbone)
    # build the anchor generator and pooler
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    # build the head and predictor
    class ResNetHead(nn.Module):
        def __init__(self, backbone):
            super(self.__class__, self).__init__()
            self.resnet_backbone = backbone

        def forward(self, x):
            x = self.resnet_backbone.layer4(x)
            x = self.resnet_backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x

    resnet_head = ResNetHead(backbone)

    # predictor
    num_anchor_poses = dope_kwargs['num_anchor_poses']
    num_classes = {k: v + 1 for k, v in num_anchor_poses.items()}
    num_posereg = {k: num_anchor_poses[k] * num_joints[k] * 5 for k in num_joints.keys()}
    predictor = Dope_Predictor(2048, num_classes, num_posereg)

    # full model
    model = Dope_RCNN(resnet_body, roi_pooler, resnet_head, predictor, rpn_anchor_generator=anchor_generator,
                      **dope_kwargs)

    return model


def _boxes_from_poses(poses, margin=0.1):  # pytorch version
    x1y1, _ = torch.min(poses, dim=1)  # N x 2
    x2y2, _ = torch.max(poses, dim=1)  # N x 2
    coords = torch.cat((x1y1, x2y2), dim=1)
    sizes = x2y2 - x1y1
    coords[:, 0:2] -= margin * sizes
    coords[:, 2:4] += margin * sizes
    return coords


def DOPE_NMS(scores, boxes, pose2d, pose3d, min_score=0.5, iou_threshold=0.1):
    if scores.numel() == 0:
        return torch.LongTensor([]), torch.LongTensor([])
    maxscores, bestcls = torch.max(scores[:, 1:], dim=1)
    valid_indices = torch.nonzero(maxscores >= min_score)
    if valid_indices.numel() == 0:
        return torch.LongTensor([]), torch.LongTensor([])
    else:
        valid_indices = valid_indices[:, 0]

    boxes = _boxes_from_poses(pose2d[valid_indices, bestcls[valid_indices], :, :], margin=0.1)
    indices = valid_indices[nms(boxes, maxscores[valid_indices, ...], iou_threshold)]
    bestcls = bestcls[indices]

    return {'score': scores[indices, bestcls + 1], 'pose2d': pose2d[indices, bestcls, :, :],
            'pose3d': pose3d[indices, bestcls, :, :]}, indices, bestcls


def _get_bbox_from_points(points2d, margin):
    """
    Compute a bounding box around 2D keypoints, with a margin.
    margin: the margin is relative to the size of the tight bounding box
    """
    assert (len(points2d.shape) == 2 and points2d.shape[1] == 2)
    mini = np.min(points2d, axis=0)
    maxi = np.max(points2d, axis=0)
    size = maxi - mini
    lower = mini - margin * size
    upper = maxi + margin * size
    box = np.concatenate((lower, upper)).astype(np.float32)
    return box


def assign_hands_to_body(body_poses, hand_poses, hand_isright, margin=1):
    if body_poses.size == 0: return []
    if hand_poses.size == 0: return [(-1, -1) for i in range(body_poses.shape[0])]
    from scipy.spatial.distance import cdist
    body_rwrist = body_poses[:, 6, :]
    body_lwrist = body_poses[:, 7, :]
    hand_wrist = hand_poses[:, 0, :]
    hand_boxes = np.concatenate(
        [_get_bbox_from_points(hand_poses[i, :, :], margin=0.1)[None, :] for i in range(hand_poses.shape[0])], axis=0)
    hand_size = np.max(hand_boxes[:, 2:4] - hand_boxes[:, 0:2], axis=1)
    # associate body and hand if the distance hand-body and body-hand is the smallest one and is this distance is smaller than 3*hand_size
    wrists_from_body = [(-1, -1) for i in range(body_poses.shape[0])]  # pair of (left_hand_id, right_hand_id)
    dist_lwrist = cdist(body_lwrist, hand_wrist)
    dist_rwrist = cdist(body_rwrist, hand_wrist)
    for i in range(body_poses.shape[0]):
        lwrist = -1
        rwrist = -1
        if hand_wrist.size > 0:
            best_lwrist = np.argmin(dist_lwrist[i, :])
            if np.argmin(dist_lwrist[:, best_lwrist]) == i and dist_lwrist[i, best_lwrist] <= margin * hand_size[
                best_lwrist]:
                lwrist = best_lwrist
            best_rwrist = np.argmin(dist_rwrist[i, :])
            if np.argmin(dist_rwrist[:, best_rwrist]) == i and dist_rwrist[i, best_rwrist] <= margin * hand_size[
                best_rwrist]:
                rwrist = best_rwrist
        wrists_from_body[i] = (lwrist, rwrist)
    return wrists_from_body  # pair of (left_hand_id, right_hand_id) for each body pose (-1 means no association)


def assign_head_to_body(body_poses, head_poses):
    if body_poses.size == 0: return []
    if head_poses.size == 0: return [-1 for i in range(body_poses.shape[0])]
    head_boxes = np.concatenate(
        [_get_bbox_from_points(head_poses[i, :, :], margin=0.1)[None, :] for i in range(head_poses.shape[0])], axis=0)
    body_heads = body_poses[:, 12, :]
    bodyhead_in_headboxes = np.empty((body_poses.shape[0], head_boxes.shape[0]), dtype=np.bool)
    for i in range(body_poses.shape[0]):
        bodyhead = body_heads[i, :]
        bodyhead_in_headboxes[i, :] = (bodyhead[0] >= head_boxes[:, 0]) * (bodyhead[0] <= head_boxes[:, 2]) * (
                bodyhead[1] >= head_boxes[:, 1]) * (bodyhead[1] <= head_boxes[:, 3])
    head_for_body = []
    for i in range(body_poses.shape[0]):
        if np.sum(bodyhead_in_headboxes[i, :]) == 1:
            j = np.where(bodyhead_in_headboxes[i, :])[0][0]
            if np.sum(bodyhead_in_headboxes[:, j]) == 1:
                head_for_body.append(j)
            else:
                head_for_body.append(-1)
        else:
            head_for_body.append(-1)
    return head_for_body


def assign_hands_and_head_to_body(detections):
    det_poses2d = {
        part: np.stack([d['pose2d'] for d in part_detections], axis=0) if len(part_detections) > 0 else np.empty(
            (0, 0, 2), dtype=np.float32) for part, part_detections in detections.items()}
    hand_isright = np.array([d['hand_isright'] for d in detections['hand']])
    body_with_wrists = assign_hands_to_body(det_poses2d['body'], det_poses2d['hand'], hand_isright, margin=1)
    BODY_RIGHT_WRIST_KPT_ID = 6
    BODY_LEFT_WRIST_KPT_ID = 7
    for i, (lwrist, rwrist) in enumerate(body_with_wrists):
        if lwrist != -1: detections['body'][i]['pose2d'][BODY_LEFT_WRIST_KPT_ID, :] = detections['hand'][lwrist][
                                                                                          'pose2d'][0, :]
        if rwrist != -1: detections['body'][i]['pose2d'][BODY_RIGHT_WRIST_KPT_ID, :] = detections['hand'][rwrist][
                                                                                           'pose2d'][0, :]
    body_with_head = assign_head_to_body(det_poses2d['body'], det_poses2d['face'])
    return detections, body_with_wrists, body_with_head


def area2d(b):
    """ compute the areas for a set of 2D boxes"""
    return (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)


def overlap2d(b1, b2):
    """ compute the overlaps between a set of boxes b1 and 1 box b2 """
    xmin = np.maximum(b1[:, 0], b2[:, 0])
    xmax = np.minimum(b1[:, 2] + 1, b2[:, 2] + 1)
    width = np.maximum(0, xmax - xmin)
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    ymax = np.minimum(b1[:, 3] + 1, b2[:, 3] + 1)
    height = np.maximum(0, ymax - ymin)
    return width * height


def iou2d(b1, b2):
    """ compute the IoU between a set of boxes b1 and 1 box b2"""
    if b1.ndim == 1: b1 = b1[None, :]
    if b2.ndim == 1: b2 = b2[None, :]
    assert b2.shape[0] == 1
    o = overlap2d(b1, b2)
    return o / (area2d(b1) + area2d(b2) - o)


def convert_to_ppi_format(scores, boxes, pose2d, pose3d, score_th=None, tonumpy=True):
    if score_th is None: score_th = 0.1 / (scores.shape[1] - 1)
    boxindices, idxlabels = np.where(scores[:, 1:] >= score_th)  # take the Ndet values with scores over a threshold
    s_scores = scores[boxindices, 1 + idxlabels]  # Ndet (number of estimations with score over score_th)
    s_boxes = boxes[boxindices, :]  # Ndetx4 (does not depend on label)
    s_pose2d = pose2d[boxindices, idxlabels, :, :]  # NdetxJx2
    s_pose3d = pose3d[boxindices, idxlabels, :, :]  # NdetxJx3

    return s_scores, s_boxes, s_pose2d, s_pose3d, boxindices, idxlabels


indTorsoHead = {13: [4, 5, 10, 11, 12], 14: [4, 5, 10, 11, 12, 13]}
indLower = {13: range(0, 4), 14: range(0, 4)}  # indices of joints for lower body
indUpper = {13: range(4, 13), 14: range(4, 14)}  # indices of joints for upper body


def LCRNet_PPI_improved(scores, boxes, pose2d, pose3d, resolution, K=None, score_th=None, th_pose3D=0.3, th_iou=0.2,
                        iou_func='bbox', min_mode_score=0.05, th_persondetect=0.002, verbose=False):
    """
    this function extends the Pose Proposals Integration (PPI) from LCR-Net in order to also handle hands (J=21), faces (J=84) and bodies with 13 or 14 joints.
    scores, boxes, pose2d, pose3d are numpy arrays of size [Nboxes x (Nclasses+1)], [Nboxes x 4], [Nboxes x Nclasses x J x 2] and [Nboxes x Nclasses x J x 3] resp.
    resolution: tuple (height,width) containing image resolution
    K: number of classes, without considering lower/upper (K=10)
    score_th: only apply ppi on pose proposals with score>=score_th (None => 0.1/K)
    th_pose3D: threshold for groupind pose in a mode, based on mean dist3D of the joints
    th_iou: 2D overlap threshold to group the pose proposals
    iou_func: how 2d overlap is defined
    min_mode_score: we consider modes with at least this score (save time and do not affect 1st mode)
    th_persondetect: we remove final detection whose cumscore is below this threshold
    return a list of detection with the following fields:
      * score: cumulative score of the detection
      * pose2d: Jx2 numpy array
      * pose3d: Jx3 numpy array
    """

    if K is None: K = scores.shape[1]
    s_scores, s_boxes, s_pose2d, s_pose3d, boxindices, idxlabels = convert_to_ppi_format(scores, boxes, pose2d, pose3d,
                                                                                         score_th=score_th)

    H, W = resolution
    wh = np.array([[W, H]], dtype=np.float32)
    J = s_pose2d.shape[1]
    if not J in [13, 14]: assert iou_func in ['bbox', 'mje']

    # compute bounding boxes from 2D poses truncated by images boundaries
    if iou_func == 'bbox_torsohead':  # using torso+head keypoints only
        xymin = np.minimum(wh - 1, np.maximum(0, np.min(s_pose2d[:, indTorsoHead[J], :], axis=1)))
        xymax = np.minimum(wh - 1, np.maximum(0, np.max(s_pose2d[:, indTorsoHead[J], :], axis=1)))
        bbox_headtorso = np.concatenate((xymin, xymax), axis=1)
    else:  # using all keypoints
        xymin = np.minimum(wh - 1, np.maximum(0, np.min(s_pose2d, axis=1)))
        xymax = np.minimum(wh - 1, np.maximum(0, np.max(s_pose2d, axis=1)))
        bboxes = np.concatenate((xymin, xymax), axis=1)

    # define iou metrics
    def compute_overlapping_poses(bboxes, poses2d, a_bbox, a_pose2d, th_iou):
        assert a_pose2d.ndim == 2 and a_bbox.ndim == 1
        a_bbox = a_bbox[None, :]
        a_pose2d = a_pose2d[None, :, :]
        assert bboxes.ndim == 2 and poses2d.ndim == 3
        if iou_func == 'bbox' or iou_func == 'bbox_torsohead':
            iou = iou2d(bboxes, a_bbox)
            return np.where(iou > th_iou)[0]
        elif iou_func == 'torso' or iou_func == 'torsoLR':
            indices = [4, 5, 10, 11]
            lr_indices = [5, 4, 11, 10]
        elif iou_func == 'torsohead' or iou_func == 'torsoheadLR':
            indices = [4, 5, 10, 11, 12] if J == 13 else [4, 5, 10, 11, 12, 13]
            lr_indices = [5, 4, 11, 10, 12] if J == 13 else [5, 4, 11, 10, 12, 13]
        elif iou_func == 'head':
            indices = [12] if J == 13 else [12, 13]
        elif iou_func == 'shoulderhead' or iou_func == 'shoulderheadLR':
            indices = [10, 11, 12] if J == 13 else [10, 11, 12, 13]
            lr_indices = [11, 10, 12] if J == 13 else [11, 10, 12, 13]
        elif iou_func == 'mje':
            indices = list(range(J))
        else:
            raise NotImplementedError('ppi.py: unknown iou_func')
        indices = np.array(indices, dtype=np.int32)
        if iou_func.endswith('LR'):
            lr_indices = np.array(lr_indices, dtype=np.int32)
            a = np.minimum(
                np.mean(np.sqrt(np.sum((poses2d[:, indices, :] - a_pose2d[:, indices, :]) ** 2, axis=2)), axis=1),
                np.mean(np.sqrt(np.sum((poses2d[:, lr_indices, :] - a_pose2d[:, indices, :]) ** 2, axis=2)), axis=1))
        else:
            a = np.mean(np.sqrt(np.sum((poses2d[:, indices, :] - a_pose2d[:, indices, :]) ** 2, axis=2)), axis=1)
        b = 2 * np.max(a_bbox[:, 2:4] - a_bbox[:, 0:2] + 1)
        return np.where(a / b < th_iou)[0]

    # group pose proposals according to 2D IoU
    Persons = []  # list containing the detected people, each person being a tuple ()
    remaining_pp = range(s_pose2d.shape[0])
    while len(remaining_pp) > 0:
        # take highest remaining score
        imax = np.argmax(s_scores[remaining_pp])
        # consider the pose proposals with high 2d overlap
        this = compute_overlapping_poses(s_boxes[remaining_pp, :], s_pose2d[remaining_pp, :],
                                         s_boxes[remaining_pp[imax], :], s_pose2d[remaining_pp[imax], :, :], th_iou)
        this_pp = np.array(remaining_pp, dtype=np.int32)[this]
        # add the person and delete the corresponding pp
        Persons.append((this_pp, np.sum(s_scores[this_pp])))
        remaining_pp = [p for p in remaining_pp if not p in this_pp]
    if verbose: print("{:d} persons/groups of poses found".format(len(Persons)))

    Detected = []
    # find modes for each person
    for iperson, (pplist, cumscore) in enumerate(Persons):

        remaining_pp = list(pplist.copy())  # create a copy, list of pp that are not assigned to any mode
        Modes = []

        while len(remaining_pp) > 0:

            # next anchor pose mode is defined as the top regscore among unassigned poses
            imax = np.argmax(s_scores[remaining_pp])
            maxscore = s_scores[remaining_pp[imax]]
            if maxscore < min_mode_score and len(
                    Modes) > 0: break  # stop if score not sufficiently high and already created a mode

            # select PP (from the entire set) close to the center of the mode
            mode_pose3D = s_pose3d[remaining_pp[imax], :, :]
            # dist3D = np.mean( np.sqrt( (mode_pose3D[ 0:13]-regpose3d[pplist, 0:13])**2 + \
            #                           (mode_pose3D[13:26]-regpose3d[pplist,13:26])**2 + \
            #                           (mode_pose3D[26:39]-regpose3d[pplist,26:39])**2 ), axis=1)
            dist3D = np.mean(np.sqrt(np.sum((mode_pose3D - s_pose3d[pplist, :, :]) ** 2, axis=2)), axis=1)
            this = np.where(dist3D < th_pose3D)[0]

            # compute the output for this mode
            this_pp = pplist[this]
            weights = s_scores[this_pp]

            # upper body is average weights by the scores
            hand_isright = None
            if J in [13, 14]:
                p3d = np.empty((J, 3), dtype=np.float32)
                p2d = np.empty((J, 2), dtype=np.float32)
                cumscore = np.sum(weights)
                p3d[indUpper[J], :] = np.sum(weights[:, None, None] * s_pose3d[this_pp, :, :][:, indUpper[J], :],
                                             axis=0) / cumscore
                p2d[indUpper[J], :] = np.sum(weights[:, None, None] * s_pose2d[this_pp, :, :][:, indUpper[J], :],
                                             axis=0) / cumscore

                assert idxlabels is not None
                # for lower body, we downweight upperbody scores
                this_ub = np.where(idxlabels[this_pp] > K)[0]  # anchor pose for upper body
                weights[this_ub] *= 0.1
                cumscoreBot = np.sum(weights)
                p3d[indLower[J], :] = np.sum(weights[:, None, None] * s_pose3d[this_pp, :, :][:, indLower[J], :],
                                             axis=0) / cumscoreBot
                p2d[indLower[J], :] = np.sum(weights[:, None, None] * s_pose2d[this_pp, :, :][:, indLower[J], :],
                                             axis=0) / cumscoreBot
            else:
                cumscore = np.sum(weights)
                p3d = np.sum(weights[:, None, None] * s_pose3d[this_pp, :, :], axis=0) / cumscore
                p2d = np.sum(weights[:, None, None] * s_pose2d[this_pp, :, :], axis=0) / cumscore
                if J == 21:
                    hand_isright = (idxlabels[imax] < K)

            this_mode = {'score': cumscore, 'pose3d': p3d, 'pose2d': p2d}
            if hand_isright is not None: this_mode['hand_isright'] = hand_isright
            Modes.append(this_mode)

            # remove pp from the list to process
            remaining_pp = [p for p in remaining_pp if not p in this_pp]
        if verbose: print("Person {:d}/{:d} has {:d} mode(s)".format(iperson + 1, len(Persons), len(Modes)))

        # keep the main mode for each person, only if score is sufficient high
        modes_score = np.array([m['score'] for m in Modes])
        bestmode = np.argmax(modes_score)
        if modes_score[bestmode] > th_persondetect:
            Detected.append(Modes[bestmode])
        else:
            if verbose: print("\tdeleting this person because of too low score")
    if verbose: print('{:d} person(s) detected'.format(len(Detected)))
    # sort detection according to score
    Detected.sort(key=lambda d: d['score'], reverse=True)
    return Detected


def _get_bones_and_colors(J, ignore_neck=False):  # colors in BGR
    """
    param J: number of joints -- used to deduce the body part considered.
    param ignore_neck: if True, the neck bone of won't be returned in case of a body (J==13)
    """
    if J == 13:  # full body (similar to LCR-Net)
        lbones = [(9, 11), (7, 9), (1, 3), (3, 5)]
        if ignore_neck:
            rbones = [(0, 2), (2, 4), (8, 10), (6, 8)] + [(4, 5), (10, 11)] + [([4, 5], [10, 11])]
        else:
            rbones = [(0, 2), (2, 4), (8, 10), (6, 8)] + [(4, 5), (10, 11)] + [([4, 5], [10, 11]), (12, [10, 11])]
        bonecolors = [[0, 255, 0]] * len(lbones) + [[255, 0, 0]] * len(rbones)
        pltcolors = ['g-'] * len(lbones) + ['b-'] * len(rbones)
        bones = lbones + rbones
    elif J == 21:  # hand (format similar to HO3D dataset)
        bones = [[(0, n + 1), (n + 1, 3 * n + 6), (3 * n + 6, 3 * n + 7), (3 * n + 7, 3 * n + 8)] for n in range(5)]
        bones = sum(bones, [])
        bonecolors = [(255, 0, 255)] * 4 + [(255, 0, 0)] * 4 + [(0, 255, 0)] * 4 + [(0, 255, 255)] * 4 + [
            (0, 0, 255)] * 4
        pltcolors = ['m'] * 4 + ['b'] * 4 + ['g'] * 4 + ['y'] * 4 + ['r'] * 4
    elif J == 84:  # face (ibug format)
        bones = [(n, n + 1) for n in range(83) if n not in [32, 37, 42, 46, 51, 57, 63, 75]] + [(52, 57), (58, 63),
                                                                                                (64, 75), (76, 83)]
        # 32 x contour + 4 x r-sourcil +  4 x l-sourcil + 7 x nose + 5 x l-eye + 5 x r-eye +20 x lip + l-eye + r-eye + lip + lip
        bonecolors = 32 * [(255, 0, 0)] + 4 * [(255, 0, 0)] + 4 * [(255, 255, 0)] + 7 * [(255, 0, 255)] + 5 * [
            (0, 255, 255)] + 5 * [(0, 255, 0)] + 18 * [(0, 0, 255)] + [(0, 255, 255), (0, 255, 0), (0, 0, 255),
                                                                       (0, 0, 255)]
        pltcolors = 32 * ['b'] + 4 * ['b'] + 4 * ['c'] + 7 * ['m'] + 5 * ['y'] + 5 * ['g'] + 18 * ['r'] + ['y', 'g',
                                                                                                           'r', 'r']
    else:
        raise NotImplementedError('unknown bones/colors for J=' + str(J))
    return bones, bonecolors, pltcolors


def _get_xy(pose2d, i):
    if isinstance(i, int):
        return pose2d[i, :]
    else:
        return np.mean(pose2d[i, :], axis=0)


def _get_xy_tupleint(pose2d, i):
    return tuple(map(int, _get_xy(pose2d, i)))


def _get_xyz(pose3d, i):
    if isinstance(i, int):
        return pose3d[i, :]
    else:
        return np.mean(pose3d[i, :], axis=0)