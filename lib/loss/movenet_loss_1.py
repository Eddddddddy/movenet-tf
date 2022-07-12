"""
@Fire
https://github.com/fire717
"""
import sys

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss

import cv2

_img_size = 192
_feature_map_size = _img_size // 4

_center_weight_path = 'lib/data/center_weight_origin.npy'


class JointBoneLoss():
    def __init__(self, joint_num):
        super(JointBoneLoss, self).__init__()
        id_i, id_j = [], []
        for i in range(joint_num):
            for j in range(i + 1, joint_num):
                id_i.append(i)
                id_j.append(j)
        self.id_i = id_i
        self.id_j = id_j

        # self.id_i = [0,1,2,3,4,5,2]
        # self.id_j = [1,2,3,4,5,6,4]

    def forward(self, joint_out, joint_gt):
        J = tf.norm(joint_out[:, self.id_i, :] - joint_out[:, self.id_j, :], p=2, dim=-1, keepdim=False)
        Y = tf.norm(joint_gt[:, self.id_i, :] - joint_gt[:, self.id_j, :], p=2, dim=-1, keepdim=False)
        loss = tf.abs(J - Y)
        # loss = loss.mean()
        loss = tf.reduce_sum(loss) / joint_out.shape[0] / len(self.id_i)
        return loss


class MovenetLoss(Loss):
    def __init__(self, use_target_weight=False, target_weight=[1], cfg=None):
        super(MovenetLoss, self).__init__()
        # self.mse = torch.nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight
        self.target_weight = target_weight
        self.cfg = cfg

        self.center_weight = tf.convert_to_tensor(np.load(_center_weight_path))
        self.make_center_w = False

        # self.range_weight_x = torch.from_numpy(np.array([[x for x in range(48)] for _ in range(48)]))
        # self.range_weight_y = self.range_weight_x.T 

        self.boneloss = JointBoneLoss(17)

    def l1(self, pre, target, kps_mask):
        # print("1 ",pre.shape, pre.device)
        # print("2 ",target.shape, target.device)
        # b

        # return torch.mean(torch.abs(pre - target)*kps_mask)

        # print("pre ", pre.shape)
        # print("target ", target.shape)
        # print("kps_mask ", kps_mask.shape)

        # pre = tf.cast(pre, tf.float64)
        # target = tf.cast(target, tf.float64)

        c = tf.abs(pre - target)
        a = tf.reduce_sum(c * kps_mask)
        b = (tf.reduce_sum(kps_mask) + 1e-4)
        # return tf.reduce_sum(tf.abs(pre - target) * kps_mask) / (kps_mask.sum() + 1e-4)
        return a / b

    def l2_loss(self, pre, target):
        loss = (pre - target)
        loss = (loss * loss) / 2 / pre.shape[0]

        return loss.sum()

    def centernetfocalLoss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        Arguments:
          pred (batch x c x h x w)
          gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = tf.pow(1 - gt, 4)

        loss = 0

        pos_loss = tf.log(pred) * tf.pow(1 - pred, 2) * pos_inds
        neg_loss = tf.log(1 - pred) * tf.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def myMSEwithWeight(self, pre, target):
        # target 0-1
        # pre = torch.sigmoid(pre)
        # print(torch.max(pre), torch.min(pre))
        # b
        loss = tf.pow((pre - target), 2)
        # loss = torch.abs(pre-target)

        # weight_mask = (target+0.1)/1.1
        weight_mask = target * 8 + 1
        # weight_mask = torch.pow(target,2)*8+1

        # gamma from focal loss
        # gamma = torch.pow(torch.abs(target-pre), 2)

        loss = loss * weight_mask  # *gamma

        loss = tf.reduce_sum(loss) / 128 / target.shape[3]

        # bg_loss = self.bgLoss(pre, target)
        return loss

    def heatmapL1(self, pre, target):
        # target 0-1
        # pre = torch.sigmoid(pre)
        # print(torch.max(pre), torch.min(pre))
        # b
        loss = tf.abs(pre - target)

        # weight_mask = (target+0.1)/1.1
        weight_mask = target * 4 + 1

        # gamma from focal loss
        # gamma = torch.pow(torch.abs(target-pre), 2)

        loss = loss * weight_mask  # *gamma

        loss = tf.reduce_sum(loss) / target.shape[0] / target.shape[1]
        return loss

    ###############
    def boneLoss(self, pred, target):
        # [64, 7, 48, 48]
        def _Frobenius(mat1, mat2):
            return tf.pow(tf.reduce_sum(tf.pow(mat1 - mat2, 2)), 0.5)
            # return torch.sum(torch.pow(mat1-mat2,2))

        _bone_idx = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [2, 4]]

        loss = 0
        for bone_id in _bone_idx:
            bone_pre = pred[:, :, bone_id[0]] - pred[:, :, bone_id[1]]
            bone_gt = target[:, :, bone_id[0]] - target[:, :, bone_id[1]]

            f = _Frobenius(bone_pre, bone_gt)
            loss += f

        loss = loss / len(_bone_idx) / pred.shape[0]
        return loss

    def bgLoss(self, pre, target):
        ##[64, 7, 48, 48]

        bg_pre = tf.reduce_sum(pre, axis=1)
        bg_pre = 1 - tf.clip_by_value(bg_pre, 0, 1)

        bg_gt = tf.reduce_sum(target, axis=1)
        bg_gt = 1 - tf.clip_by_value(bg_gt, 0, 1)

        # weight_mask = (1-bg_gt)*4+1

        loss = tf.reduce_sum(tf.pow((bg_pre - bg_gt), 2)) / pre.shape[0]

        return loss

    def heatmapLoss(self, pred, target, batch_size):
        # [64, 7, 48, 48]
        # print(pred.shape, target.shape)

        # heatmaps_pred = pred.reshape((batch_size, pred.shape[1], -1)).split(1, 1)
        # #对tensor在某一dim维度下，根据指定的大小split_size=int，或者list(int)来分割数据，返回tuple元组
        # #print(len(heatmaps_pred), heatmaps_pred[0].shape)#7 torch.Size([64, 1, 48*48]
        # heatmaps_gt = target.reshape((batch_size, pred.shape[1], -1)).split(1, 1)

        # loss = 0

        # for idx in range(pred.shape[1]):
        #     heatmap_pred = heatmaps_pred[idx].squeeze()#[64, 40*40]
        #     heatmap_gt = heatmaps_gt[idx].squeeze()
        #     if self.use_target_weight:
        #         loss += self.centernetfocalLoss(
        #                         heatmap_pred.mul(self.target_weight[idx//2]),
        #                         heatmap_gt.mul(self.target_weight[idx//2])
        #                     )
        #     else:

        #         loss += self.centernetfocalLoss(heatmap_pred, heatmap_gt)
        # loss /= pred.shape[1]

        return self.myMSEwithWeight(pred, target)

    def centerLoss(self, pred, target, batch_size):
        # heatmaps_pred = pred.reshape((batch_size, -1))
        # heatmaps_gt = target.reshape((batch_size, -1))
        return self.myMSEwithWeight(pred, target)

    def regsLoss(self, pred, target, cx0, cy0, kps_mask, batch_size, num_joints):
        # [64, 14, 48, 48]
        # print('regsLoss', target.shape, cx0.shape, cy0.shape)#torch.Size([64, 14, 48, 48]) torch.Size([64]) torch.Size([64])

        _dim0 = tf.range(0, batch_size, dtype=tf.int32)
        # _dim0 = tf.cast(_dim0, tf.int64)

        _dim1 = tf.zeros(batch_size, dtype=tf.int32)
        # _dim1 = tf.cast(_dim1, tf.int64)

        # print("regsLoss: " , cx0,cy0)
        # print(target.shape)#torch.Size([1, 14, 48, 48])
        # print(torch.max(target[0][2]), torch.min(target[0][2]))
        # print(torch.max(target[0][3]), torch.min(target[0][3]))

        # cv2.imwrite("t.jpg", target[0][2].cpu().numpy()*255)
        loss = 0
        for idx in range(num_joints):
            # gt_x = target[_dim0, _dim1 + idx * 2, cy0, cx0]
            # print(target)

            gt_x = target[cy0[idx], cx0[idx], idx * 2]
            gt_y = target[cy0[idx], cx0[idx], idx * 2 + 1]
            pre_x = pred[cy0[idx], cx0[idx], idx * 2]
            pre_y = pred[cy0[idx], cx0[idx], idx * 2 + 1]

            # gt_x = np.zeros(batch_size)
            # gt_y = np.zeros(batch_size)
            # pre_x = np.zeros(batch_size)
            # pre_y = np.zeros(batch_size)
            # for idx2, (i, j, k, l) in enumerate(zip(_dim0, _dim1 + idx * 2, cy0, cx0)):
            #     gt_x[idx2] = target[i, j, k, l]
            #     pre_x[idx2] = pred[i, j, k, l]
            #     gt_y[idx2] = target[i, j + 1, k, l]
            #     pre_y[idx2] = pred[i, j + 1, k, l]
            #
            # gt_x = tf.convert_to_tensor(gt_x, dtype=tf.float32)
            # gt_y = tf.convert_to_tensor(gt_y, dtype=tf.float32)
            # pre_x = tf.convert_to_tensor(pre_x, dtype=tf.float32)
            # pre_y = tf.convert_to_tensor(pre_y, dtype=tf.float32)

            # gt_x = tf.gather(target, (_dim0, _dim1 + idx * 2, cy0, cx0))
            # gt_y = tf.gather(target, (_dim0, _dim1 + idx * 2 + 1, cy0, cx0))

            # pre_x = pred[_dim0, _dim1 + idx * 2, cy0, cx0]
            # pre_y = pred[_dim0, _dim1 + idx * 2 + 1, cy0, cx0]

            # pre_x = tf.gather(pred, (_dim0, _dim1 + idx * 2, cy0, cx0))
            # pre_y = tf.gather(pred, (_dim0, _dim1 + idx * 2 + 1, cy0, cx0))

            # print(torch.max(target[_dim0,_dim1+idx*2,:,:]),torch.min(target[_dim0,_dim1+idx*2,:,:]))
            # print(gt_x,pre_x)                                       
            # print(gt_y,pre_y)

            # print(kps_mask[:, idx])
            # print(gt_x, pre_x)
            # print(self.l1(gt_x, pre_x, kps_mask[:, idx]))
            # print('---')

            loss += self.l1(gt_x, pre_x, kps_mask[idx])
            loss += self.l1(gt_y, pre_y, kps_mask[idx])
        # b
        # offset_x_pre = torch.clip(pre_x,0,_feature_map_size-1).long()
        # offset_y_pre = torch.clip(pre_y,0,_feature_map_size-1).long()
        # offset_x_gt = torch.clip(gt_x+cx0,0,_feature_map_size-1).long()
        # offset_y_gt = torch.clip(gt_y+cy0,0,_feature_map_size-1).long()

        return loss / num_joints

    def offsetLoss(self, pred, target, cx0, cy0, regs, kps_mask, batch_size, num_joints):
        _dim0 = tf.range(0, batch_size, dtype=tf.int32)
        _dim1 = tf.zeros(batch_size, dtype=tf.int32)
        loss = 0
        # print(gt_y,gt_x)
        for idx in range(num_joints):
            gt_x = regs[cy0[idx], cx0[idx], idx * 2] + tf.cast(cx0, tf.float32)
            gt_y = regs[cy0[idx], cx0[idx], idx * 2 + 1] + tf.cast(cy0, tf.float32)

            # gt_x = np.zeros(batch_size, dtype=np.int32)
            # gt_y = np.zeros(batch_size, dtype=np.int32)
            # gt_offset_x = np.zeros(batch_size)
            # gt_offset_y = np.zeros(batch_size)
            # pre_offset_x = np.zeros(batch_size)
            # pre_offset_y = np.zeros(batch_size)
            # for idx2, (i, j, k, l) in enumerate(zip(_dim0, _dim1 + idx * 2, cy0, cx0)):
            #     gt_x[idx2] = regs[i, j, k, l]
            #     gt_y[idx2] = regs[i, j + 1, k, l]

            # # gt_x = regs[_dim0, _dim1 + idx * 2, cy0, cx0].long() + cx0
            # gt_x = tf.gather(regs, (_dim0, _dim1 + idx * 2, cy0, cx0))
            # gt_x = tf.cast(gt_x, tf.int32) + cx0
            # # gt_y = regs[_dim0, _dim1 + idx * 2 + 1, cy0, cx0].long() + cy0
            # gt_y = tf.gather(regs, (_dim0, _dim1 + idx * 2 + 1, cy0, cx0))
            # gt_y = tf.cast(gt_y, tf.int32) + cy0

            # gt_x[gt_x > 47] = 47
            # gt_x[gt_x < 0] = 0
            # gt_y[gt_y > 47] = 47
            # gt_y[gt_y < 0] = 0

            gt_x = tf.cast(tf.clip_by_value(gt_x, 0, 47), tf.int32)
            gt_y = tf.cast(tf.clip_by_value(gt_y, 0, 47), tf.int32)

            # gt_x = tf.convert_to_tensor(gt_x, dtype=tf.float32)
            # gt_y = tf.convert_to_tensor(gt_y, dtype=tf.float32)

            gt_offset_x = target[gt_y[idx], gt_x[idx], idx * 2]
            gt_offset_y = target[gt_y[idx], gt_x[idx], idx * 2 + 1]
            pre_offset_x = pred[gt_y[idx], gt_x[idx], idx * 2]
            pre_offset_y = pred[gt_y[idx], gt_x[idx], idx * 2 + 1]

            # for idx2, (i, j, k, l) in enumerate(zip(_dim0, _dim1 + idx * 2, gt_y, gt_x)):
            #     gt_offset_x[idx2] = target[i, j, k, l]
            #     gt_offset_y[idx2] = target[i, j + 1, k, l]
            #     pre_offset_x[idx2] = pred[i, j, k, l]
            #     pre_offset_y[idx2] = pred[i, j + 1, k, l]
            #
            # gt_offset_x = tf.convert_to_tensor(gt_offset_x, dtype=tf.float32)
            # gt_offset_y = tf.convert_to_tensor(gt_offset_y, dtype=tf.float32)
            # pre_offset_x = tf.convert_to_tensor(pre_offset_x, dtype=tf.float32)
            # pre_offset_y = tf.convert_to_tensor(pre_offset_y, dtype=tf.float32)

            # # gt_offset_x = target[_dim0, _dim1 + idx * 2, gt_y, gt_x]
            # # gt_offset_y = target[_dim0, _dim1 + idx * 2 + 1, gt_y, gt_x]
            # gt_offset_x = tf.gather(target, (_dim0, _dim1 + idx * 2, gt_y, gt_x))
            # gt_offset_y = tf.gather(target, (_dim0, _dim1 + idx * 2 + 1, gt_y, gt_x))
            #
            # # pre_offset_x = pred[_dim0, _dim1 + idx * 2, gt_y, gt_x]
            # # pre_offset_y = pred[_dim0, _dim1 + idx * 2 + 1, gt_y, gt_x]
            # pre_offset_x = tf.gather(pred, (_dim0, _dim1 + idx * 2, gt_y, gt_x))
            # pre_offset_y = tf.gather(pred, (_dim0, _dim1 + idx * 2 + 1, gt_y, gt_x))

            # print(gt_offset_x, torch.max(target[_dim0,_dim1+idx*2,...]),torch.min(target[_dim0,_dim1+idx*2,...]))
            # print(gt_offset_y, torch.max(target[_dim0,_dim1+idx*2+1,...]),torch.min(target[_dim0,_dim1+idx*2+1,...]))
            loss += self.l1(gt_offset_x, pre_offset_x, kps_mask[idx])
            loss += self.l1(gt_offset_y, pre_offset_y, kps_mask[idx])
        #     print(gt_y,gt_x)    
        # b
        return loss / num_joints

        """
        0.0 0.5
        0.0 0.75
        0.75 0.25
        0.0 0.75
        0.0 0.5
        """

    def maxPointPth(self, heatmap, center_weight, center=True):
        # pytorch version
        # n,1,h,w
        # 计算center heatmap的最大值得到中心点
        if center:
            # heatmap = heatmap * self.center_weight
            # heatmap = heatmap * tf.Graph.get_collection('center_weight')[0]
            heatmap = heatmap * center_weight
            # 加权取最靠近中间的

        h, w, c = heatmap.shape
        # print(heatmap)
        heatmap = tf.reshape(heatmap, [48 * 48, c])
        # print(heatmap[0])
        # max_id = torch.argmax(heatmap, 1)#64, 1
        max_id = tf.argmax(heatmap, axis=0)
        # print(max_id)
        # max_v, max_id = tf.reduce_max(heatmap, 1)  # 64, 1
        # print(max_v)
        # print("max_i: ",max_i)

        # mask0 = torch.zeros(max_v.shape).to(heatmap.device)
        # mask1 = torch.ones(max_v.shape).to(heatmap.device)
        # mask = torch.where(torch.gt(max_v,th), mask1, mask0)
        # print(mask)
        # b
        y = max_id // w
        x = max_id % w

        return x, y

    def call(self, y_pred, y_true):  # y_true: target and kps_mask, y_pred: output
        # print("output: ", output.shape)
        # 更改通道顺序    (batch, h, w, num_joints)
        # y_pred[0] = tf.transpose(y_pred[0], [0, 3, 1, 2])
        # y_pred[1] = tf.transpose(y_pred[1], [0, 3, 1, 2])
        # y_pred[2] = tf.transpose(y_pred[2], [0, 3, 1, 2])
        # y_pred[3] = tf.transpose(y_pred[3], [0, 3, 1, 2])

        # y_pred = tf.expand_dims(y_pred, axis=0)

        # y_pred0 = y_pred[0]
        # y_pred1 = y_pred[1]
        # y_pred2 = y_pred[2]
        # y_pred3 = y_pred[3]

        # for y_pred0, y_pred1, y_pred2, y_pred3 in zip(y_pred0, y_pred1, y_pred2, y_pred3):

        # batch_size = y_pred.shape[0]
        batch_size = self.cfg['batch_size']

        # print("batch_size: ", batch_size)
        # print(y_pred1)
        # num_joints = y_pred0.shape[2]
        # print("num_joints: ", num_joints)

        # print("output: ", [x.shape for x in output])
        # [64, 7, 48, 48] [64, 1, 48, 48] [64, 14, 48, 48] [64, 14, 48, 48]
        # print("target: ", [x.shape for x in target])#[64, 36, 48, 48]
        # print(weights.shape)# [14,]

        # print(y_true)

        # target :
        # {"img_name": "000000425226_0.jpg",
        #  "keypoints": [0.16815476190476192, -0.08333333333333333, 0,
        #                0.16815476190476192, -0.08333333333333333, 0,
        #                0.16815476190476192, -0.08333333333333333, 0,
        #                0.16815476190476192, -0.08333333333333333, 0,
        #                0.16815476190476192, -0.08333333333333333, 0,
        #                0.3794642857142857, 0.37648809523809523, 1,
        #                0.43154761904761907, 0.39285714285714285, 2,
        #                0.4523809523809524, 0.5089285714285714, 2,
        #                0.5208333333333334, 0.38839285714285715, 2,
        #                0.5148809523809523, 0.5505952380952381, 2,
        #                0.6235119047619048, 0.26339285714285715, 2,
        #                0.30505952380952384, 0.5892857142857143, 2,
        #                0.35119047619047616, 0.6130952380952381, 2,
        #                0.16815476190476192, -0.08333333333333333, 0,
        #                0.5416666666666666, 0.6145833333333334, 2,
        #                0.16815476190476192, -0.08333333333333333, 0,
        #                0.40922619047619047, 0.7366071428571429, 2],
        #  "center": [0.5, 0.5],
        #  "bbox": [0.2767857142857143, 0.22321428571428573,
        #           0.7232142857142857, 0.7767857142857143],
        #  "other_centers": [],
        #  "other_keypoints": [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]}

        # {"img_name": "000000292456_0.jpg",
        #  "keypoints": [0.46511627906976744, 0.27505827505827507, 1,
        #                0.4744186046511628, 0.26107226107226106, 2,
        #                -0.13023255813953488, 0.0979020979020979, 0,
        #                0.49534883720930234, 0.26573426573426573, 2,
        #                -0.13023255813953488, 0.0979020979020979, 0,
        #                0.4604651162790698, 0.358974358974359, 2,
        #                0.5837209302325581, 0.32867132867132864, 1,
        #                0.4186046511627907, 0.4358974358974359, 2,
        #                0.6558139534883721, 0.44755244755244755, 1,
        #                0.33488372093023255, 0.47086247086247085, 2,
        #                -0.13023255813953488, 0.0979020979020979, 0,
        #                0.49767441860465117, 0.5384615384615384, 1,
        #                0.586046511627907, 0.5361305361305362, 1,
        #                0.46744186046511627, 0.627039627039627, 2,
        #                0.7046511627906977, 0.6713286713286714, 1,
        #                0.45348837209302323, 0.7668997668997669, 1,
        #                -0.13023255813953488, 0.0979020979020979, 0],
        #  "center": [0.44651162790697674, 0.5],
        #  "bbox": [0.29767441860465116, 0.20279720279720279,
        #           0.5953488372093023, 0.7972027972027972],
        #  "other_centers": [[0.6546511627906977, 0.36363636363636365],
        #                    [0.23953488372093024, 0.44755244755244755]],
        #  "other_keypoints": [[[0.6744186046511628, 0.3123543123543124],
        #                       [0.25813953488372093, 0.23776223776223776]],
        #                      [[0.6767441860465117, 0.3006993006993007],
        #                       [0.2627906976744186, 0.22377622377622378]],
        #                      [[0.6604651162790698, 0.29836829836829837],
        #                       [0.24883720930232558, 0.22377622377622378]],
        #                      [],
        #                      [[0.6302325581395349, 0.30303030303030304],
        #                       [0.22093023255813954, 0.22377622377622378]],
        #                      [[0.6046511627906976, 0.351981351981352],
        #                       [0.24186046511627907, 0.3146853146853147]],
        #                      [[0.6534883720930232, 0.351981351981352],
        #                       [0.2069767441860465, 0.317016317016317]],
        #                      [],
        #                      [[0.7023255813953488, 0.44988344988344986],
        #                       [0.2255813953488372, 0.3682983682983683]],
        #                      [],
        #                      [[0.7534883720930232, 0.5151515151515151],
        #                       [0.27906976744186046, 0.4195804195804196]],
        #                      [[0.5883720930232558, 0.5104895104895105],
        #                       [0.23488372093023255, 0.44522144522144524]],
        #                      [[0.6395348837209303, 0.5151515151515151],
        #                       [0.2116279069767442, 0.44988344988344986]],
        #                      [[0.2255813953488372, 0.5501165501165501]],
        #                      [[0.22093023255813954, 0.578088578088578]],
        #                      [[0.20232558139534884, 0.6456876456876457]],
        #                      [[0.2302325581395349, 0.696969696969697]]]}

        # y_true_e = tf.expand_dims(y_true[0], axis=0)
        # kps_mask = tf.expand_dims(y_true[1], axis=0)

        # heatmaps = y_true[0, :, :, :17]
        centers = y_true[:, :, :, 17:18]
        # regs = y_true[0, :, :, 18:52]
        # offsets = y_true[0, :, :, 52:86]
        # kps_mask = y_true[0, 0, 0, 86:]

        # heatmap_loss = self.heatmapLoss(y_pred0, heatmaps, batch_size)

        # bg_loss = self.bgLoss(output[0], heatmaps)
        # bone_loss = self.boneloss(output[0], heatmaps)
        # bone_loss = self.boneLoss(y_pred0, heatmaps)
        # print(heatmap_loss)
        center_loss = self.centerLoss(y_pred, centers, batch_size)

        # if not self.make_center_w:
        #     center_weight = tf.reshape(self.center_weight, (48, 48, 1))
        #     center_weight = tf.tile(center_weight, (1, 1, num_joints))
        #     # g = tf.get_default_graph()
        #     # g.add_to_collection(name='center_weight', value=center_weight)
        #
        #     # print(self.center_weight.shape)
        #     # b
        #     # self.center_weight = self.center_weight
        #     # self.make_center_w = True
        #     # self.center_weight.requires_grad_(False)
        #
        #     # self.range_weight_x = self.range_weight_x.to(target.device)
        #     # self.range_weight_y = self.range_weight_y.to(target.device)
        #     # self.range_weight_x.requires_grad_(False)
        #     # self.range_weight_y.requires_grad_(False)
        # # print(self.center_weight)
        #
        # cx0, cy0 = self.maxPointPth(centers, center_weight)
        # # cx1, cy1 = self.maxPointPth(pre_centers)
        # # cx0 = tf.clip_by_value(cx0, 0, _feature_map_size - 1).to_int64()
        # cx0 = tf.cast(tf.clip_by_value(cx0, 0, _feature_map_size - 1), tf.int32)
        # # cy0 = tf.clip_by_value(cy0, 0, _feature_map_size - 1).to_int64()
        # cy0 = tf.cast(tf.clip_by_value(cy0, 0, _feature_map_size - 1), tf.int32)
        # # cx1 = torch.clip(cx1,0,_feature_map_size-1).long()
        # # cy1 = torch.clip(cy1,0,_feature_map_size-1).long()
        #
        # # print(cx0, cy0)
        # # bbb
        # # cv2.imwrite("_centers.jpg", centers[0][0].cpu().numpy()*255)
        # # b
        #
        # regs_loss = self.regsLoss(y_pred2, regs, cx0, cy0, kps_mask, batch_size, num_joints)
        # offset_loss = self.offsetLoss(y_pred3, offsets,
        #                               cx0, cy0, regs,
        #                               kps_mask, batch_size, num_joints)
        #
        # # total_loss = heatmap_loss+center_loss+0.1*regs_loss+offset_loss
        # # print(heatmap_loss,center_loss,regs_loss,offset_loss)
        # # b
        #
        # """
        #
        # """
        # # boneloss = self.boneLoss(output[3], offsets,
        # #                     cx0, cy0,regs,
        # #                     kps_mask,batch_size, num_joints)
        #
        # # heatmap_loss = tf.cast(heatmap_loss, tf.float32)
        # # center_loss = tf.cast(center_loss, tf.float32)
        # # regs_loss = tf.cast(regs_loss, tf.float32)
        # # offset_loss = tf.cast(offset_loss, tf.float32)
        #
        # all_loss = tf.reduce_sum(heatmap_loss + center_loss + regs_loss + offset_loss + bone_loss)
        #
        # # return all_loss
        return center_loss

movenetLoss = MovenetLoss(use_target_weight=False)


def calculate_loss(predict, label):
    loss = movenetLoss(predict, label)
    return loss
