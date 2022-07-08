import tensorflow as tf
import numpy as np

_img_size = 192
_feature_map_size = _img_size // 4

_center_weight_path = 'lib/data/center_weight_origin.npy'


class MovenetLoss(tf.keras.Model):
    def __init__(self, use_target_weight=False, target_weight=[1]):
        super(MovenetLoss, self).__init__()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.use_target_weight = use_target_weight
        self.target_weight = target_weight

        self.center_weight = tf.keras.backend.variable(np.load('center_weight.npy'))
        self.make_center_w = False

        self.boneloss = JointBoneLoss(17)

    def l1(self, pre, target, kps_mask):
        return tf.reduce_sum(tf.abs(pre - target) * kps_mask) / (tf.reduce_sum(kps_mask) + 1e-4)

    def l2_loss(self, pre, target):
        loss = (pre - target)
        loss = (loss ** 2) / 2 / pre.shape[0]

        return tf.reduce_sum(loss)

    def centernetfocalLoss(self, pred, gt):
        pass

    def myMSEwithWeight(self, pre, target):
        loss = tf.math.pow(pre - target, 2)

        weight_mask = target * 8 + 1

        loss = loss * weight_mask

        loss = tf.reduce_sum(loss) / target.shape[0] / target.shape[1]

        return loss

    def heatmapL1(self, pre, target):
        loss = tf.math.abs(pre - target)
        weight_mask = target * 4 + 1
        loss = loss * weight_mask
        loss = tf.reduce_sum(loss) / target.shape[0] / target.shape[1]
        return loss

    def boneLoss(self, pred, target):
        def _Frobenius(mat1, mat2):
            return tf.math.pow(tf.math.reduce_sum(tf.math.pow(mat1 - mat2, 2)), 0.5)

        _bone_idx = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [2, 4]]

        loss = 0
        for bone_id in _bone_idx:
            bone_pre = pred[:, bone_id[0], :, :] - pred[:, bone_id[1], :, :]
            bone_gt = target[:, bone_id[0], :, :] - target[:, bone_id[1], :, :]
            f = _Frobenius(bone_pre, bone_gt)
            loss += f

        loss = loss / len(_bone_idx) / pred.shape[0]
        return loss

    def bgLoss(self, pre, target):
        bg_pre = tf.math.reduce_sum(pre, axis=1)
        bg_pre = 1 - tf.clip_by_value(bg_pre, 0, 1)

        bg_gt = tf.math.reduce_sum(target, axis=1)
        bg_gt = 1 - tf.clip_by_value(bg_gt, 0, 1)

        loss = tf.math.reduce_sum(tf.math.pow(bg_pre - bg_gt, 2)) / target.shape[0]

        return loss

    def heatmapLoss(self, pred, target, batch_size):
        return self.myMSEwithWeight(pred, target)

    def centerLoss(self, pred, target, batch_size):
        return self.myMSEwithWeight(pred, target)

    def regsLoss(self, pred, target, cx0, cy0, kps_mask, batch_size, num_joints):
        _dim0 = tf.range(0, batch_size).to_int64()
        _dim1 = tf.zeros(batch_size).to_int64()

        loss = 0
        for idx in range(num_joints):
            gt_x = target[_dim0, _dim1 + idx * 2, cy0, cx0]
            gt_y = target[_dim0, _dim1 + idx * 2 + 1, cy0, cx0]
            pre_x = pred[_dim0, _dim1 + idx * 2, cy0, cx0]
            pre_y = pred[_dim0, _dim1 + idx * 2 + 1, cy0, cx0]
            loss += self.l1(gt_x, pre_x, kps_mask[:, idx])
            loss += self.l1(gt_y, pre_y, kps_mask[:, idx])
        return loss / num_joints

    def offsetLoss(self, pred, target, cx0, cy0, regs, kps_mask, batch_size, num_joints):
        _dim0 = tf.range(0, batch_size).to_int64()
        _dim1 = tf.zeros(batch_size).to_int64()
        loss = 0
        for idx in range(num_joints):
            gt_x = regs[_dim0, _dim1 + idx * 2, cy0, cx0].to_int64() + cx0
            gt_y = regs[_dim0, _dim1 + idx * 2 + 1, cy0, cx0].to_int64() + cy0

            gt_x[gt_x > 47] = 47
            gt_x[gt_x < 0] = 0
            gt_y[gt_y > 47] = 47
            gt_y[gt_y < 0] = 0

            gt_offset_x = target[_dim0, _dim1 + idx * 2, gt_y, gt_x]
            gt_offset_y = target[_dim0, _dim1 + idx * 2 + 1, gt_y, gt_x]

            pre_offset_x = pred[_dim0, _dim1 + idx * 2, gt_y, gt_x]
            pre_offset_y = pred[_dim0, _dim1 + idx * 2 + 1, gt_y, gt_x]

            loss += self.l1(gt_offset_x, pre_offset_x, kps_mask[:, idx])
            loss += self.l1(gt_offset_y, pre_offset_y, kps_mask[:, idx])

        return loss / num_joints

    def maxPointPth(self, heatmap, center=True):
        if center:
            heatmap = heatmap * self.center_weight[:heatmap.shape[0], ...]
        n, c, h, w = heatmap.shape
        heatmap = tf.reshape(heatmap, (n, -1))
        max_id = tf.argmax(heatmap, axis=1)
        y = max_id // w
        x = max_id % w
        return x, y

    @tf.function
    def call(self, output, target, kps_mask):
        batch_size = output[0].size(0)
        num_joints = output[0].size(1)

        heatmap = target[:, :17, :, :]
        centers = target[:, 17:18, :, :]
        regs = target[:, 18:52, :, :]
        offset = target[:, 52:, :, :]

        heatmap_loss = self.heatmapLoss(output[0], heatmap, batch_size)

        bone_loss = self.boneLoss(output[0], heatmap)

        center_loss = self.centerLoss(output[1], centers, batch_size)

        if not self.make_center_w:
            self.center_weight = tf.reshape(self.center_weight, (1, 1, 48, 48))
            self.center_weight = tf.tile(self.center_weight, (output[1].shape[0], output[1].shape[1], 1, 1))
            self.make_center_w = True

        cx0, cy0 = self.maxPointPth(centers)
        cx0 = tf.clip_by_value(cx0, 0, _feature_map_size - 1).to_int64()
        cy0 = tf.clip_by_value(cy0, 0, _feature_map_size - 1).to_int64()

        regs_loss = self.regsLoss(output[2], regs, cx0, cy0, kps_mask, batch_size, num_joints)
        offset_loss = self.offsetLoss(output[3], offset, cx0, cy0, regs, kps_mask, batch_size, num_joints)

        return [heatmap_loss, bone_loss, center_loss, regs_loss, offset_loss]


class JointBoneLoss(tf.keras.Model):
    def __init__(self, joint_num):
        super(JointBoneLoss, self).__init__()
        id_i, id_j = [], []
        for i in range(joint_num):
            for j in range(i + 1, joint_num):
                id_i.append(i)
                id_j.append(j)
        self.id_i = id_i
        self.id_j = id_j

    @tf.function
    def call(self, joint_out, joint_gt):
        J = tf.norm(joint_out[:, self.id_i, :] - joint_out[:, self.id_j, :], ord=2, axis=-1, keepdims=False)
        Y = tf.norm(joint_gt[:, self.id_i, :] - joint_gt[:, self.id_j, :], ord=2, axis=-1, keepdims=False)
        loss = tf.abs(J - Y)
        loss = tf.reduce_sum(loss) / joint_out.shape[0] / len(self.id_i)
        return loss


movenetLoss = MovenetLoss(use_target_weight=False)


def calculate_loss(predict, label):
    loss = movenetLoss(predict, label)
    return loss
