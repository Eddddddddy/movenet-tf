import tensorflow as tf


# A tensorflow model has backbone layer and header layer.
class MoveNet(tf.keras.Model):
    def __init__(self, num_classes, width_mult=1, mode='train'):
        super().__init__()
        self.backbone = Backbone()
        self.header = Header(num_classes, mode)
        # self._initialize_weights()

    def call(self, inputs):
        x = self.backbone(inputs)
        x = self.header(x)
        return x

    # def _initialize_weights(self):
    #     # get all the trainable variables
    #     trainable_variables = self.trainable_variables


class Backbone(tf.keras.Model):
    def __init__(self):
        super(Backbone, self).__init__()

        input_channel = 32

        self.features1 = tf.keras.Sequential([
            conv_3x3_act(3, input_channel, 2),
            dw_conv(input_channel, 16, 1),
            InvertedResidual(16, 24, 2, 6, 1)
        ])
        self.features2 = InvertedResidual(24, 32, 2, 6, 2)
        self.features3 = InvertedResidual(32, 64, 2, 6, 3)
        self.features4 = tf.keras.Sequential([
            InvertedResidual(64, 96, 1, 6, 2),
            InvertedResidual(96, 160, 2, 6, 2),
            InvertedResidual(160, 320, 1, 6, 0),
            conv_1x1_act(320, 1280),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="valid", use_bias=False),
            tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        ])

        self.upsample2 = upsample(64, 32)
        self.upsample1 = upsample(32, 24)

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="valid")
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="valid")
        self.conv1 = tf.keras.layers.Conv2D(filters=24, kernel_size=(1, 1), strides=(1, 1), padding="valid")

        self.conv4 = dw_conv3(24, 24, 1)

    def call(self, inputs):
        inputs = inputs / 127.5 - 1
        f1 = self.features1(inputs)
        # print(f1.shape)
        f2 = self.features2(f1)
        # print(f2.shape)
        f3 = self.features3(f2)
        # print(f3.shape)
        f4 = self.features4(f3)
        # print(f4.shape)

        f3 = self.conv3(f3)
        # print(f3.shape)
        # print(f4.shape)
        f4 += f3
        f4 = self.upsample2(f4)

        f2 = self.conv2(f2)
        f4 += f2
        f4 = self.upsample1(f4)

        f1 = self.conv1(f1)
        f4 += f1

        f4 = self.conv4(f4)

        # print(f4.shape)

        return f4


class Header(tf.keras.Model):
    def __init__(self, num_classes, mode='train'):
        super().__init__()
        self.mode = mode

        self.header_heatmaps = tf.keras.Sequential([
            dw_conv3(24, 96),
            tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                                   use_bias=True, activation='sigmoid')
        ])

        self.header_centers = tf.keras.Sequential([
            dw_conv3(24, 96),
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding="valid", use_bias=True,
                                   activation='sigmoid')
        ])

        self.header_regs = tf.keras.Sequential([
            dw_conv3(24, 96),
            tf.keras.layers.Conv2D(filters=num_classes * 2, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                                   use_bias=True)
        ])

        self.header_offsets = tf.keras.Sequential([
            dw_conv3(24, 96),
            tf.keras.layers.Conv2D(filters=num_classes * 2, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                                   use_bias=True)
        ])

    # def argmax2loc(self, x, h=48, w=48):
    #     y0 = tf.divide(x, w).to_int64()
    #     x0 = tf.subtract(x, y0 * w).to_int64()
    #     return x0, y0

    def call(self, inputs):
        res = []

        if self.mode == 'train':
            h1 = self.header_heatmaps(inputs)
            h2 = self.header_centers(inputs)
            h3 = self.header_regs(inputs)
            h4 = self.header_offsets(inputs)
            res = [h1, h2, h3, h4]

        # elif self.mode == 'test':
        #     pass
        #
        # elif self.mode == 'all':
        #     pass
        #
        # else:
        #     print("Error: mode is not defined")

        return res


class InvertedResidual(tf.keras.Model):
    def __init__(self, inp, oup, stride, expand_ratio, n):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.n = n

        self.conv1 = tf.keras.Sequential([
            # pw
            tf.keras.layers.Conv2D(filters=hidden_dim, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.ReLU(),
            # dw
            tf.keras.layers.Conv2D(filters=hidden_dim, kernel_size=(3, 3), strides=(stride, stride), padding="same",
                                   groups=hidden_dim, use_bias=True),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.ReLU(),
            # pw-linear
            tf.keras.layers.Conv2D(filters=oup, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(axis=1)
        ])
        if n > 0:
            self.conv2 = tf.keras.Sequential([
                # pw
                tf.keras.layers.Conv2D(filters=hidden_dim, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                                       use_bias=False),
                tf.keras.layers.BatchNormalization(axis=1),
                tf.keras.layers.ReLU(),
                # dw
                tf.keras.layers.Conv2D(filters=hidden_dim, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                       groups=hidden_dim, use_bias=False),
                tf.keras.layers.BatchNormalization(axis=1),
                tf.keras.layers.ReLU(),
                # pw-linear
                tf.keras.layers.Conv2D(filters=oup, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                                       use_bias=False),
                tf.keras.layers.BatchNormalization(axis=1)
            ])

    def call(self, inputs):
        x = self.conv1(inputs)

        for _ in range(self.n):
            x = x + self.conv2(x)

        return x


def upsample(inp, oup, scale=2):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=inp, kernel_size=(3, 3), strides=(1, 1), padding="same", groups=inp),
        tf.keras.layers.ReLU(),
        conv_1x1_act2(inp, oup),
        tf.keras.layers.UpSampling2D(size=scale, interpolation='bilinear')
    ])


def dw_conv3(inp, oup, stride=1):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=inp, kernel_size=(3, 3), strides=(stride, stride), padding="same", groups=inp,
                               use_bias=False),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.Conv2D(filters=oup, kernel_size=(1, 1), strides=(1, 1), padding="valid", use_bias=False),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.ReLU()
    ])


def dw_conv2(inp, oup, stride=1):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=inp, kernel_size=(3, 3), strides=(stride, stride), padding="same", groups=inp,
                               use_bias=False),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.Conv2D(filters=oup, kernel_size=(1, 1), strides=(1, 1), padding="valid", use_bias=False),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.ReLU()
    ])


def dw_conv(inp, oup, stride=1):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=inp, kernel_size=(3, 3), strides=(stride, stride), padding="same", groups=inp,
                               use_bias=False),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=oup, kernel_size=(1, 1), strides=(1, 1), padding="valid", use_bias=False),
        tf.keras.layers.BatchNormalization(axis=1)
    ])


def conv_1x1_act2(inp, oup):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=oup, kernel_size=(1, 1), strides=(1, 1), padding="valid", use_bias=False),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.ReLU()
    ])


def conv_1x1_act(inp, oup):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=oup, kernel_size=(1, 1), strides=(1, 1), padding="valid", use_bias=False),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.ReLU()
    ])


def conv_3x3_act(inp, oup, stride):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(192, 192, inp), filters=oup, kernel_size=(3, 3), strides=(stride, stride),
                               padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.ReLU()
    ])
