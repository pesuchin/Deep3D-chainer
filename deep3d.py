import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import matplotlib.pyplot as plt


def select_layer(masks, left_image, left_shift=16):
    """Use Selection Layer for applying mask to image.

    Arguments:
        masks {Variable} -- extracted feature by deep3d (shape N, H, W, S)
        left_image {Variable} -- original left image (shape N, C, H, W)
    returns:
        predict right image {Variable} -- applyed mask to original left image (shape N, C, H, W)
    """
    _, H, W, S = masks.shape
    padded = F.pad(left_image,
                   [[0, 0], [0, 0], [0, 0], [left_shift, left_shift]],
                   mode='constant')

    for s in np.arange(S):
        mask_slice = masks[:, :, :, s]
        mask_slice = F.expand_dims(mask_slice, axis=1)
        pad_slice = F.get_item(padded, (slice(None), slice(None), slice(0, H), slice(s, W+s)))
        if s == 0:
            pred = F.expand_dims(pad_slice * mask_slice, axis=4)
        else:
            tmp = F.expand_dims(pad_slice * mask_slice, axis=4)
            pred = F.concat([pred, tmp], axis=4)
    return F.sum(pred, axis=4)


def plot_argmax(masks, left, masked_left_images, right):
    """Plot depth map, original left image, predict image, right image.

    Arguments:
        masks {Variable} -- extracted feature by deep3d
        left {Variable} -- original left image
        masked_left_images {Variable} -- applyed mask to original left image (predict right image)
        right {Variable} -- right image
    """
    mask = chainer.cuda.to_cpu(masks.data[0, :, :, :])
    fig = plt.figure(figsize=(18, 12))
    fig.add_subplot(1, 4, 1)
    plt.imshow(np.argmax(mask, axis=2), cmap='inferno')
    fig.add_subplot(1, 4, 2)
    plt.imshow(chainer.cuda.to_cpu(left[0, :, :, :]).transpose(1, 2, 0))
    fig.add_subplot(1, 4, 3)
    plt.imshow(chainer.cuda.to_cpu(masked_left_images.data[0, :, :, :]).transpose(1, 2, 0))
    fig.add_subplot(1, 4, 4)
    plt.imshow(chainer.cuda.to_cpu(right[0, :, :, :]).transpose(1, 2, 0))
    plt.show()


class Deep3D(chainer.Chain):
    """Deep3D network architecture.

    Reference:
        https://github.com/piiswrong/deep3d/blob/e9433221662001717cfafe89c5f8a7e3b26fe1ee/sym.py
        https://github.com/JustinTTL/Deep3D_TF/blob/master/Deep3D_Final.py
    """

    def __init__(self, batchsize, size):
        """Init Deep3D.

        Arguments:
            batchsize {integer} -- batchsize for learning
            size {tuple} -- image size (width, height)
        """
        super(Deep3D, self).__init__()
        self.batchsize = batchsize
        self.size = size
        with self.init_scope():
            self.vgg_layers = L.VGG16Layers()
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(None, int(33 * self.size[0]/32 * self.size[1]/32))

            self.branch_conv1 = L.Convolution2D(64, 33, ksize=(3, 3), stride=1, pad=(1, 1))
            self.branch_conv2 = L.Convolution2D(128, 33, ksize=(3, 3), stride=1, pad=(1, 1))
            self.branch_conv3 = L.Convolution2D(256, 33, ksize=(3, 3), stride=1, pad=(1, 1))
            self.branch_conv4 = L.Convolution2D(512, 33, ksize=(3, 3), stride=1, pad=(1, 1))

            self.batch_norm1 = L.BatchNormalization(64)
            self.batch_norm2 = L.BatchNormalization(128)
            self.batch_norm3 = L.BatchNormalization(256)
            self.batch_norm4 = L.BatchNormalization(512)

            scale = 1
            W, bias = self.get_initial_deconv_value(scale, 33, 33)
            self.deconv1 = L.Deconvolution2D(33, 33, ksize=(1, 1),
                                             stride=(1, 1),
                                             pad=(0, 0),
                                             initialW=W,
                                             initial_bias=bias)
            scale *= 2
            W, bias = self.get_initial_deconv_value(2*scale, 33, 33)
            self.deconv2 = L.Deconvolution2D(33, 33, ksize=(2*scale, 2*scale),
                                             stride=(scale, scale), 
                                             pad=(scale//2, scale//2),
                                             initialW=W,
                                             initial_bias=bias)
            scale *= 2
            W, bias = self.get_initial_deconv_value(2*scale, 33, 33)
            self.deconv3 = L.Deconvolution2D(33, 33, ksize=(2*scale, 2*scale),
                                             stride=(scale, scale),
                                             pad=(scale//2, scale//2),
                                             initialW=W,
                                             initial_bias=bias)
            scale *= 2
            W, bias = self.get_initial_deconv_value(2*scale, 33, 33)
            self.deconv4 = L.Deconvolution2D(33, 33, ksize=(2*scale, 2*scale),
                                             stride=(scale, scale),
                                             pad=(scale//2, scale//2),
                                             initialW=W,
                                             initial_bias=bias)
            scale *= 2
            W, bias = self.get_initial_deconv_value(2*scale, 33, 33)
            self.deconv5 = L.Deconvolution2D(33, 33, ksize=(2*scale, 2*scale),
                                             stride=(scale, scale),
                                             pad=(scale//2, scale//2),
                                             initialW=W,
                                             initial_bias=bias)
            scale = 2
            W, bias = self.get_initial_deconv_value(2*scale, 33, 33)
            self.up_deconv_layer = L.Deconvolution2D(33, 33, ksize=(2*scale, 2*scale),
                                                     stride=(scale, scale),
                                                     pad=(scale//2, scale//2),
                                                     initialW=W,
                                                     initial_bias=bias)
            self.up_conv_layer = L.Convolution2D(33, 33, ksize=(3, 3),
                                                 stride=1,
                                                 pad=(1, 1))

    def __call__(self, left, right, original_left):
        """Calculate loss function by mean absolute error.

        Arguments:
            left {Variable} -- left image (BGR)
            right {Variable} -- right image (RGB)
            original_left {Variable} -- original left image (RGB)

        Returns:
            loss {Variable} -- loss value

        """
        masked_left_images, mask = self.predict(left, original_left)
        if np.random.random() < 0.01:
            plot_argmax(mask, original_left, masked_left_images, right)
        loss = F.mean_absolute_error(masked_left_images, right)
        chainer.report({'loss': loss}, self)
        return loss

    def predict(self, left, original_left):
        """Predict right image.

        Arguments:
            left {Variable} -- left image (BGR)
            original_left {Variable} -- original left image (RGB)

        Returns:
            masked_left_images {Variable} -- predicted right image
            mask {Variable} -- mask

        """
        mask = self.get_mask_value(left)
        masked_left_images = select_layer(mask, original_left)
        return masked_left_images, mask

    def get_mask_value(self, bgr):
        """Get mask.

        Arguments:
            bgr {Variable} -- left image (BGR)

        Returns:
            mask {Variable} -- mask for obtaining prediction right image

        """
        # VGG
        use_layers = ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']
        vgg_result = self.vgg_layers(bgr, layers=use_layers)
        vgg_fc6 = F.dropout(F.relu(self.fc6(vgg_result['pool5'])))
        vgg_fc7 = F.dropout(F.relu(self.fc7(vgg_fc6)))
        vgg_fc8 = F.dropout(F.relu(self.fc8(vgg_fc7)))
        
        # -------branch 1-----
        bn_pool1 = self.batch_norm1(vgg_result['pool1'])
        branch1_1 = F.relu(self.branch_conv1(bn_pool1))
        branch1_2 = self.deconv1(branch1_1)

        # -------branch 2-----
        bn_pool2 = self.batch_norm2(vgg_result['pool2'])
        branch2_1 = F.relu(self.branch_conv2(bn_pool2))
        branch2_2 = self.deconv2(branch2_1)

        # -------branch 3-----
        bn_pool3 = self.batch_norm3(vgg_result['pool3'])
        branch3_1 = F.relu(self.branch_conv3(bn_pool3))
        branch3_2 = self.deconv3(branch3_1)

        # -------branch 4-----
        bn_pool4 = self.batch_norm4(vgg_result['pool4'])
        branch4_1 = F.relu(self.branch_conv4(bn_pool4))
        branch4_2 = self.deconv4(branch4_1)

        # -------branch 5-----
        # Upscaling last branch
        fc_RS = F.reshape(vgg_fc8, [self.batchsize, 33, int(self.size[1]/32), int(self.size[0]/32)])
        branch5_1 = F.relu(fc_RS)
        branch5_2 = self.deconv5(branch5_1)

        # Combine and x2 Upsample
        up_sum = branch1_2 + branch2_2 + branch3_2 + branch4_2 + branch5_2
        up = self.up_deconv_layer(up_sum)
        up = F.relu(up)

        # Last Conv Layer
        up_conv = F.relu(self.up_conv_layer(up))

        # Add + Mask + Selection
        mask = F.softmax(up_conv)
        mask = F.transpose(mask, (0, 2, 3, 1))
        return mask


# =========== Macro Layers =========== #
    def get_initial_deconv_value(self, filter_size, in_channels, out_channels,
                                 bias=True, initialization='bilinear'):
        """Get initial weight by bilinear interpolation.

        Arguments:
            filter_size {integer} -- filter size of CNN
            in_channels {integer} -- input channels of CNN
            out_channels {integer} -- output channels of CNN

        Keyword Arguments:
            bias {bool} -- whether use bias (default: {True})
            initialization {str} -- initialization method (default: {'bilinear'})

        Returns:
            filters [numpy array] -- initial value of filter
            biases [numpy array] -- initial value of biases

        """
        # Initializing to bilinear interpolation
        if initialization == 'bilinear':
            S = filter_size / 2
            C = (filter_size - 1 - (S % 2)) / (filter_size)
            initial_value = np.zeros([in_channels, out_channels, filter_size, filter_size])
            for i in range(0, filter_size):
                for j in range(0, filter_size):
                    initial_value[i, j] = (1 - np.abs(i / (S - C))) * (1 - np.abs(j / (S - C)))
            filters = initial_value
        else:
            filters = None

        biases = None
        if bias:
            biases = chainer.initializers.Normal(scale=0.01)
        return filters, biases
