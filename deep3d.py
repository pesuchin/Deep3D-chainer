import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import cuda, optimizers, serializers, Variable
import math
import six
from chainer.links.caffe.protobuf3 import caffe_pb2 as caffe_pb


def select(masks, left_image, left_shift=16):
    '''
    assumes inputs:
        masks, shape N, H, W, S
        left_image, shape N, H, W, C
    returns
        right_image, shape N, H, W, C
    '''

    _, H, W, S = masks.shape
    padded = F.pad(left_image, [[0,0],[0,0],[0,0],[left_shift, left_shift]], mode='constant')
    
    for s in np.arange(S):
        mask_slice = masks[:, :, :, s]
        mask_slice = F.expand_dims(mask_slice, axis=1)
        pad_slice = F.get_item(padded, (slice(None), slice(None), slice(0, H), slice(s, W+s)))
        if s == 0:
            pred = F.expand_dims(F.scale(pad_slice, mask_slice, axis=0), axis=4)
        else:
            tmp = F.expand_dims(F.scale(pad_slice, mask_slice, axis=0), axis=4)
            pred = F.concat([pred, tmp], axis=4)
    return F.sum(pred, axis=4)    

# 参考: 
# https://github.com/piiswrong/deep3d/blob/e9433221662001717cfafe89c5f8a7e3b26fe1ee/sym.py
# https://github.com/JustinTTL/Deep3D_TF/blob/master/Deep3D_Final.py
class Deep3D(chainer.Chain):
    def __init__(self, batchsize, size):
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
            self.deconv1 = L.Deconvolution2D(33, 33, ksize=(1, 1), stride=(1, 1), pad=(0, 0), initialW=W, initial_bias=bias)
            scale *= 2
            W, bias = self.get_initial_deconv_value(2*scale, 33, 33)
            self.deconv2 = L.Deconvolution2D(33, 33, ksize=(2*scale, 2*scale), stride=(scale, scale), pad=(scale//2, scale//2), initialW=W, initial_bias=bias)
            scale *= 2
            W, bias = self.get_initial_deconv_value(2*scale, 33, 33)
            self.deconv3 = L.Deconvolution2D(33, 33, ksize=(2*scale, 2*scale), stride=(scale, scale), pad=(scale//2, scale//2), initialW=W, initial_bias=bias)
            scale *= 2
            W, bias = self.get_initial_deconv_value(2*scale, 33, 33)
            self.deconv4 = L.Deconvolution2D(33, 33, ksize=(2*scale, 2*scale), stride=(scale, scale), pad=(scale//2, scale//2), initialW=W, initial_bias=bias)
            scale *= 2
            W, bias = self.get_initial_deconv_value(2*scale, 33, 33)
            self.deconv5 = L.Deconvolution2D(33, 33, ksize=(2*scale, 2*scale), stride=(scale, scale), pad=(scale//2, scale//2), initialW=W, initial_bias=bias)
            scale = 2
            W, bias = self.get_initial_deconv_value(2*scale, 33, 33)
            self.up_deconv_layer = L.Deconvolution2D(33, 33, ksize=(2*scale, 2*scale), stride=(scale, scale), pad=(scale//2, scale//2), initialW=W, initial_bias=bias)
            self.up_conv_layer = L.Convolution2D(33, 33, ksize=(3, 3), stride=1, pad=(1, 1))


    def __call__(self, left, right, original_left):
        masked_left_images = self.predict(left, original_left)
        loss = F.mean_absolute_error(masked_left_images, right)
        chainer.report({'loss': loss}, self)
        return loss

    def predict(self, left, original_left):
        mask = self.get_mask_value(left)
        masked_left_images = select(mask, original_left)
        return masked_left_images

    def get_mask_value(self, left):
        bgr = left
        #bgr = chainer.links.model.vision.vgg.prepare(left)

        # VGG
        vgg_result = self.vgg_layers(bgr, layers=['pool1', 'pool2', 'pool3', 'pool4', 'pool5'])
        vgg_fc6 = F.dropout(F.relu(self.fc6(vgg_result['pool5'])))
        vgg_fc7 = F.dropout(F.relu(self.fc7(vgg_fc6)))
        vgg_fc8 = F.dropout(F.relu(self.fc8(vgg_fc7)))
        
        #-------branch 1-----
        bn_pool1 = self.batch_norm1(vgg_result['pool1'])
        branch1_1 = F.relu(self.branch_conv1(bn_pool1))
        branch1_2 = self.deconv1(branch1_1)

        #-------branch 2-----
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

        #Initializing to bilinear interpolation
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
