# This network was used in Table 1 CIFAR10 and CIFAR100 results.


"""
-input_shape: shape of the image; in CIFAR10 each image is a 3x32x32 tensor
-layers: information of the convnet
    -layer: type of the sub model
    -args: sub model params
    -bn: batch normalization
    -act: nonlinear activation
-local_task_idx:
    layer 0 to layer (local_task_idx[0]-1) computes feat map for feat maps (C_{phi} in paper)
    layer local_task_idx[0] to last layer computes feat map for global feats based on local feats
"""

"""
Encoder does not extract local features, it just return a feature map (4D tensor) and a global feature (2D tensor)
local dim is computed based on concatenating the feature map and the global feature
"""

basic32x32 = dict(
    # Module=Convnet,
    input_shape=(3, 32, 32),
    layers=[dict(layer='conv', args=(64, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(128, 4, 2, 1), bn=True, act='ReLU'),  # <- output feature map
            dict(layer='conv', args=(256, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(512, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='flatten'),
            dict(layer='linear', args=(64,), bn=False, act=None)],  # <- output global feature
    # local_task_idx=(2, 3),
    feature_idx = 2
    # classifier_idx=dict(conv=2, fc=4, glob=-1)
)

basic64x64 = dict(
    # Module=Convnet,
    input_shape=(3, 64, 64),
    layers=[dict(layer='conv', args=(64, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(128, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(256, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(512, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='flatten'),
            dict(layer='linear', args=(1024,), bn=True, act='ReLU')],
    local_task_idx=(2, -1),
    # classifier_idx=dict(conv=3, fc=5, glob=-1)
)