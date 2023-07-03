model_list = ['resnet50', 'vgg16', 'resnet101', 'seresnet50', 'densenet', 'mobilenet_v2',
              'efficientnet', 'shufflenet_v2', 'repvgg', 'swin_transformer', 'vit', 'inceptionv3',
              'retinanet', 'faster_rcnn_r50', 'ssd300', 'yolov3', 'atss', 'fcos', 'mask_rcnn',
              'solo', 'centernet', 'cascade_rcnn', 'detr',
              'unet', 'upernet', 'pspnet', 'fcn', 'deeplabv3', 'deeplabv3plus',
              'sar', 'dbnet', 'stgcn', 'crnn', 'hrnet', 'deeppose', 'tsn', 'slowfast', 'llama']

model_op_list = {
    'resnet50': ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div', 'max_pool2d', 'softmax'],
    'resnet101': ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div', 'max_pool2d', 'softmax'],
    'seresnet50': ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'sigmoid', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div', 'max_pool2d', 'softmax'],
    'densenet': ['mean', 'normal_', 'fill_', 'randperm', 'flip', 'sub', 'div', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'cat', 'avg_pool2d', 'adaptive_avg_pool2d',
                 'linear', 'cross_entropy', 'sum', 'mul', 'sgd', 'arange'],
    'mobilenet_v2': ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'hardtanh', 'sum', 'mean', 'mul', 'div', 'softmax'],
    'efficientnet': ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'linspace', 'pad', 'sigmoid', 'sum', 'mean', 'mul', 'div', 'softmax'],
    'vgg16': ['sgd', 'randperm', 'conv2d', 'relu', 'batch_norm', 'max_pool2d', 'linear', 'dropout', 'cross_entropy', 'sum', 'mean', 'add', 'mul', 'div', 'softmax'],
    'shufflenet_v2': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'cat', 'transpose', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'sgd', 'mul', 'div', 'softmax'],
    'repvgg': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'sgd', 'mul', 'div', 'softmax'],
    'swin_transformer': ['linspace', 'arange', 'randperm', 'one_hot', 'mul', 'add', 'conv2d', 'transpose', 'layer_norm', 'dropout',
                         'pad', 'permute', 'linear', 'matmul', 'softmax', 'gelu', 'roll', 'sub', 'ne', 'masked_fill', 'eq', 'rand', 'div', 'floor', 'unfold', 'linear', 'adaptive_avg_pool2d', 'neg',
                         'log_softmax', 'sum', 'mean', 'norm', 'stack', 'reciprocal', 'clamp', 'adamw', 'addcmul', 'sqrt', 'addcdiv', 'uniform', 'im2col'],
    'vit': ['randperm', 'one_hot', 'mul', 'add', 'conv2d', 'transpose', 'expand', 'cat', 'dropout', 'layer_norm', 'linear',
            'permute', 'matmul', 'gelu', 'tanh', 'neg', 'log_softmax', 'sum', 'div', 'mean', 'norm', 'stack', 'reciprocal', 'clamp', 'adamw', 'addcmul', 'sqrt', 'addcdiv', 'softmax'],
    'inceptionv3': ['mean', 'uniform', 'erfinv', 'mul', 'add', 'clamp', 'fill_', 'randperm', 'flip', 'sub', 'div', 'conv2d', 'batch_norm', 'relu', 'max_pool2d', 'avg_pool2d', 'cat',
                    'adaptive_avg_pool2d', 'dropout', 'linear', 'cross_entropy', 'sum', 'sgd', 'arange', 'softmax', 'argmax'],
    # det
    'fcos': ['conv2d', 'batch_norm', 'relu', 'max_pool2d', 'add', 'interpolate', 'group_norm', 'mul', 'exp', 'arange', 'stack', 'expand', 'cat', 'sub', 'min', 'gt', 'max', 'ge', 'le', 'logical_and',
             'eq', 'split', 'permute', 'lt', 'nonzero', 'sum', 'div', 'sqrt', 'any', 'maximum', 'minimum', 'clamp', 'log', 'neg', 'ne', 'binary_cross_entropy_with_logits', 'mean', 'norm', 'reciprocal', 'sgd', 'sigmoid', 'sort'],
    'atss': ['conv2d', 'batch_norm', 'relu', 'max_pool2d', 'add', 'interpolate', 'group_norm', 'mul', 'arange', 'stack', 'logical_and', 'expand', 'cat', 'any', 'split', 'sum', 'sub', 'maximum',
             'minimum', 'clamp', 'div', 'pow', 'sqrt', 'topk', 'mean', 'std', 'ge', 'min', 'gt', 'transpose', 'max', 'ne', 'nonzero', 'unique', 'eq', 'permute', 'lt', 'exp', 'binary_cross_entropy_with_logits',
             'sgd', 'sigmoid', 'sort'],
    'ssd300': ['maximum', 'randperm', 'conv2d', 'relu', 'max_pool2d', 'pow', 'sum', 'sqrt', 'add', 'expand', 'mul', 'div', 'arange', 'stack', 'logical_and', 'cat', 'maximum', 'minimum', 'any', 'unique',
               'sub', 'max', 'min', 'clamp', 'ge', 'lt', 'gt', 'index_select', 'log', 'permute', 'cross_entropy', 'topk', 'abs', 'where', 'mean', 'eq', 'sgd', 'nonzero', 'sort', 'exp'],
    'retinanet': ['arange', 'randperm', 'conv2d', 'batch_norm', 'relu', 'max_pool2d', 'add', 'interpolate', 'orange', 'mul', 'stack', 'logical_and', 'expand', 'cat', 'any', 'sub', 'maximum',
                  'minimum', 'clamp', 'div', 'max', 'ge', 'lt', 'eq', 'gt', 'nonzero', 'unique', 'log', 'permute', 'sum', 'abs', 'mean', 'sgd', 'sigmoid', 'sort'],
    'faster_rcnn_r50': ['randperm', 'conv2d', 'batch_norm', 'relu', 'max_pool2d', 'add', 'interpolate', 'orange', 'mul', 'stack', 'logical_and', 'expand', 'cat', 'any', 'sub', 'maximum',
                        'minimum', 'clamp', 'div', 'max', 'ge', 'lt', 'eq', 'gt', 'nonzero', 'unique', 'log', 'permute', 'ne', 'binary_cross_entropy_with_logits', 'arange', 'sqrt',
                        'sum', 'abs', 'sigmoid', 'sort', 'exp', 'all', 'sort', 'log2', 'floor', 'linear', 'cross_entropy', 'topk', 'transpose', 'mean', 'sgd', 'split', 'softmax'],
    'yolov3': ['randperm', 'conv2d', 'batch_norm', 'leaky_relu', 'add', 'interpolate', 'cat', 'arange', 'mul', 'stack', 'div', 'floor', 'expand', 'sub', 'maximum', 'minimum',
               'clamp', 'max', 'ge', 'le', 'logical_and', 'bitwise_not', 'gt', 'eq', 'nonzero', 'unique', 'log', 'one_hot', 'permute', 'ne', 'binary_cross_entropy_with_logits', 'sum',
               'mse_loss', 'mean', 'norm', 'reciprocal', 'sgd', 'sigmoid', 'exp'],
    'mask_rcnn': ['conv2d', 'batch_norm', 'relu', 'max_pool2d', 'add', 'interpolate', 'arange', 'mul', 'stack', 'logical_and', 'expand', 'cat', 'any', 'sub', 'maximum', 'minimum', 'clamp', 'div', 'max', 'ge', 'lt', 'eq',
                  'gt', 'nonzero', 'unique', 'randperm', 'log', 'permute', 'ne', 'binary_cross_entropy_with_logits', 'sum', 'abs', 'sigmoid', 'sort', 'exp', 'all', 'sqrt', 'log2', 'floor', 'linear', 'cross_entropy',
                  'topk', 'transpose', 'conv_transpose2d', 'index_select', 'mean', 'sgd', 'split', 'softmax'],
    'solo': ['conv2d', 'batch_norm', 'relu', 'max_pool2d', 'add', 'interpolate', 'linspace', 'expand', 'cat', 'group_norm', 'sub', 'mul', 'sqrt', 'ge', 'le', 'logical_and', 'nonzero', 'sum', 'gt', 'arange', 'clamp',
             'div', 'permute', 'sigmoid', 'mean', 'sgd', 'eq', 'pow', 'cumsum'],
    'detr': ['conv2d', 'batch_norm', 'relu', 'max_pool2d', 'add', 'interpolate', 'sub', 'cumsum', 'div', 'mul', 'arange', 'pow', 'sin', 'cos', 'stack', 'cat', 'permute', 'linear', 'transpose', 'expand', 'masked_fill',
             'bmm', 'softmax', 'dropout', 'sum', 'layer_norm', 'sigmoid', 'neg', 'split', 'cdist', 'maximum', 'minimum', 'clamp', 'gt', 'nonzero', 'unique', 'eq', 'cross_entropy', 'any', 'mean', 'abs', 'norm', 'reciprocal', 'adamw', 'max', 'topk'],
    'cascade_rcnn': ['conv2d', 'batch_norm', 'relu', 'max_pool2d', 'add', 'interpolate', 'arange', 'mul', 'stack', 'logical_and', 'expand', 'cat', 'ge', 'lt', 'any', 'sub', 'maximum', 'minimum', 'clamp', 'div', 'max', 'eq', 'gt', 'nonzero',
                     'unique', 'randperm', 'log', 'permute', 'ne', 'binary_cross_entropy_with_logits', 'sum', 'abs', 'where', 'sigmoid', 'sort', 'exp', 'all', 'sqrt', 'log2', 'floor', 'linear', 'cross_entropy', 'topk', 'transpose', 'argmax', 'mean', 'sgd', 'split', 'softmax'],
    'centernet': ['conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'conv_transpose2d', 'sigmoid', 'mul', 'div', 'cat', 'sub', 'pow', 'lt', 'arange', 'neg', 'exp', 'max', 'gt', 'maximum', 'eq', 'sum', 'log', 'abs',
                  'mean', 'norm', 'stack', 'reciprocal', 'clamp', 'sgd', 'topk', 'permute', 'gather', 'remainder'],
    # seg
    'unet': ['dropout2d', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'interpolate', 'cat', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],
    'upernet': ['dropout2d', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'adaptive_avg_pool2d', 'interpolate', 'cat', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],
    'pspnet': ['dropout2d', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'adaptive_avg_pool2d', 'interpolate', 'cat', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],
    'fcn': ['dropout2d', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'cat', 'interpolate', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],
    'deeplabv3': ['dropout2d', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'adaptive_avg_pool2d', 'interpolate', 'cat', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],
    'deeplabv3plus': ['dropout2d', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'adaptive_avg_pool2d', 'interpolate', 'cat', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],
    # other
    'dbnet': ['mean', 'randperm', 'sub', 'div', 'stack', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'interpolate', 'cat', 'conv_transpose2d', 'sigmoid', 'mul', 'exp', 'reciprocal', 'nonzero', 'bitwise_not', 'sum', 'max', 'le',
              'min', 'ge', 'binary_cross_entropy_with_logits', 'topk', 'smooth_l1_loss', 'sgd'],
    'stgcn': ['randperm', 'permute', 'add', 'batch_norm', 'mul', 'conv2d', 'relu', 'dropout', 'adaptive_avg_pool2d', 'mean', 'cross_entropy', 'div', 'sgd'],
    'hrnet': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'interpolate', 'sum', 'gt', 'sub', 'pow', 'expand', 'mul', 'mean', 'stack', 'permute', 'neg', 'exp', 'div', 'adam', 'addcmul', 'sqrt', 'addcdiv'],
    'sar': ['mean', 'fill_', 'uniform', 'randperm', 'sub', 'div', 'stack', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'permute', 'linear', 'embedding', 'cat', 'tanh', 'masked_fill', 'softmax', 'mul',
            'sum', 'expand', 'dropout', 'cross_entropy', 'adam', 'addcmul', 'sqrt', 'addcdiv', 'arange', 'max'],
    'crnn': ['mean', 'fill_', 'uniform', 'randperm', 'sub', 'div', 'stack', 'conv2d', 'relu', 'max_pool2d', 'add', 'batch_norm', 'permute', 'linear', 'adadelta', 'mul', 'addcmul', 'sqrt', 'arange', 'softmax'],
    'tsn': ['randperm', 'conv2d', 'batch_norm', 'relu', 'max_pool2d', 'add', 'adaptive_avg_pool2d', 'mean', 'dropout', 'linear', 'cross_entropy', 'mul', 'norm', 'stack', 'div', 'reciprocal', 'clamp', 'sgd'],
    'slowfast': ['randperm', 'interpolate', 'add', 'batch_norm', 'relu', 'cat', 'dropout', 'linear', 'cross_entropy', 'mul', 'mean', 'norm', 'stack', 'div', 'reciprocal', 'clamp', 'sgd',
                 'conv3d', 'max_pool3d', 'adaptive_avg_pool3d'],
    'deeppose': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'adaptive_avg_pool2d', 'linear', 'sigmoid', 'sub', 'div', 'mul', 'leaky_relu', 'tanh', 'neg', 'exp', 'sum', 'expand', 'eq',
                 'all', 'cholesky_ex', 'permute', 'triangular_solve', 'pow', 'transpose', 'log', 'abs', 'mean', 'adam'],
    'llama': ['ne', 'embedding', 'triu', 'pow', 'mean', 'add', 'rsqrt', 'mul', 'linear', 'view_as_complex', 'view_as_real', 'transpose', 'matmul', 'div', 'softmax', 'silu', 'sort', 'cumsum', 'sub', 'gt', 'sum',
              'multinomial', 'gather', 'where'],
    'llama_train': ['normal_', 'uniform', 'arange', 'div', 'pow', 'reciprocal', 'mul', 'embedding', 'dropout', 'sub', 'mean', 'add', 'rsqrt', 'linear', 'transpose', 'view_as_complex', 'view_as_real', 'matmul', 'softmax', 'gt', 'log_softmax', 'argmax', 'eq', 'sum', 'addcmul', 'sqrt', 'addcdiv']
}
