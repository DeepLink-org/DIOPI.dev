import numpy as np

sar_config = {
    'mean': dict(
        name=["mean"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 3, 3, 3), (64,), (128, 64, 3, 3), (128,), (256, 128, 3, 3), (256, 256, 3, 3), (256,), (256, 128, 1, 1), (512, 256, 3, 3), (512, 512, 3, 3), (512,), (512, 256, 1, 1), (2048, 512), (2048,), (512, 512), (1, 512), (1,), (93, 512), (93, 1536), (93,), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'normal_': dict(
        name=["normal_"],
        no_output_ref=True,
        para=dict(
            size=[(64, 3, 3, 3), (128, 64, 3, 3), (256, 128, 3, 3), (256, 256, 3, 3), (256, 128, 1, 1), (512, 256, 3, 3), (512, 512, 3, 3), (512, 256, 1, 1)],
            mean=[0, 0, 0, 0, 0, 0, 0, 0],
            std=[0.05892556509887897, 0.04166666666666667, 0.029462782549439483, 0.029462782549439483, 0.08838834764831845, 0.020833333333333336, 0.020833333333333336, 0.0625],
        ),
    ),

    'fill_': dict(
        name=["fill_"],
        interface=["torch.Tensor"],
        para=dict(
            value=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64,), (128,), (256,), (512,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'uniform': dict(
        name=["uniform"],
        no_output_ref=True,
        para=dict(
            start=[0, 0, 0, 0],
            end=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64,), (128,), (256,), (512,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'randperm': dict(
        name=["randperm"],
        no_output_ref=True,
        para=dict(
            n=[93828],
        ),
    ),

    'sub': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3, 48, 160)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'div': dict(
        name=["div"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3, 48, 160)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'stack': dict(
        name=["stack"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160), (3, 48, 160)), ((3, 48, 160),)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
            seq_name='tensors',
        ),
    ),

    'conv2d': dict(
        name=["conv2d"],
        atol=1e-03,
        rtol=1e-03,
        para=dict(
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(384, 3, 48, 160), (384, 64, 48, 160), (384, 256, 24, 80), (384, 256, 12, 40), (384, 512, 6, 40), (1, 3, 48, 160), (1, 64, 48, 160), (1, 256, 24, 80), (1, 256, 12, 40), (1, 512, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 3, 3, 3), (128, 64, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (512, 512, 3, 3), (64, 3, 3, 3), (128, 64, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (512, 512, 3, 3)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (128,), (256,), (256,), (512,), (64,), (128,), (256,), (256,), (512,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'add': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'batch_norm': dict(
        name=["batch_norm"],
        atol=1e-03,
        rtol=1e-04,
        atol_half=1e-01,
        rtol_half=1e-02,
        para=dict(
            training=[True, True, True, True, True, False, False, False, False, False],
            momentum=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(384, 64, 48, 160), (384, 128, 48, 160), (384, 256, 24, 80), (384, 256, 12, 40), (384, 512, 6, 40), (1, 64, 48, 160), (1, 128, 48, 160), (1, 256, 24, 80), (1, 256, 12, 40), (1, 512, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_mean"],
                    "shape": [(64,), (128,), (256,), (256,), (512,), (64,), (128,), (256,), (256,), (512,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_var"],
                    "shape": [(64,), (128,), (256,), (256,), (512,), (64,), (128,), (256,), (256,), (512,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64,), (128,), (256,), (256,), (512,), (64,), (128,), (256,), (256,), (512,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (128,), (256,), (256,), (512,), (64,), (128,), (256,), (256,), (512,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        para=dict(
            inplace=[True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 64, 48, 160), (384, 128, 48, 160), (384, 256, 24, 80), (384, 256, 12, 40), (384, 512, 6, 40), (1, 64, 48, 160), (1, 128, 48, 160), (1, 256, 24, 80), (1, 256, 12, 40), (1, 512, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[2, 2, (2, 1), 2, 2, (2, 1)],
            stride=[2, 2, (2, 1), 2, 2, (2, 1)],
            padding=[0, 0, 0, 0, 0, 0],
            dilation=[1, 1, 1, 1, 1, 1],
            ceil_mode=[True, True, True, True, True, True],
            return_indices=[False, False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(384, 128, 48, 160), (384, 256, 24, 80), (384, 256, 12, 40), (1, 128, 48, 160), (1, 256, 24, 80), (1, 256, 12, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'conv2d_1': dict(
        name=["conv2d"],
        atol=1e-03,
        rtol=1e-03,
        para=dict(
            bias=[None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (1, 1), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(384, 128, 24, 80), (384, 256, 24, 80), (384, 128, 24, 80), (384, 256, 12, 40), (384, 256, 6, 40), (384, 512, 6, 40), (384, 256, 6, 40), (1, 128, 24, 80), (1, 256, 24, 80), (1, 128, 24, 80), (1, 256, 12, 40), (1, 256, 6, 40), (1, 512, 6, 40), (1, 256, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 128, 3, 3), (256, 256, 3, 3), (256, 128, 1, 1), (256, 256, 3, 3), (512, 256, 3, 3), (512, 512, 3, 3), (512, 256, 1, 1), (256, 128, 3, 3), (256, 256, 3, 3), (256, 128, 1, 1), (256, 256, 3, 3), (512, 256, 3, 3), (512, 512, 3, 3), (512, 256, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'add_1': dict(
        name=["add"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 256, 24, 80), (384, 256, 12, 40), (384, 512, 6, 40), (1, 256, 24, 80), (1, 256, 12, 40), (1, 512, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(384, 256, 24, 80), (384, 256, 12, 40), (384, 512, 6, 40), (1, 256, 24, 80), (1, 256, 12, 40), (1, 512, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'max_pool2d_1': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[(6, 1), (6, 1)],
            stride=[1, 1],
            padding=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(384, 512, 6, 40), (1, 512, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'permute': dict(
        name=["permute"],
        interface=["torch.Tensor"],
        para=dict(
            dims=[(0, 2, 1), (0, 2, 1), (0, 2, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 512, 40), (384, 29, 93), (1, 512, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'stack_1': dict(
        name=["stack"],
        interface=["torch"],
        para=dict(
            dim=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,)), ((512,),)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
            seq_name='tensors',
        ),
    ),

    'linear': dict(
        name=["linear"],
        atol=1e-03,
        rtol=1e-04,
        atol_half=1e-01,
        rtol_half=1e-02,
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(384, 512), (384, 31, 512), (384, 31, 6, 40, 512), (384, 31, 1536), (1, 512), (1, 31, 512), (1, 31, 6, 40, 512), (1, 31, 1536)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(512, 512), (512, 512), (1, 512), (93, 1536), (512, 512), (512, 512), (1, 512), (93, 1536)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(512,), (512,), (1,), (93,), (512,), (512,), (1,), (93,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'embedding': dict(
        name=["embedding"],
        para=dict(
            padding_idx=[91, 91],
            max_norm=[None, None],
            norm_type=[2.0, 2.0],
            scale_grad_by_freq=[False, False],
            sparse=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 30), (1,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(93, 512), (93, 512)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((384, 1, 512), (384, 30, 512)), ((1, 1, 512), (1, 30, 512))],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
            seq_name='tensors',
        ),
    ),

    'add_2': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 1, 512, 6, 40), (1, 1, 512, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(384, 31, 512, 1, 1), (1, 31, 512, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'tanh': dict(
        name=["tanh"],
        interface=["torch"],
        saved_args=dict(output=0),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(384, 31, 512, 6, 40), (1, 31, 512, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'permute_1': dict(
        name=["permute"],
        interface=["torch.Tensor"],
        para=dict(
            dims=[(0, 1, 3, 4, 2), (0, 1, 4, 2, 3), (0, 1, 3, 4, 2), (0, 1, 4, 2, 3)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 31, 512, 6, 40), (384, 31, 6, 40, 1), (1, 31, 512, 6, 40), (1, 31, 6, 40, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'masked_fill': dict(
        name=["masked_fill"],
        interface=["torch.Tensor"],
        para=dict(
            value=[float("-inf"), float("-inf")],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 31, 6, 40, 1), (1, 31, 6, 40, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["mask"],
                    "shape": [(384, 31, 6, 40, 1), (1, 31, 6, 40, 1)],
                    "dtype": [np.bool],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'softmax': dict(
        name=["softmax"],
        saved_args=dict(output=0),
        para=dict(
            dim=[-1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 31, 240), (1, 31, 240)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'mul': dict(
        name=["mul"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 1, 512, 6, 40), (1, 1, 512, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(384, 31, 1, 6, 40), (1, 31, 1, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'sum': dict(
        name=["sum"],
        interface=["torch"],
        para=dict(
            dim=[(3, 4), (3, 4)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 31, 512, 6, 40), (1, 31, 512, 6, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'expand': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(384, 31, 512), (-1, 30, -1), (1, 31, 512)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 1, 512), (1, 1, 512), (1, 1, 512)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'cat_1': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((384, 31, 512), (384, 31, 512), (384, 31, 512)), ((1, 31, 512), (1, 31, 512), (1, 31, 512))],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
            seq_name='tensors',
        ),
    ),

    'dropout': dict(
        name=["dropout"],
        no_output_ref=True,
        para=dict(
            p=[0.1, 0.1],
            training=[True, False],
            inplace=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 31, 93), (1, 31, 93)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        para=dict(
            ignore_index=[91],
            reduction=['mean'],
            label_smoothing=[0.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(384, 93, 29)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["target"],
                    "shape": [(384, 29)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'add_3': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'div_1': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'adam': dict(
        name=["adam"],
        interface=["CustomizedTest"],
        para=dict(
            step=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            amsgrad=[False for i in range(20)],
            beta1=[0.9 for i in range(20)],
            beta2=[0.999 for i in range(20)],
            lr=[0.001 for i in range(20)],
            weight_decay=[0 for i in range(20)],
            eps=[1e-08 for i in range(20)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(64, 3, 3, 3), (64,), (128, 64, 3, 3), (128,), (256, 128, 3, 3), (256, 256, 3, 3), (256,), (256, 128, 1, 1), (512, 256, 3, 3), (512, 512, 3, 3), (512,), (512, 256, 1, 1), (2048, 512), (2048,), (512, 512), (1, 512), (1,), (93, 512), (93, 1536), (93,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["exp_avg", "max_exp_avg_sq"],
                    "shape": [(64, 3, 3, 3), (64,), (128, 64, 3, 3), (128,), (256, 128, 3, 3), (256, 256, 3, 3), (256,), (256, 128, 1, 1), (512, 256, 3, 3), (512, 512, 3, 3), (512,), (512, 256, 1, 1), (2048, 512), (2048,), (512, 512), (1, 512), (1,), (93, 512), (93, 1536), (93,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["exp_avg_sq"],
                    "shape": [(64, 3, 3, 3), (64,), (128, 64, 3, 3), (128,), (256, 128, 3, 3), (256, 256, 3, 3), (256,), (256, 128, 1, 1), (512, 256, 3, 3), (512, 512, 3, 3), (512,), (512, 256, 1, 1), (2048, 512), (2048,), (512, 512), (1, 512), (1,), (93, 512), (93, 1536), (93,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
                },
            ],
        ),
    ),

    'arange': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            end=[2077],
        ),
    ),

    'max': dict(
        name=["max"],
        interface=["torch"],
        para=dict(
            dim=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 93)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'stack_2': dict(
        name=["stack"],
        interface=["torch"],
        para=dict(
            dim=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93), (1, 93))],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
            seq_name='tensors',
        ),
    ),

    'softmax_1': dict(
        name=["softmax"],
        saved_args=dict(output=0),
        para=dict(
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 30, 93)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'max_1': dict(
        name=["max"],
        interface=["torch"],
        para=dict(
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(30, 93)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
