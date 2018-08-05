resnet_fine_tune_model = dict(
    batch_size=64,
    learning_rate=1e-4,
    resize_side_min=224,
    resize_side_max=512,
    train_path='CUB_200_2011/CUB_200_2011/splits/train.txt',
    valid_path='CUB_200_2011/CUB_200_2011/splits/valid.txt',
    test_path='CUB_200_2011/CUB_200_2011/splits/test.txt',

    train_epoch=20,
    train_step_per_epoch=300,
    test_step_per_epoch=5794 // 64 + 1,
    logging_step=50,
)

recurrent_attention_model = dict(
    batch_size=2,
    pth_size=64,

    cell_size=256,
    glimpse_output_size=512,

    variance=0.22,

    num_glimpses=5,
    glimpse_times=5,

    learning_rate=5e-4,
    learning_rate_decay_factor=0.97,
    min_learning_rate=1e-5,
    max_gradient_norm=5.0,

    resize_side_min=224,
    resize_side_max=320,
    train_path='CUB_200_2011/CUB_200_2011/splits/train.txt',
    valid_path='CUB_200_2011/CUB_200_2011/splits/valid.txt',
    test_path='CUB_200_2011/CUB_200_2011/splits/test.txt',

    train_epoch=20,
    train_step_per_epoch=6000 // 16 + 1,
    test_step_per_epoch=5794 // 16 + 1,
    valid_step_per_epoch=900 // 16 + 1,
    logging_step=50,
)
