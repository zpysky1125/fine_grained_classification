resnet_fine_tune_model = dict(
    batch_size=64,
    learning_rate=3e-4,
    resize_side_min=280,
    resize_side_max=384,

    drop1=0.3,
    drop2=0.3,
    internal_size=512,
    train_method='Adam',
    parse_mode='crop',

    train_path='CUB_200_2011/CUB_200_2011/splits/train.txt',
    valid_path='CUB_200_2011/CUB_200_2011/splits/valid.txt',
    test_path='CUB_200_2011/CUB_200_2011/splits/test.txt',

    train_epoch=300,
    train_step_per_epoch=5094 // 64 + 1,
    test_step_per_epoch=5794 // 64 + 1,
    valid_step_per_epoch=900 // 64 + 1,
    logging_step=50,
)

recurrent_attention_model = dict(
    batch_size=64,
    multi_batch_size=2,
    pth_size=128,

    cell_size=256,
    glimpse_output_size=512,

    variance=0.22,

    num_glimpses=5,
    glimpse_times=5,

    drop1=0.3,
    drop2=0.3,

    train_method='Adam',
    crop_or_mask='crop',
    reinforce_mode='baseline',
    result_mode='single',

    learning_rate=5e-4,
    learning_rate_decay_factor=0.99,
    min_learning_rate=3e-5,
    max_gradient_norm=5.0,

    resize_side_min=280,
    resize_side_max=384,
    train_path='CUB_200_2011/CUB_200_2011/splits/train.txt',
    valid_path='CUB_200_2011/CUB_200_2011/splits/valid.txt',
    test_path='CUB_200_2011/CUB_200_2011/splits/test.txt',

    train_epoch=300,
    train_step_per_epoch=5094 // 64 + 1,
    test_step_per_epoch=5794 // 64 + 1,
    valid_step_per_epoch=900 // 64 + 1,
    test_multi_step_per_epoch=5794 // 16 + 1,
    logging_step=50,
)


recurrent_attention_model_2 = dict(
    batch_size=64,
    multi_batch_size=2,
    pth_size=128,

    internal1=512,
    internal2=512,
    internal3=512,

    variance=0.22,

    drop1=0.3,
    drop2=0.3,

    train_method='Adam',
    crop_or_mask='crop',
    reinforce_mode='baseline',

    learning_rate=5e-4,
    min_learning_rate=3e-5,

    resize_side_min=280,
    resize_side_max=384,
    train_path='CUB_200_2011/CUB_200_2011/splits/train.txt',
    valid_path='CUB_200_2011/CUB_200_2011/splits/valid.txt',
    test_path='CUB_200_2011/CUB_200_2011/splits/test.txt',

    train_epoch=300,
    logging_step=50,
)
