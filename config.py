resnet_fine_tune_model = dict(
    batch_size=4,
    learning_rate=1e-4,
    resize_side_min=224,
    resize_side_max=512,
    train_path='CUB_200_2011/CUB_200_2011/splits/train.txt',
    valid_path='CUB_200_2011/CUB_200_2011/splits/valid.txt',
    test_path='CUB_200_2011/CUB_200_2011/splits/test.txt',

    train_epoch=20,
    train_step_per_epoch=300,
    test_step_per_epoch=182,
    logging_step=50,
)
