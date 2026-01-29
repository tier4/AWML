weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = None  # train process will init a random seed and record
save_path = "exp/default"
num_worker = 16  # total worker in all gpu
batch_size = 16  # total batch size in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
epoch = 100  # total epoch, data loop = epoch // eval_epoch
eval_epoch = 100  # sche total eval & checkpoint epoch
clip_grad = None  # disable with None, enable with a float

sync_bn = False
enable_amp = False
amp_dtype = "float16"
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False

enable_wandb = True
wandb_project = "LitePT"  # wandb project name
wandb_key = None  # wandb token, default is None. If None, login with `wandb login` in your terminal

mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=3),
    dict(type="PreciseEvaluator", test_last=False),
]

# Trainer
train = dict(type="DefaultTrainer")

# Tester
test = dict(type="SemSegTester", verbose=True)

# visualization
class_colors = {
    0: (0, 0, 0),  # Black
    1: (0, 255, 255),  # Cyan / Aqua
    2: (233, 233, 229),  # Light Beige/Grey
    3: (110, 110, 110),  # Dark Grey
    4: (0, 175, 0),  # Green
    5: (232, 35, 244),  # Magenta
    6: (255, 158, 0),  # Orange
    7: (0, 0, 230),  # Blue
    8: (255, 0, 0),  # Bright Red
    9: (255, 127, 80),  # Coral
    10: (160, 60, 60),  # Brownish Red
    11: (255, 140, 0),  # Dark Orange
    12: (255, 215, 0),  # Gold
    13: (220, 20, 60),  # Crimson
    14: (255, 61, 99),  # Reddish Pink
    15: (230, 230, 0),  # Yellow
    16: (128, 128, 128),  # Medium Grey
    17: (255, 0, 255),  # Bright Fuchsia
    18: (47, 79, 79),  # Dark Slate Grey
    19: (139, 69, 19),  # Saddle Brown
    20: (218, 165, 32),  # Goldenrod
    21: (128, 0, 128),  # Purple
    22: (135, 206, 235),  # Sky Blue
    23: (0, 0, 128),  # Navy Blue
    24: (170, 170, 170),  # Light Grey
    25: (205, 133, 63),  # Peru
    26: (255, 99, 71),  # Tomato Red
    27: (240, 240, 240),  # Ghost White
    28: (50, 50, 50),  # Very Dark Grey
}
