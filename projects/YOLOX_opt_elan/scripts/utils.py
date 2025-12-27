import re


def yolox_s_opt_to_mmdet_key(key):
    """
    Convert the weights of yolox_s_opt to the weights of MMDetection.
    """
    x = key
    x = re.sub(r"backbone.backbone", r"backbone", x)
    x = re.sub(r"stem.down1", r"stem.conv.0", x)
    x = re.sub(r"stem.conv2", r"stem.conv.1", x)
    x = re.sub(r"(?<=darknet)[1-9]", lambda exp: str(int(exp.group(0)) - 1), x)
    x = re.sub(r"dark(?=[0-9].[0-9].)", r"stage", x)
    x = re.sub(r"(?<=stage)[1-9]", lambda exp: str(int(exp.group(0)) - 1), x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.1|4.2).)conv1", r"main_conv", x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.1|4.2).)conv2", r"short_conv", x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.1|4.2).)conv3", r"final_conv", x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.1|4.2).)m(?=\.)", r"blocks", x)
    x = re.sub(r"backbone.lateral_conv(?=[0-9])", r"neck.reduce_layers.", x)
    x = re.sub(r"(backbone.C3_(?=p[3-4]))", r"neck.top_down_blocks.", x)
    x = re.sub(r"p4.conv1", r"0.main_conv", x)
    x = re.sub(r"p4.conv2", r"0.short_conv", x)
    x = re.sub(r"p4.conv3", r"0.final_conv", x)
    x = re.sub(r"p4.m", r"0.blocks", x)
    x = re.sub(r"backbone.reduce_conv(?=[0-9]\.)", r"neck.reduce_layers.", x)
    x = re.sub(r"p3.conv1", r"1.main_conv", x)
    x = re.sub(r"p3.conv2", r"1.short_conv", x)
    x = re.sub(r"p3.conv3", r"1.final_conv", x)
    x = re.sub(r"p3.m", r"1.blocks", x)
    x = re.sub(r"backbone.bu_conv2", r"neck.downsamples.0", x)
    x = re.sub(r"(backbone.C3_(?=n[3-4]))", r"neck.bottom_up_blocks.", x)
    x = re.sub(r"n3.conv1", r"0.main_conv", x)
    x = re.sub(r"n3.conv2", r"0.short_conv", x)
    x = re.sub(r"n3.conv3", r"0.final_conv", x)
    x = re.sub(r"n3.m", r"0.blocks", x)
    x = re.sub(r"backbone.bu_conv1", r"neck.downsamples.1", x)
    x = re.sub(r"n4.conv1", r"1.main_conv", x)
    x = re.sub(r"n4.conv2", r"1.short_conv", x)
    x = re.sub(r"n4.conv3", r"1.final_conv", x)
    x = re.sub(r"n4.m", r"1.blocks", x)
    x = re.sub(r"head.cls_convs", r"bbox_head.multi_level_cls_convs", x)
    x = re.sub(r"head.reg_convs", r"bbox_head.multi_level_reg_convs", x)
    x = re.sub(r"head.cls_preds", r"bbox_head.multi_level_conv_cls", x)
    x = re.sub(r"head.reg_preds", r"bbox_head.multi_level_conv_reg", x)
    x = re.sub(r"head.obj_preds", r"bbox_head.multi_level_conv_obj", x)

    x = re.sub(r"head.stems", r"neck.out_convs", x)

    return x
