custom_imports = dict(
    imports=["tools.rosbag.rosbag"], allow_failed_imports=False)

preprocess = Preprocess(image_crop=[0.1, 0.1, 0.9, 0.9])
selector = Selector(
    min_crop_time=5,  #[sec]
)

open_vocabulary_selector = [
    OpenVocabulary(
        task="bicycle",
        text="bicycle",
        confidence=0.5,
        object_num_threshold=1,
        scene_threshold=10),
    OpenVocabulary(
        task="traffic cone",
        text="traffic cone",
        confidence=0.3,
        object_num_threshold=1,
        scene_threshold=10),
]

#open_vocabulary_segmentation_selector = [
#  OpenVocabulary2dSeg(
#    task="pedestrian on road",
#    open_vocabulary=OpenVocabulary(text="pedestrian", confidence=0.2, object_num_threshold=1, scene_threshold=10),
#    segmentation=Segmentation(background="road"),
#  )
#  OpenVocabulary2dSeg(
#    task="traffic cone on road",
#    open_vocabulary=OpenVocabulary(text="traffic cone", confidence=0.2, object_num_threshold=1, scene_threshold=10),
#    segmentation=Segmentation(background="road"),
#  )
#]
