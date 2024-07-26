#[TODO] implement
from tools.scene_selector.tasks.open_vocabulary_detection import \
    OpenVocabularyDetection


class SelectScene():

    def __init__(self, tasks_config: list[dict]):
        tasks = SelectScene._init_tasks(tasks_config)

    def inference(self):
        pass

    def get_result(self):
        pass

    @staticmethod
    def _init_tasks(model, tasks_config: list[dict]):
        tasks = []
        for task_config in tasks_config:
            tasks.append(OpenVocabularyDetection())
        return tasks
