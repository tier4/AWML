from mmpretrain.registry import METRICS
from mmpretrain.evaluation.metrics import MultiLabelMetric


@METRICS.register_module()
class TLRClassificationMetric(MultiLabelMetric):

    def __init__(self, class_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names = class_names

    def compute_metrics(self, *args, **kwargs):
        self.average = None

        results = super().compute_metrics(*args, **kwargs)

        # Header
        header = f"| {'Class Name':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Counts':<10} |"
        separator = "-" * len(header)

        print("\nClass-wise Metrics:")
        print(separator)
        print(header)
        print(separator)

        # Rows
        for i, class_name in enumerate(self.class_names):
            precision = results['precision_top1_classwise'][i]
            recall = results['recall_top1_classwise'][i]
            f1_score = results['f1-score_top1_classwise'][i]
            support = results['support_top1_classwise'][i]
            print(
                f"| {class_name:<20} | {precision:<10.2f} | {recall:<10.2f} | {f1_score:<10.2f} | {support:<10} |"
            )

        print(separator)

        self.average = "macro"
        results_macro = [
            f"{k}: {v:<10.2f}"
            for k, v in super().compute_metrics(*args, **kwargs).items()
        ]

        print(f"Overall results: ", ", ".join(results_macro))
        return results
