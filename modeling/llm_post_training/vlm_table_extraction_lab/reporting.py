"""Small grouped W&B view; retain all diagnostic metrics in local reports."""

GROUPED_METRICS = {
    "quality/rd_similarity": "official_rd_similarity_raw",
    "quality/cell_f1": "cell_f1",
    "quality/numeric_f1": "numeric_f1",
    "quality/table_exact": "table_exact",
    "structure/row_count_exact": "row_count_exact",
    "structure/column_count_exact": "column_count_exact",
    "structure/span_f1": "span_f1",
    "structure/exact": "structure_exact",
    "runtime/format_pass_rate": "parse_success",
    "runtime/truncation_rate": "truncated",
    "runtime/prediction_coverage": "prediction_present",
    "runtime/mean_latency_seconds": "latency_seconds",
    "runtime/wall_seconds": "wall_seconds",
}


def grouped_metrics(metrics):
    result = {
        name: metrics["eval/" + key]
        for name, key in GROUPED_METRICS.items()
        if "eval/" + key in metrics
    }
    if "eval/input_tokens_total" in metrics and "eval/output_tokens_total" in metrics:
        result["runtime/total_tokens"] = (
            metrics["eval/input_tokens_total"] + metrics["eval/output_tokens_total"]
        )
    return result
