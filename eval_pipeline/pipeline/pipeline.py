from typing import Any, Dict, List, Optional, Text

from tfx import v1 as tfx
from ml_metadata.proto import metadata_store_pb2
from components import FileListGen
from components import BatchPredictionGen
from components import PerformanceEvaluator
from components import SpanPreparator
from components import PipelineTrigger


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
) -> tfx.dsl.Pipeline:
    # Generate a file list for batch preditions.
    # More details on the structure of this file here:
    # https://bit.ly/3BzfHVu.
    filelist_gen = FileListGen(
        project=project_id,
        gcs_source_bucket=data_gcs_bucket,
        gcs_source_prefix=data_gcs_prefix,
    ).with_id("filelist_gen")

    # Submit a batch prediction job.
    batch_pred_component = BatchPredictionGen(
        project=project_id,
        location=region,
        job_display_name=job_display_name,
        model_resource_name=model_resource_name,
        gcs_source=filelist_gen.outputs["outpath"],
        gcs_destination=f"gs://{batch_job_gcs}/results/",
        accelerator_count=0,
        accelerator_type=None,
    ).with_id("bulk_inferer_vertex")
    batch_pred_component.add_upstream_node(filelist_gen)

    # Evaluate the performance of the predictions.
    # In a real-world project, this evaluation takes place
    # separately, typically with the help of domain experts.
    final_gcs_destination = f"gs://{batch_job_gcs}/results/"
    evaluator = PerformanceEvaluator(
        gcs_destination=f'gs://{final_gcs_destination.split("/")[2]}',
        local_directory=final_gcs_destination.split("/")[-2],
        threshold=threshold,
    ).with_id("batch_prediction_evaluator")
    evaluator.add_upstream_node(batch_pred_component)

    span_preparator = SpanPreparator(
        is_retrain=evaluator.outputs["trigger_pipeline"],
        gcs_source_bucket=data_gcs_bucket,
        gcs_source_prefix=data_gcs_prefix,
        gcs_destination_bucket=data_gcs_destination,
    ).with_id("span_preparator")
    span_preparator.add_upstream_node(evaluator)

    trigger = PipelineTrigger(
        is_retrain=evaluator.outputs["trigger_pipeline"],
        latest_span_id=span_preparator.outputs["latest_span_id"],
        pipeline_spec_path=training_pipeline_spec,
        project_id=project_id,
        region=region,
    ).with_id("training_pipeline_trigger")
    trigger.add_upstream_node(span_preparator)

    components = [
        filelist_gen,
        batch_pred_component,
        evaluator,
        span_preparator,
        trigger,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata_connection_config,
    )
