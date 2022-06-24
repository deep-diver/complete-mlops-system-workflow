from typing import List, Optional

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx

from ml_metadata.proto import metadata_store_pb2
from tfx.proto import example_gen_pb2

def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    preprocessing_fn: str,
    run_fn: str,
    train_args: tfx.proto.TrainArgs,
    eval_args: tfx.proto.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: str,
    schema_path: Optional[str] = None,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[str]] = None,
) -> tfx.dsl.Pipeline:
  components = []

  input_config = example_gen_pb2.Input(splits=[
      example_gen_pb2.Input.Split(name='train', pattern='train/*'),
      example_gen_pb2.Input.Split(name='eval', pattern='test/*')
  ])
  example_gen = tfx.components.ImportExampleGen(input_base=data_path, input_config=input_config)
  components.append(example_gen)

  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])
  components.append(statistics_gen)

  if schema_path is None:
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'])
    components.append(schema_gen)
  else:
    schema_gen = tfx.components.ImportSchemaGen(schema_file=schema_path)
    components.append(schema_gen)

#   example_validator = tfx.components.ExampleValidator(  
#       statistics=statistics_gen.outputs['statistics'],
#       schema=schema_gen.outputs['schema'])
#   components.append(example_validator)

  transform = tfx.components.Transform(  
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      preprocessing_fn=preprocessing_fn)
  components.append(transform)

  trainer = tfx.components.Trainer(
      run_fn=run_fn,
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      train_args=train_args,
      eval_args=eval_args)
  components.append(trainer)

  # Get the latest blessed model for model validation.
  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')
  # TODO(step 5): Uncomment here to add Resolver to the pipeline.
  components.append(model_resolver)

#   # Uses TFMA to compute a evaluation statistics over features of a model and
#   # perform quality validation of a candidate model (compared to a baseline).
#   eval_config = tfma.EvalConfig(
#       model_specs=[
#           tfma.ModelSpec(
#               signature_name='serving_default',
#               label_key=features.LABEL_KEY,
#               # Use transformed label key if Transform is used.
#               # label_key=features.transformed_name(features.LABEL_KEY),
#               preprocessing_function_names=['transform_features'])
#       ],
#       slicing_specs=[tfma.SlicingSpec()],
#       metrics_specs=[
#           tfma.MetricsSpec(metrics=[
#               tfma.MetricConfig(
#                   class_name='SparseCategoricalAccuracy',
#                   threshold=tfma.MetricThreshold(
#                       value_threshold=tfma.GenericValueThreshold(
#                           lower_bound={'value': eval_accuracy_threshold}),
#                       change_threshold=tfma.GenericChangeThreshold(
#                           direction=tfma.MetricDirection.HIGHER_IS_BETTER,
#                           absolute={'value': -1e-10})))
#           ])
#       ])
  evaluator = tfx.components.Evaluator(  # pylint: disable=unused-variable
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'])
      # Change threshold will be ignored if there is no baseline (first run).
    #   eval_config=eval_config)
  # TODO(step 5): Uncomment here to add Evaluator to the pipeline.
#   components.append(evaluator)

  # Pushes the model to a file destination if check passed.
  pusher = tfx.components.Pusher(  # pylint: disable=unused-variable
      model=trainer.outputs['model'],
    #   model_blessing=evaluator.outputs['blessing'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=serving_model_dir)))
  # TODO(step 5): Uncomment here to add Pusher to the pipeline.
  components.append(pusher)

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      # Change this value to control caching of execution results. Default value
      # is `False`.
      enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )
