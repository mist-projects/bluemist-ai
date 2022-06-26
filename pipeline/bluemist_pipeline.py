pipeline_steps = {}
pipelines = {}


def add_pipeline_step(estimator_name, pipeline_step):
    if estimator_name not in pipeline_steps:
        pipeline_steps[estimator_name] = []

    if estimator_name in pipeline_steps:
        steps = pipeline_steps[estimator_name]
        steps.append(pipeline_step)
        print(steps)
        return steps


def save_pipeline(estimator_name, pipeline):
    pipelines[estimator_name] = pipeline
    print(pipelines)
