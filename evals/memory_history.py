import sys
from textwrap import dedent
from typing import Final

from attr import dataclass
from langfuse import Evaluation, Langfuse

from ask.core.agent import AgentASK
from eval.agent import create_config, make_llm_config
from eval.data import serialize_config, task_executor_agent
from eval.instrumentation import setup_instrumentation

langfuse: Final[Langfuse] = setup_instrumentation()


@dataclass
class Metadata:
    instructions: str


DATASET: Final[str] = "memory history simple"
try:
    dataset = langfuse.get_dataset(name=DATASET)
except Exception:
    langfuse.create_dataset(
        name=DATASET,
        description="memory history simple question. no tools",
        metadata=Metadata(
            instructions=dedent(
                """
                    Must return an one line answer.
                """,
            ),
        ),
    )
    input = [
        "hi",
        "who are you?",
        "who built you?",
        "1 + 100",
        "1 - 100",
        "1 * 100",
        "1 / 100",
        "what is the capital of France?",
    ]
    # reverse the input to have the same order as before
    for i in reversed(input):
        langfuse.create_dataset_item(
            dataset_name=DATASET,
            input=i,
        )

dataset = langfuse.get_dataset(name=DATASET)
if dataset.metadata and isinstance(dataset.metadata, dict):
    dataset.metadata = Metadata(**dataset.metadata)


def _accuracy_evaluator(*, input, output, expected_output, size=1):
    try:
        # check if output is a 1 line text
        lines = output.strip().split("\n")
        return Evaluation(name="accuracy", value=1 if len(lines) == size else 0)
    except Exception as e:
        print(f"Output is not valid JSON {e}", file=sys.stderr)
        return Evaluation(name="accuracy", value=0.0)


def run_experiment(model: str, base_url: str, session_id: str):
    config = create_config(
        llm=make_llm_config(model=model, base_url=base_url),
        instructions=dataset.metadata.instructions if dataset.metadata else "",
    )
    agent = AgentASK.create_from_config(config=config)
    result = dataset.run_experiment(
        name="simple 1",
        run_name=config.llm.model,
        description="simple 1 line answer",
        task=lambda *, item, **kwargs: task_executor_agent(
            agent=agent,
            item=item,
            callback=lambda: langfuse.update_current_trace(
                session_id=session_id, user_id="eval", tags=[config.llm.model]
            ),
        ),
        evaluators=[
            lambda *, input, output, expected_output, **kwargs: _accuracy_evaluator(
                input=input,
                output=output,
                expected_output=expected_output,
                size=1,  # expect 1 line answer
            )
        ],
        max_concurrency=1,  # Limit concurrency to 1 to avoid race timing issues
        metadata={
            "config": serialize_config(config),
        },
    )
    print(result.format(), file=sys.stderr)
