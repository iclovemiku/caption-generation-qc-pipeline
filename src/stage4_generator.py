"""
Stage 4: 下游任务分发（SQA-CoT / AQA-CoT / 多轮对话）。

职责：按 sqa_compatibility 与策略分流；生成单轮共情 CoT 与多轮 JSONL。
"""


def run_stage4_sqa_aqa(*args, **kwargs):
    """TODO: 单轮 SQA / AQA 生产线"""
    raise NotImplementedError


def run_stage4_multi_turn(*args, **kwargs):
    """TODO: 多轮对话生产线"""
    raise NotImplementedError
