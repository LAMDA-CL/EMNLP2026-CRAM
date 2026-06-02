# config/benchmarks/__init__.py
from .CoIN import COIN_TASKS
from .UCIT import UCIT_TASKS
from .TriGap import TRIGAP_TASKS

# 可以继续添加其他benchmark
# from .other import OTHER_TASKS

BENCHMARKS = {
    "coin": COIN_TASKS,
    "ucit": UCIT_TASKS,
    "trigap": TRIGAP_TASKS,
}

# 与 BENCHMARKS 中任务列表长度一致；
BENCHMARK_TASK_NUM: dict[str, int] = {
    name: len(tasks) for name, tasks in BENCHMARKS.items()}
