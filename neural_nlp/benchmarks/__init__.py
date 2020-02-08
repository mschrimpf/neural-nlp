from .neural import benchmark_pool as _neural_pool
from .performance import benchmark_pool as _performance_pool
from .glue import benchmark_pool as _glue_pool

benchmark_pool = {
    **_neural_pool,
    **_performance_pool,
    **_glue_pool,
}
