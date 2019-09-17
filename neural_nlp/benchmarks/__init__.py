from .neural import benchmark_pool as _neural_pool
from .performance import benchmark_pool as _performance_pool

benchmark_pool = {
    **_neural_pool,
    **_performance_pool,
}
