import numpy as np
from decimal import Decimal

import logging
import matplotlib

logger = logging.getLogger(__name__)


def savefig(fig, savename):
    fig.tight_layout()
    for extension, kwargs in [('png', dict(dpi=600)), ('svg', {})]:
        savepath = f"{savename}.{extension}"
        logger.info(f"Saving to {savepath}")
        fig.savefig(savepath, **kwargs, bbox_inches='tight')


@matplotlib.ticker.FuncFormatter
def score_formatter(score, pos):
    if 0 <= score < 1:
        mod = Decimal(f"{score}") % Decimal(f"{.1}")
        assert mod < .001 or mod > .099  # ensure we don't display rounding errors
        return f"{score:.1f}"[1:]  # strip "0" in front of e.g. "0.2"
    elif np.abs(score - 1) < .001:
        return "1."
    elif score > 1:
        return ""
    else:
        return f"{score}"
