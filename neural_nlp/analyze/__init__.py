import logging

logger = logging.getLogger(__name__)


def savefig(fig, savename):
    fig.tight_layout()
    for extension, kwargs in [('png', dict(dpi=600)), ('svg', {})]:
        savepath = f"{savename}.{extension}"
        logger.info(f"Saving to {savepath}")
        fig.savefig(savepath, **kwargs, bbox_inches='tight')
