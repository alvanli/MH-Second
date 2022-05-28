from .partA2_head import PartA2FCHead
from .second_head import SECONDHead
from .roi_head_template import RoIHeadTemplate


__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'SECONDHead': SECONDHead,
}
