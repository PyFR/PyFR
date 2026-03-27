from pyfr.plugins.triggers.base import BaseTriggerSource, TriggerManager
from pyfr.plugins.triggers.field import FieldTriggerSource
from pyfr.plugins.triggers.point import PointTriggerSource
from pyfr.plugins.triggers.sources import (AllTriggerSource, AnyTriggerSource,
                                           DurationTriggerSource,
                                           ExpressionTriggerSource,
                                           FileTriggerSource,
                                           ManualTriggerSource,
                                           SignalTriggerSource,
                                           TimeTriggerSource,
                                           WallclockTriggerSource)
from pyfr.plugins.triggers.steady import SteadyTriggerSource
from pyfr.util import subclass_where


def get_trigger(name, *args, **kwargs):
    return subclass_where(BaseTriggerSource, name=name)(*args, **kwargs)
