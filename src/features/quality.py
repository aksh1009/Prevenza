"""Measurement-validity indicators (do not modify values; only qualify them).

Heat index (US NWS approximation, not WBGT): the NWS heat-index chart tabulates air temperatures up to 110 F
(43.3 C) and heat-index values up to about 137 F (58.3 C). Values beyond either bound come from extrapolating the
Rothfusz regression outside the range it was fitted/tabulated on, so their magnitude is unreliable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NWS_MAX_TEMPERATURE_C = (110 - 32) * 5 / 9      # 43.33
NWS_MAX_HEAT_INDEX_C = (137 - 32) * 5 / 9       # 58.33
VALIDITY = ("within_range", "extrapolated", "unavailable")


def heat_index_validity(temp_c, heat_index_c, max_temperature_c: float = NWS_MAX_TEMPERATURE_C,
                        max_heat_index_c: float = NWS_MAX_HEAT_INDEX_C) -> np.ndarray:
    """'unavailable' if heat index is NA; 'extrapolated' if HI or (known) temperature exceeds the NWS chart range;
    otherwise 'within_range'. Values themselves are never altered."""
    hi = pd.to_numeric(pd.Series(heat_index_c), errors="coerce").to_numpy(float)
    t = pd.to_numeric(pd.Series(temp_c), errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore"):
        extrap = (hi > max_heat_index_c) | (~np.isnan(t) & (t > max_temperature_c))
    return np.where(np.isnan(hi), "unavailable", np.where(extrap, "extrapolated", "within_range"))
