"""Helpers shared by the Stage II S-GE modules."""


def normalize_sge_scales(scales):
    """Validate and normalize S-GE conv scales to an order-preserving unique tuple."""
    if scales is None:
        scales = (1, 3, 5)
    if isinstance(scales, str):
        scales = scales.strip().strip("()[]")
        scales = tuple(int(s.strip()) for s in scales.split(",") if s.strip())
    else:
        scales = tuple(int(s) for s in scales)
    allowed = {1, 3, 5}
    if not scales:
        raise ValueError("SGE_SCALES must contain at least one scale")
    if any(s not in allowed for s in scales):
        raise ValueError(f"SGE_SCALES must be a subset of {{1, 3, 5}}, got {scales}")
    return tuple(dict.fromkeys(scales))


# Backward-compatible alias
_normalize_sge_scales = normalize_sge_scales
