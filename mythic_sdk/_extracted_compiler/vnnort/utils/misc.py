def format_large_number(n: float, digits: int = 2) -> str:
    """Convert a large number to a human-readable string with K, M, G, etc."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.{digits}f}G"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.{digits}f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.{digits}f}K"
    return f"{n:.{digits}f}"
