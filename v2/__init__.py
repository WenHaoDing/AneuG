__all__ = ["CanonicalSpec", "MultiCanonicalGHDReconstruct"]


def __getattr__(name):
    if name in __all__:
        from .ghd_reconstruct import CanonicalSpec, MultiCanonicalGHDReconstruct
        return {
            "CanonicalSpec": CanonicalSpec,
            "MultiCanonicalGHDReconstruct": MultiCanonicalGHDReconstruct,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
