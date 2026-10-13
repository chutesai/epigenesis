"""Explicit EDA kernel identity for model loading and comparison artifacts."""
from pathlib import Path
import sys

DEFAULT = "naive_fp32"
CHOICES = (DEFAULT, "fla_kda_2t")
FLA_DEPS = Path("/workspace/malleable/pydeps_fla")


def add_argument(parser):
    parser.add_argument("--eda-backend", choices=CHOICES, default=DEFAULT,
                        help="EDA kernel backend for this run (no fallback)")


def selected(args):
    return _checked(getattr(args, "eda_backend", DEFAULT))


def add_crosseval_argument(parser):
    parser.add_argument("--allow-backend-crosseval", action="store_true",
                        help="allow saved patches from another EDA backend for evaluation only")


def validate_crosseval(args):
    if getattr(args, "allow_backend_crosseval", False) and any(
            getattr(args, key, 0) > 0 for key in ("steps", "wake_steps", "repair_steps")):
        raise ValueError("--allow-backend-crosseval is evaluation-only; training steps must be zero")


def require_patch_backend(record, args, context="patch"):
    """Opt-in patch loading only; cache and artifact checks remain strict."""
    args = getattr(args, "_fsa_crosseval_args", args)
    backend = getattr(args, "_fsa_eda_backend", None) or selected(args)
    actual = recorded(record)
    if actual == backend:
        return actual
    if not getattr(args, "allow_backend_crosseval", False):
        return require_backend(record, backend, context)
    validate_crosseval(args)
    crossing = dict(saved=actual, eval=backend)
    previous = getattr(args, "backend_crosseval", None)
    if previous is not None and previous != crossing:
        raise ValueError("mixed saved EDA backends in cross-evaluation")
    if previous is None:
        args.backend_crosseval = crossing
        print(f"CROSS-EVAL saved={actual} eval={backend}", flush=True)
    return actual


def crosseval_metadata(args):
    args = getattr(args, "_fsa_crosseval_args", args)
    crossing = getattr(args, "backend_crosseval", None)
    return {"backend_crosseval": dict(crossing)} if crossing is not None else {}


def _checked(name):
    if name not in CHOICES:
        raise ValueError(f"unsupported EDA backend: {name!r}")
    return name


def recorded(record):
    """Legacy artifacts are naive_fp32; conflicting nested identities are invalid."""
    names = []
    if "eda_backend" in record:
        names.append(_checked(record["eda_backend"]))
    for key in ("meta", "metadata", "world"):
        if isinstance(record.get(key), dict) and "eda_backend" in record[key]:
            names.append(_checked(record[key]["eda_backend"]))
    if len(set(names)) > 1:
        raise ValueError(f"conflicting EDA backends in artifact: {names}")
    return names[0] if names else DEFAULT


def require_backend(record, backend, context="artifact"):
    actual = recorded(record)
    if actual != _checked(backend):
        raise ValueError(f"EDA backend mismatch for {context}: saved {actual}, eval {backend}")
    return actual


def require_same(records, context="comparison"):
    names = {recorded(record) for record in records}
    if len(names) > 1:
        raise ValueError(f"mixed EDA backends in {context}: {sorted(names)}")
    return next(iter(names), DEFAULT)


def announce(args):
    print(f"EDA-BACKEND {selected(args)}", flush=True)


def prepare(backend):
    _checked(backend)
    if backend == "fla_kda_2t":
        if FLA_DEPS.exists():
            path = str(FLA_DEPS)
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
        try:
            from fla.ops.kda import chunk_kda
            if not callable(chunk_kda):
                raise ImportError("chunk_kda is not callable")
        except Exception as exc:
            raise RuntimeError("EDA backend fla_kda_2t requires fla.ops.kda.chunk_kda; "
                               "refusing fallback to naive_fp32") from exc


def build_model(builder, args, *positional, **kwargs):
    validate_crosseval(args)
    try:
        from . import c44_runtime as recovery
    except ImportError:
        import c44_runtime as recovery
    backend = selected(args)
    prepare(backend)
    recovery.heartbeat('model_build', 0, 1, force=True)
    model, config, device = builder(*positional, eda_kernel_backend=backend, **kwargs)
    recovery.heartbeat('model_build', 1, 1, force=True)
    model._fsa_eda_backend = backend
    if getattr(args, "allow_backend_crosseval", False):
        model._fsa_crosseval_args = args
    return model, config, device
