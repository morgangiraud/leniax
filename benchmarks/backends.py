###
# File taken from https://github.com/dionhaefner/pyhpc-benchmarks
###

import os


class BackendNotSupported(Exception):
    pass


class SetupContext:
    def __init__(self, f):
        self._f = f
        self._f_args = (tuple(), dict())

    def __call__(self, *args, **kwargs):
        self._f_args = (args, kwargs)
        return self

    def __enter__(self):
        self._env = os.environ.copy()
        args, kwargs = self._f_args
        self._f_iter = iter(self._f(*args, **kwargs))

        try:
            module = next(self._f_iter)
        except Exception as e:
            raise BackendNotSupported(str(e)) from None

        return module

    def __exit__(self, *args, **kwargs):
        try:
            next(self._f_iter)
        except StopIteration:
            pass
        os.environ = self._env


setup_function = SetupContext

# setup function definitions


@setup_function
def setup_jax(device="cpu"):
    os.environ.update(
        XLA_FLAGS=(
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=1 "
            "inter_op_parallelism_threads=1 "
        ),
    )

    if device in ("cpu", "gpu"):
        os.environ.update(JAX_PLATFORM_NAME=device)

    import jax
    from jax.config import config

    if device == "tpu":
        config.update("jax_xla_backend", "tpu_driver")
        config.update("jax_backend_target", os.environ.get("JAX_BACKEND_TARGET"))

    if device != "tpu":
        # use 64 bit floats (not supported on TPU)
        config.update("jax_enable_x64", True)

    if device == "gpu":
        assert len(jax.devices()) > 0

    yield jax


__backends__ = {
    "jax": setup_jax,
}
