"""Windows shim for the Unix-only stdlib `fcntl` module.

SANA's training-data loaders (`diffusion/data/wids/*`) do a top-level `import fcntl`
for file locking. They are dragged in transitively by `diffusion.model.builder` even on
the inference path, but the locking functions are never CALLED during inference. This stub
exists only so the import succeeds on Windows; every function is a no-op.
"""

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


def flock(fd, operation):  # noqa: D401 - stub
    return None


def lockf(fd, operation, length=0, start=0, whence=0):
    return None


def fcntl(fd, cmd, arg=0):
    return 0


def ioctl(fd, request, arg=0, mutate_flag=True):
    return 0
