from typing import TypeVar
from functools import wraps
from typing import Callable, Coroutine

T = TypeVar("T")
P = TypeVar("P")
V = TypeVar("V")


def run_coro(c: Coroutine[T, P, V]) -> V:
    import asyncio

    return asyncio.run(c)


def as_sync(f: Callable[..., Coroutine]):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return run_coro(f(args, kwargs))

    return wrapper


def run_as_sync(f: Callable[..., Coroutine], *args, **kwargs):
    return as_sync(f)(args, kwargs)
