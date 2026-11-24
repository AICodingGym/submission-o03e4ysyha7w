"""Elapsed time that excludes Windows sleep for long runs."""
import ctypes
import os
import time


def awake_seconds():
    if os.name != "nt":
        return time.monotonic()
    ticks = ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(ticks)):
        raise ctypes.WinError()
    return ticks.value / 10_000_000


if __name__ == "__main__":
    print(awake_seconds())
