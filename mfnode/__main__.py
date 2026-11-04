"""Punto de entrada para `python -m mfnode`."""
from .main import _parse_args, run

if __name__ == '__main__':
    run(_parse_args())
