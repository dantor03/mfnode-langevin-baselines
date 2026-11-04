"""Punto de entrada para `python -m mfnode`."""
from .main import _parse_args, main

if __name__ == '__main__':
    args = _parse_args()
    main(experiment=args.experiment, epochs=args.epochs)
