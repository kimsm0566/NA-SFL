"""Single entrypoint for GPU NA-SFL experiments and auxiliary-client sweeps."""
from mpi4py import MPI
from utils.option import parse_args


def main(argv=None):
    args = parse_args(argv)
    from utils.na_sfl_run import run_na_sfl
    _, path = run_na_sfl(args, MPI.COMM_WORLD)
    print(path, flush=True)


if __name__ == '__main__':
    main()
