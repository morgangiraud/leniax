import argparse


def get_default_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='''Lenia jax by Morgan Giraud

        recommanded settings: (2D) -d2 -p2, (wide) -d2 -p0 -w 10 9, (3D) -d3 -p3, (4D) -d4 -p4'''
    )

    # World params
    parser.add_argument(
        '-d',
        '--dims',
        dest='nb_dims',
        default=2,
        action='store',
        type=int,
        help='number of world\' dimensions (default 2D)'
    )
    parser.add_argument(
        '-c',
        '--channels',
        dest='nb_channels',
        default=1,
        action='store',
        type=int,
        help='number of world\'s channels (default 1)'
    )
    parser.add_argument('-R', '--ratio', dest='R', default=13, action='store', type=int, help='Ratio')
    parser.add_argument(
        '-T',
        '--timestep',
        dest='T',
        default=10,
        action='store',
        type=int,
        help='number of world\'s channels (default 1)'
    )

    # Rendering params
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '-w',
        '--win_size',
        dest='W',
        default=[9],
        action='store',
        type=int,
        nargs='+',
        help='window size = 2^W (apply to all sides if only one value, default 2^9 = 512)'
    )
    group.add_argument(
        '-s',
        '--size',
        dest='S',
        default=None,
        action='store',
        type=int,
        nargs='+',
        help='size of the world (number of pixels) = 2^S (apply to all sides if only one value, default 2^(W-P) = 128)'
    )
    parser.add_argument(
        '-p', '--pixel', dest='P', default=None, action='store', type=int, help='pixel size = 2^P (default 2^D)'
    )
    parser.add_argument(
        '-b', '--pixel_border', dest='B', default=0, action='store', type=int, help='pixel border (default 0)'
    )

    # Configuration params
    parser.add_argument(
        '--seed',
        dest='seed',
        default=0,
        action='store',
        type=str,
    )
    parser.add_argument(
        '--config',
        dest='config',
        default=None,
        action='store',
        type=str,
    )

    parser.add_argument(
        '--max_run_iter',
        dest='max_run_iter',
        default=512,
        action='store',
        type=int,
    )

    return parser
