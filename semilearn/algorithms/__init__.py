# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semilearn.core.utils import ALGORITHMS

name2alg = ALGORITHMS


def get_algorithm(args, net_builder, tb_log, logger):
    if args.algorithm in ALGORITHMS:
        alg = ALGORITHMS[args.algorithm](args=args, net_builder=net_builder, tb_log=tb_log, logger=logger)  # name2alg[args.algorithm](
        return alg
    else:
        raise KeyError(f"Unknown algorithm: {str(args.algorithm)}")
