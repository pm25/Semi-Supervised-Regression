# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

from semilearn.algorithms import name2alg
from semilearn.core.utils import CLS_ALGORITHMS

name2clsalg = CLS_ALGORITHMS


def get_cls_algorithm(args, net_builder, tb_log, logger):
    if args.cls_algorithm not in name2clsalg:
        print(f"Unknown semi-supervised classification algorithm: {args.cls_algorithm }")

    class DummyClass(name2clsalg[args.cls_algorithm], name2alg[args.algorithm]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    alg = DummyClass(args=args, net_builder=net_builder, tb_log=tb_log, logger=logger)
    return alg
