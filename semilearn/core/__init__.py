# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

from .algorithmbase import AlgorithmBase, ClsAlgorithmBase
from .utils.registry import import_all_modules_for_register

import_all_modules_for_register()
