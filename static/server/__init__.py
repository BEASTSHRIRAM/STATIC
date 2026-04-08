# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NeoVent environment server components."""

from .static_environment import NeoVentEnvironment

# Backward compatibility alias for older import paths.
StaticEnvironment = NeoVentEnvironment

__all__ = ["NeoVentEnvironment", "StaticEnvironment"]
