#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lidar Scatterer Augmentation (LISA)
    - Given lidar point cloud and a rain rate generates corresponding noisy signal
    - Reflection data must be normalized to range [0 1] and
      range must be in units of meters
"""
import numpy as np
from scipy.special import gamma
from scipy.integrate import trapz
import PyMieScatt as ps


