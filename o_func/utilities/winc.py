#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import platform
def winc(i):
        if platform == "win32" or "Windows":
            b = i.replace('\\','/')
        else:
            b = i
        return b