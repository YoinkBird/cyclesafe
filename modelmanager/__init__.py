# from helpers import *
#dbg# print("hi - modelgen/modelmanager")

import os,sys

# get file path src: https://stackoverflow.com/a/3430395
moduledir = os.path.dirname(os.path.abspath(__file__))

sys.path.append("%s" % moduledir)

from . import *

