#from modelgen.code.helpers import *
#from modelgen.code.feature_definitions import *
#from modelgen.code.txdot_parse import *
#---
#from code.helpers import *
#from code.feature_definitions import *
#from code.txdot_parse import *
#--- vv good vv
import os,sys
from os import sys

# get file path src: https://stackoverflow.com/a/3430395
filedir = os.path.dirname(os.path.abspath(__file__))
# start here: https://stackoverflow.com/questions/50598995/how-do-i-import-all-functions-from-a-package-in-python
# moduledir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
moduledir = os.path.dirname(os.path.abspath(__file__))

sys.path.append("%s/code" % moduledir)

from code import *
# vvv suggestion from src: https://stackoverflow.com/questions/16480898/receiving-import-error-no-module-named-but-has-init-py
#from os import sys
#sys.path.append("code")
# ^^^
print("hi - modelgen")

# also referred to pandas __init__.py src : # https://github.com/pandas-dev/pandas/blob/master/pandas/__init__.py
# __init__.py src: https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html#converting-a-folder-of-scripts-into-an-importable-package-of-modules
# __init__.py src: https://stackoverflow.com/questions/2183205/importing-a-module-in-nested-packages
