import os
import sys
import re

version = sys.argv[1].removeprefix('v')

new_preamble = f'''#-Begin-preamble-------------------------------------------------------
#
#                           CERN
#
#     European Organization for Nuclear Research
#
#
#     This file is part of the code:
#
#                   PyECLOUD Version {version}
#
#
#     Main author:          Giovanni IADAROLA
#                           BE-ABP Group
#                           CERN
#                           CH-1211 GENEVA 23
#                           SWITZERLAND
#                           giovanni.iadarola@cern.ch
#
#     Contributors:         Eleonora Belli
#                           Philipp Dijkstal
#                           Lorenzo Giacomel
#                           Lotta Mether
#                           Annalisa Romano
#                           Giovanni Rumolo
#                           Eric Wulff
#
#
#     Copyright  CERN,  Geneva  2011  -  Copyright  and  any   other
#     appropriate  legal  protection  of  this  computer program and
#     associated documentation reserved  in  all  countries  of  the
#     world.
#
#     Organizations collaborating with CERN may receive this program
#     and documentation freely and without charge.
#
#     CERN undertakes no obligation  for  the  maintenance  of  this
#     program,  nor responsibility for its correctness,  and accepts
#     no liability whatsoever resulting from its use.
#
#     Program  and documentation are provided solely for the use  of
#     the organization to which they are distributed.
#
#     This program  may  not  be  copied  or  otherwise  distributed
#     without  permission. This message must be retained on this and
#     any other authorized copies.
#
#     The material cannot be sold. CERN should be  given  credit  in
#     all references.
#
#-End-preamble---------------------------------------------------------'''

pattern = re.compile(
    r'#-Begin-preamble-+\n.*?#-End-preamble-+',
    re.DOTALL
)

for dirpath, _, filenames in os.walk('./PyECLOUD'):
    for filename in filenames:
        if not filename.endswith('.py'):
            continue

        path = os.path.join(dirpath, filename)

        if os.path.abspath(path) == os.path.abspath(__file__):
            continue

        with open(path, 'r') as f:
            content = f.read()

        new_content, replacements = pattern.subn(new_preamble, content)

        if replacements:
            print(f'Changing preamble: {path}')

            with open(path, 'w') as f:
                f.write(new_content)