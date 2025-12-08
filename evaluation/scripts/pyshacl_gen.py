#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license,
#  the text of which is available at https://opensource.org/license/MIT/
#  or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#
# Reference : https://github.com/RDFLib/pySHACL/issues/189#issuecomment-1626387650

from pyshacl import Validator
from rdflib import Graph

with open('uc.ttl') as f:
    smts = f.read()
    print(smts)
g = Graph()
g.parse(data=smts)

with open('uc_shacl.ttl') as f:
    smts = f.read()
    print(smts)
myshapes = Graph()
myshapes.parse(data=smts)

v = Validator(
    g,
    shacl_graph=myshapes,
    options={"advanced": True, "inference": "rdfs"}
)
conforms, report_graph, report_text = v.run()
expanded_graph = v.target_graph #<-- This gets the expanded data graph

print(v.target_graph.serialize(format='ttl'))

# --- EOF --------------------------------------------------------------
