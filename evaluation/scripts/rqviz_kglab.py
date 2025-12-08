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

# Reference : https://derwen.ai/docs/kgl/ex4_0/

import kglab

with open('uc_query.sparql') as f:
    sparql = f.read()
    print(sparql)

kg = kglab.KnowledgeGraph()

pyvis_graph = kg.visualize_query(sparql, notebook=True)

pyvis_graph.force_atlas_2based()
pyvis_graph.show("uc_query.html")

# --- EOF --------------------------------------------------------------
