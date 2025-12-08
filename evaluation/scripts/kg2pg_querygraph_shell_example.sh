#
# Software Name: PyGraft-gen
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# the text of which is available at https://opensource.org/license/MIT/
# or see the "LICENSE" file for more details.
#
# Authors: See CONTRIBUTORS.txt
# Software description: A RDF Knowledge Graph stochastic generation solution.
#
#

# Note that we consider the to "resilience" dataset for the below example

# Extract vertices from the use case pattern RDF graph
~/.local/jena/bin/sparql \
  --data ./resilience/uc_kg.ttl \
  --query ./rq_kg2pg_vertices.sparql \
  --results csv > out/kg2pg_vertices.csv

# Extract edges from the use case pattern RDF graph
~/.local/jena/bin/sparql \
  --data ./resilience/uc_kg.ttl \
  --query ./rq_kg2pg_edges.sparql \
  --results csv > out/kg2pg_edges.csv

# Convert the CSV output to the JSON format
mlr --icsv --ojson cat out/kg2pg_vertices.csv > out/kg2pg_vertices_only.json
mlr --icsv --ojson cat out/kg2pg_edges.csv > out/kg2pg_edges_only.json

# Combine extracted features into a NetworkX
sed -e '/MY_NODES/{r out/kg2pg_vertices_only.json' -e 'd}' networkx_template.json > out/nx_query_graph.json
sed -i -e '/MY_LINKS/{r out/kg2pg_edges_only.json' -e 'd}' out/nx_query_graph.json

# Remove temporary files
rm -f out/kg2pg_vertices*.*
rm -f out/kg2pg_edges*.*
