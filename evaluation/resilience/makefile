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

# ----------------------------------------------------------------------

SHELL := /bin/bash
ROOT_DIR := $(PWD)

CHOWLK_API := https://chowlk.linkeddata.es/api

# Tools (absolute paths so they work regardless of current directory)
JENA_HOME := $(abspath ../../../lib/apache-jena/apache-jena-5.4.0)
export JENA_HOME
JENA_SPARQL := $(abspath $(JENA_HOME)/bin/sparql)

RQVIZ := $(abspath ../../scripts/rqviz_kglab.py)

# ----------------------------------------------------------------------

# Loading (optional) environment variables from file.
# For example, to define PROXY setting :
# PROXY_SRV=xxx.yyy.fr
# PROXY_PORT=8080
-include ./.env

# ----------------------------------------------------------------------

## makefile for the pygraft-gen project
help:	## Show this help.
	# Get lines with double dash comments and display it
	@fgrep -h "## " $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/## //'

# ----------------------------------------------------------------------

uc-to-ttl: uc_kg.ttl uc_kg_extended.ttl

uc_kg.ttl: uc_diagram.drawio.chowlk_response.json ## Extract TTL data from the Chowlk response
	@echo -e "\033[35m > Extract TTL data from the Chowlk response \033[0m "
	jq \
	  -r \
	  '.ttl_data' \
	  uc_diagram.drawio.chowlk_response.json \
	  > uc_kg.ttl
	@echo -e "\033[35m > Done "

uc_diagram.drawio.chowlk_response.json: uc_diagram.drawio  ## Send diagram to Chowlk
	@echo -e "\033[35m > Send diagram to Chowlk \033[0m (see https://github.com/oeg-upm/Chowlk)"
	@echo -e "PROXY_SRV = ${PROXY_SRV} / PROXY_PORT = ${PROXY_PORT}"
	curl \
	  -F data=@./uc_diagram.drawio \
	  ${CHOWLK_API} \
	  --proxy ${PROXY_SRV}:${PROXY_PORT} \
	  --output uc_diagram.drawio.chowlk_response.json
	@echo -e "\033[35m > Done  \033[0m"

# ----------------------------------------------------------------------

uc_kg_extended.ttl: uc_diagram_extended.drawio.chowlk_response.json ## Extract TTL data from the Chowlk response
	@echo -e "\033[35m > Extract TTL data from the Chowlk response \033[0m "
	jq \
	  -r \
	  '.ttl_data' \
	  uc_diagram_extended.drawio.chowlk_response.json \
	  > uc_kg_extended.ttl
	@echo -e "\033[35m > Done "

uc_diagram_extended.drawio.chowlk_response.json: uc_diagram_extended.drawio  ## Send diagram to Chowlk
	@echo -e "\033[35m > Send diagram to Chowlk \033[0m (see https://github.com/oeg-upm/Chowlk)"
	@echo -e "PROXY_SRV = ${PROXY_SRV} / PROXY_PORT = ${PROXY_PORT}"
	curl \
	  -F data=@./uc_diagram_extended.drawio \
	  ${CHOWLK_API} \
	  --proxy ${PROXY_SRV}:${PROXY_PORT} \
	  --output uc_diagram_extended.drawio.chowlk_response.json
	@echo -e "\033[35m > Done  \033[0m"

# ----------------------------------------------------------------------

clean:
	@echo -e "\033[35m > Clean intermediary files \033[0m"
	rm -f uc_kg*.ttl
	rm -f *.chowlk_response.json
	rm -f out_*.*
	@echo -e "\033[35m > Done  \033[0m"

# ----------------------------------------------------------------------

eval-sparql: out_query_sparql.csv out_query_sparql_extended.csv	## Run SPARQL query on use case

out_query_sparql.csv: uc_query.sparql uc_kg.ttl	## Run SPARQL query on use case
	@echo -e "\033[35m > Run SPARQL queries  \033[0m"
	@$(JENA_SPARQL) --data ./uc_kg.ttl --query ./uc_query.sparql --results csv > out_query_sparql.csv
	@echo -e "\033[35m > Done  \033[0m"

out_query_sparql_extended.csv: uc_query.sparql uc_kg_extended.ttl	## Run SPARQL query on use case
	@echo -e "\033[35m > Run SPARQL queries  \033[0m"
	@$(JENA_SPARQL) --data ./uc_kg_extended.ttl --query ./uc_query.sparql --results csv > out_query_sparql_extended.csv
	@echo -e "\033[35m > Done  \033[0m"

# ----------------------------------------------------------------------

sparql-query-viz: uc_query.sparql	## Make a diagram of the SPARQL query
	@echo -e "\033[35m > Make a diagram of the SPARQL query  \033[0m (relies on https://github.com/DerwenAI/kglab/)"
	@python3 $(RQVIZ)
	@echo -e "\033[35m > Done  \033[0m"

# ----------------------------------------------------------------------
# KG2PG

kg2pg-query-graph:
	@echo -e "\033[35m > Extract vertices from the use case pattern RDF graph  \033[0m"
	@$(JENA_SPARQL) \
	  --data ./uc_kg.ttl \
	  --query ../../scripts/rq_kg2pg_vertices.sparql \
	  --results csv > ./kg2pg_vertices.csv
	@echo -e "\033[35m > Extract edges from the use case pattern RDF graph  \033[0m"
	@$(JENA_SPARQL) \
	  --data ./uc_kg.ttl \
	  --query ../../scripts/rq_kg2pg_edges.sparql \
	  --results csv > ./kg2pg_edges.csv
	@echo -e "\033[35m > Convert the CSV output to the JSON format  \033[0m"
	@mlr --icsv --ojson cat ./kg2pg_vertices.csv > ./kg2pg_vertices_only.json
	@mlr --icsv --ojson cat ./kg2pg_edges.csv > ./kg2pg_edges_only.json
	@echo -e "\033[35m > Combine extracted features into a NetworkX  \033[0m"
	@sed -e '/MY_NODES/{r ./kg2pg_vertices_only.json' -e 'd}' ../../scripts/networkx_template.json > ./nx_query_graph.json
	@sed -i -e '/MY_LINKS/{r ./kg2pg_edges_only.json' -e 'd}' ./nx_query_graph.json
	@echo -e "\033[35m > Remove temporary files  \033[0m"
	@rm -f ./kg2pg_vertices*.*
	@rm -f ./kg2pg_edges*.*
	@echo -e "\033[35m > Done  \033[0m"

kg2pg-data-graph-extended:
	@echo -e "\033[35m > Extract vertices from the extended use case pattern RDF graph  \033[0m"
	@$(JENA_SPARQL) \
	  --data ./uc_kg_extended.ttl \
	  --query ../../scripts/rq_kg2pg_vertices.sparql \
	  --results csv > ./kg2pg_vertices.csv
	@echo -e "\033[35m > Extract edges from the extended use case pattern RDF graph  \033[0m"
	@$(JENA_SPARQL) \
	  --data ./uc_kg_extended.ttl \
	  --query ../../scripts/rq_kg2pg_edges.sparql \
	  --results csv > ./kg2pg_edges.csv
	@echo -e "\033[35m > Convert the CSV output to the JSON format  \033[0m"
	@mlr --icsv --ojson cat ./kg2pg_vertices.csv > ./kg2pg_vertices_only.json
	@mlr --icsv --ojson cat ./kg2pg_edges.csv > ./kg2pg_edges_only.json
	@echo -e "\033[35m > Combine extracted features into a NetworkX  \033[0m"
	@sed -e '/MY_NODES/{r ./kg2pg_vertices_only.json' -e 'd}' ../../scripts/networkx_template.json > ./nx_data_graph_extended.json
	@sed -i -e '/MY_LINKS/{r ./kg2pg_edges_only.json' -e 'd}' ./nx_data_graph_extended.json
	@echo -e "\033[35m > Remove temporary files  \033[0m"
	@rm -f ./kg2pg_vertices*.*
	@rm -f ./kg2pg_edges*.*
	@echo -e "\033[35m > Done  \033[0m"

eval-vf2pp:
	@echo -e "\033[35m > Subgraph matching using VF2++ against the extended use case graph  \033[0m"
	python3 ../../scripts/subgraph-matching.py \
	  --log 10 \
	  --queryGraph ./nx_query_graph.json \
	  --dataGraph ./nx_data_graph_extended.json \
	  --destDir .
	@echo -e "\033[35m > Done  \033[0m"

# --- EOF --------------------------------------------------------------