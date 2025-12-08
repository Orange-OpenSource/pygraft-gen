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
# --- Config -----------------------------------------------------------

SHELL := /bin/bash
ROOT_DIR := $(PWD)

YELLOW='\033[1;33m'
NC='\033[0m' # No Color

VIRTDB_IMG=openlink/virtuoso-opensource-7

CHOWLK_API := https://chowlk.linkeddata.es/api

# ----------------------------------------------------------------------

# Loading (optional) environment variables from file.
# For example, to define PROXY setting :
# PROXY_SRV=xxx.yyy.fr
# PROXY_PORT=8080
-include ./.env

# Tools (absolute paths so they work regardless of current directory)
JENA_HOME := $(abspath ../lib/apache-jena/apache-jena-5.4.0)
export JENA_HOME
JENA_SPARQL := $(abspath $(JENA_HOME)/bin/sparql)
ROBOT := java -jar $(abspath ./lib/ontodev-robot/robot.jar)
MLR := $(abspath ../lib/miller/miller-6.13.0-linux-amd64/mlr)
JQ := $(abspath ../lib/jq/jq-linuxamd64)
AJV := $(abspath ../lib/ajv-cli/node_modules/ajv-cli/dist/index.js)

# Generic “current” output directory --> set to the Jena dir by default
OUT_JENA_DIR := ./out/out_jena
OUT_ROBOT_DIR := ./out/out_robot
OUT_OOPS_DIR := ./out/out_oops
OUT_DIR := ./out

# Graph set for running subgraph matching (generic)
EVAL_GRAPH_LIST = ../config-gen/kg_gen_experiments/noria_new/G1/G1_full_graph.ttl ../config-gen/kg_gen_experiments/noria_new/G2/G2_full_graph.ttl ../config-gen/kg_gen_experiments/noria_new/G3/G3_full_graph.ttl ../config-gen/kg_gen_experiments/noria_new/G4/G4_full_graph.ttl ../config-gen/kg_gen_experiments/noria_new/G5/G5_full_graph.ttl
#EVAL_GRAPH_LIST = ../config-gen/kg_gen_experiments/noria_new/G1/G1_full_graph.ttl

# Graph set for running subgraph matching (vf2pp)
EVAL_GRAPH_LIST_NX = ./out/kg2pg/nx_data_graph_G1_full_graph.json ./out/kg2pg/nx_data_graph_G2_full_graph.json ./out/kg2pg/nx_data_graph_G3_full_graph.json ./out/kg2pg/nx_data_graph_G4_full_graph.json ./out/kg2pg/nx_data_graph_G5_full_graph.json
#EVAL_GRAPH_LIST_NX = ./out/kg2pg/nx_data_graph_G3_full_graph.json

# Query file for subgraph matching using SPARQL
EVAL_SPARQL_QUERY_FILE = uc_query.sparql
#EVAL_SPARQL_QUERY_FILE = uc_query_count.sparql

# Graph set for loading into a graph store
#EVAL_GRAPH_LIST = ../config-gen/kg_gen_experiments/noria_new/G1/G1_full_graph.ttl ../config-gen/kg_gen_experiments/noria_new/G2/G2_full_graph.ttl ../config-gen/kg_gen_experiments/noria_new/G3/G3_full_graph.ttl ../config-gen/kg_gen_experiments/noria_new/G4/G4_full_graph.ttl ../config-gen/kg_gen_experiments/noria_new/G5/G5_full_graph.ttl
GSTORE_GRAPH_LIST = ../config-gen/kg_gen_experiments/noria_new/G2/G2_full_graph.ttl


# ----------------------------------------------------------------------

## makefile for the pygraft-gen project / evaluation
help:	## Show this help.
	# Get lines with double dash comments and display it
	@fgrep -h "## " $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/## //'

# ----------------------------------------------------------------------
# --- INSTALL ----------------------------------------------------------

install-apache-jena:
	@echo -e "\033[35m > Install Apache Jena  \033[0m - Need Internet access, see https://jena.apache.org/"
	@echo -e "PROXY_SRV = ${PROXY_SRV:-} / PROXY_PORT = ${PROXY_PORT:-}"
	@PROXY_OPT=$$(if [ -n "${PROXY_SRV:-}" ] && [ -n "${PROXY_PORT:-}" ]; then echo --proxy ${PROXY_SRV}:${PROXY_PORT}; else echo ""; fi); \
	curl -fL $$PROXY_OPT \
	  -o ../lib/apache-jena/apache-jena-5.4.0.tar.gz \
	  --create-dirs \
	  https://archive.apache.org/dist/jena/binaries/apache-jena-5.4.0.tar.gz
	@echo -e "\033[35m > Extracting Apache Jena  \033[0m"
	@tar -xzf ../lib/apache-jena/apache-jena-5.4.0.tar.gz -C ./lib/apache-jena/
	@rm -f ../lib/apache-jena/apache-jena-5.4.0.tar.gz
	@chmod -R +x ../lib/apache-jena/apache-jena-5.4.0/bin || true
	@echo -e "\033[35m > Done  \033[0m"

install-sparqlworks:
	@echo -e "\033[35m > Install SPARQLWorks  \033[0m - Need Internet access, see https://github.com/danielhmills/sparqlworks"
	@echo -e "PROXY_SRV = ${PROXY_SRV} / PROXY_PORT = ${PROXY_PORT}"
	@PROXY_OPT=$$(if [ -n "${PROXY_SRV}" ] && [ -n "${PROXY_PORT}" ]; then echo --proxy ${PROXY_SRV}:${PROXY_PORT}; else echo ""; fi); \
	curl -fL $$PROXY_OPT \
	  -o ../lib/sparqlworks/sparqlworks.zip \
	  --create-dirs \
	  https://github.com/danielhmills/sparqlworks/archive/refs/tags/0.7.0.zip
	@unzip ../lib/sparqlworks/sparqlworks.zip -d ../lib/sparqlworks/
	@echo -e "\033[35m > Done  \033[0m"


# ----------------------------------------------------------------------
# --- Subgraph matching using SPARQL ----------------------------------

eval-sparql-pattern: eval-sparql-pattern-resilience eval-sparql-pattern-topology eval-sparql-pattern-unreach

eval-sparql-pattern-resilience:
	@echo -e "\033[35m > [JENA] Run SPARQL resilience pattern query  ($(EVAL_SPARQL_QUERY_FILE))  \033[0m"
	@mkdir -p ${OUT_DIR}/patterns/resilience
	@for g in $(EVAL_GRAPH_LIST); do \
	  echo -e "\033[35m > Processing $$g ... \033[0m"; \
	  base=$$(basename "$$g" .ttl); \
	  $(JENA_SPARQL) \
	    --data "$$g" \
	    --query "./patterns/resilience/$(EVAL_SPARQL_QUERY_FILE)" \
	    --results CSV \
	    --quiet \
	  > "${OUT_DIR}/patterns/resilience/sparql_$$base.csv" ; \
	done
	@echo -e "\033[35m > Done  \033[0m"

eval-sparql-pattern-topology:
	@echo -e "\033[35m > [JENA] Run SPARQL topology pattern query  ($(EVAL_SPARQL_QUERY_FILE))  \033[0m"
	@mkdir -p ${OUT_DIR}/patterns/topology
	@for g in $(EVAL_GRAPH_LIST); do \
	  echo -e "\033[35m > Processing $$g ... \033[0m"; \
	  base=$$(basename "$$g" .ttl); \
	  $(JENA_SPARQL) \
	    --data "$$g" \
	    --query "./patterns/topology/$(EVAL_SPARQL_QUERY_FILE)" \
	    --results CSV \
	    --quiet \
	  > "${OUT_DIR}/patterns/topology/sparql_$$base.csv" ; \
	done
	@echo -e "\033[35m > Done  \033[0m"

eval-sparql-pattern-unreach:
	@echo -e "\033[35m > [JENA] Run SPARQL unreach pattern query ($(EVAL_SPARQL_QUERY_FILE)) \033[0m"
	@mkdir -p ${OUT_DIR}/patterns/unreach
	@for g in $(EVAL_GRAPH_LIST); do \
	  echo -e "\033[35m > Processing $$g ... \033[0m"; \
	  base=$$(basename "$$g" .ttl); \
	  $(JENA_SPARQL) \
	    --data "$$g" \
	    --query "./patterns/unreach/$(EVAL_SPARQL_QUERY_FILE)" \
	    --results CSV \
	    --quiet \
	  > "${OUT_DIR}/patterns/unreach/sparql_$$base.csv" ; \
	done
	@echo -e "\033[35m > Done  \033[0m"

# ----------------------------------------------------------------------
# --- Subgraph matching using VF2++ -----------------------------------

kg2pg-data-graph:
	@echo -e "\033[35m > KG2PG processing \033[0m"
	@mkdir -p ${OUT_DIR}/kg2pg
	@for g in $(EVAL_GRAPH_LIST); do \
	  echo -e "\033[35m > Extract vertices from the RDF graph: $$g \033[0m" ; \
	  $(JENA_SPARQL) \
	    --data $$g \
	    --query ./scripts/rq_kg2pg_vertices.sparql \
	    --results csv > ${OUT_DIR}/kg2pg/kg2pg_vertices.csv ; \
	  echo -e "\033[35m > Extract edges from the RDF graph: $$g  \033[0m" ; \
	  $(JENA_SPARQL) \
	    --data $$g \
	    --query ./scripts/rq_kg2pg_edges.sparql \
	    --results csv > ${OUT_DIR}/kg2pg/kg2pg_edges.csv ; \
	  echo -e "\033[35m > Convert the CSV output to the JSON format  \033[0m" ; \
	  mlr --icsv --ojson cat ${OUT_DIR}/kg2pg/kg2pg_vertices.csv > ${OUT_DIR}/kg2pg/kg2pg_vertices_only.json ; \
	  mlr --icsv --ojson cat ${OUT_DIR}/kg2pg/kg2pg_edges.csv > ${OUT_DIR}/kg2pg/kg2pg_edges_only.json ; \
	  echo -e "\033[35m > Combine extracted features into a NetworkX file \033[0m" ; \
	  sed -e '/MY_NODES/{r ${OUT_DIR}/kg2pg/kg2pg_vertices_only.json' -e 'd}' ./scripts/networkx_template.json > ${OUT_DIR}/kg2pg/nx_data_graph.json ; \
	  sed -i -e '/MY_LINKS/{r ${OUT_DIR}/kg2pg/kg2pg_edges_only.json' -e 'd}' ${OUT_DIR}/kg2pg/nx_data_graph.json ; \
	  echo -e "\033[35m > Give a name to the NetworkX file ... \033[0m" ; \
	  base=$$(basename "$$g" .ttl) ; \
	  mv ${OUT_DIR}/kg2pg/nx_data_graph.json ${OUT_DIR}/kg2pg/nx_data_graph_$$base.json ; \
	  echo -e "\033[35m > Give a name to the NetworkX file: ${OUT_DIR}/kg2pg/nx_data_graph_$$base.json \033[0m" ; \
	  echo -e "\033[35m > Remove temporary files  \033[0m" ; \
	  rm -f ${OUT_DIR}/kg2pg/kg2pg_vertices*.* ; \
	  rm -f ${OUT_DIR}/kg2pg/kg2pg_edges*.* ; \
	done
	@echo -e "\033[35m > Done  \033[0m"

eval-vf2pp-pattern: eval-vf2pp-pattern-resilience eval-vf2pp-pattern-topology eval-vf2pp-pattern-unreach

eval-vf2pp-pattern-resilience:
	@echo -e "\033[35m > Subgraph matching using VF2++ for the 'resilience' query graph  \033[0m"
	@mkdir -p ${OUT_DIR}/kg2pg/resilience
	@for g in $(EVAL_GRAPH_LIST_NX); do \
	  echo -e "\033[35m > Processing $$g ... \033[0m"; \
	  base=$$(basename "$$g" .json); \
	  python3 ./scripts/subgraph-matching.py \
	    --log 30 \
	    --mappingStorage "none" \
	    --queryGraph "./patterns/resilience/nx_query_graph.json" \
	    --dataGraph "$$g" \
	    --destDir ${OUT_DIR}/kg2pg/resilience ; \
	  echo -e "\033[35m > Give a name to the mappings file ... \033[0m" ; \
	  mv ${OUT_DIR}/kg2pg/resilience/vf2pp_mappings.json ${OUT_DIR}/kg2pg/resilience/vf2pp_mappings_$$base.json ; \
	done
	@echo -e "\033[35m > Done  \033[0m"

eval-vf2pp-pattern-topology:
	@echo -e "\033[35m > Subgraph matching using VF2++ for the 'topology' query graph  \033[0m"
	@mkdir -p ${OUT_DIR}/kg2pg/topology
	@for g in $(EVAL_GRAPH_LIST_NX); do \
	  echo -e "\033[35m > Processing $$g ... \033[0m"; \
	  base=$$(basename "$$g" .json); \
	  python3 ./scripts/subgraph-matching.py \
	    --log 20 \
	    --mappingStorage "none" \
	    --queryGraph "./patterns/topology/nx_query_graph.json" \
	    --dataGraph "$$g" \
	    --destDir ${OUT_DIR}/kg2pg/topology ; \
	  echo -e "\033[35m > Give a name to the mappings file ... \033[0m" ; \
	  mv ${OUT_DIR}/kg2pg/topology/vf2pp_mappings.json ${OUT_DIR}/kg2pg/topology/vf2pp_mappings_$$base.json ; \
	done
	@echo -e "\033[35m > Done  \033[0m"

eval-vf2pp-pattern-unreach:
	@echo -e "\033[35m > Subgraph matching using VF2++ for the 'unreach' query graph  \033[0m"
	@mkdir -p ${OUT_DIR}/kg2pg/unreach
	@for g in $(EVAL_GRAPH_LIST_NX); do \
	  echo -e "\033[35m > Processing $$g ... \033[0m"; \
	  base=$$(basename "$$g" .json); \
	  python3 ./scripts/subgraph-matching.py \
	    --log 20 \
	    --mappingStorage "none" \
	    --queryGraph "./patterns/unreach/nx_query_graph.json" \
	    --dataGraph "$$g" \
	    --destDir ${OUT_DIR}/kg2pg/unreach ; \
	  echo -e "\033[35m > Give a name to the mappings file ... \033[0m" ; \
	  mv ${OUT_DIR}/kg2pg/unreach/vf2pp_mappings.json ${OUT_DIR}/kg2pg/unreach/vf2pp_mappings_$$base.json ; \
	done
	@echo -e "\033[35m > Done  \033[0m"

# ----------------------------------------------------------------------
# --- Utilities --------------------------------------------------------

virtdb-start:	## Start a Virtuoso instance on localhost
	@echo -e "\033[35m > Create the virtdb directory  \033[0m"
	mkdir -p virtdb
	@echo -e "\033[35m > Start a Virtuoso instance on localhost  \033[0m"
	@docker run \
		--name virtdb \
		--interactive \
		--rm \
		-d \
		--tty \
		--env DBA_PASSWORD=mysecret \
		--publish 1111:1111 \
		--publish  8890:8890 \
		--volume `pwd`/virtdb:/database \
		--volume `pwd`/scripts:/mnt \
		${VIRTDB_IMG}
	@echo -e "\033[35m > The Virtuoso instance is now available on http://localhost:8890 with admin account dba/mysecret \033[0m"

	@echo -e "\033[35m > Done  \033[0m"

virtdb-config:	## Configure the Virtuoso instance on localhost
	@echo -e "\033[35m > Send configuration file \033[0m"
	docker exec \
	  --interactive \
	  virtdb isql localhost dba mysecret /mnt/virtdb-config.sql || true
	@echo -e "\033[35m > Done  \033[0m"

virtdb-stop:	## Stop the Virtuoso instance on localhost
	@echo -e "\033[35m > Stop the Virtuoso instance on localhost  \033[0m"
	@docker container stop virtdb
	@echo -e "\033[35m > Done  \033[0m"

push-kg-virtdb: virtdb-config  ## Push KG data to local virtdb instance
	@echo -e "\033[35m > Push KG data to local virtdb instance \033[0m"
	@for g in $(GSTORE_GRAPH_LIST); do \
	    echo -e "\n\n*** ${YELLOW}Processing $g ${NC}***\n"; \
	    curl --digest --user dba:mysecret --verbose --url "http://localhost:8890/sparql-graph-crud-auth?graph-uri=urn:graph:update:pygraft-gen" -T $$g ;\
	done
	@echo -e "\033[35m > Done  \033[0m"

start-sparqlworks-firefox: ## Open a local SPARQLWorks instance using Firefox
	@echo -e "\033[35m > Stop the RDF4J instance on localhost  \033[0m"
	firefox ../lib/sparqlworks/sparqlworks-0.7.0/sparqlworks.html
	@echo -e "\033[35m > Done  \033[0m"

# --- EOF --------------------------------------------------------------
