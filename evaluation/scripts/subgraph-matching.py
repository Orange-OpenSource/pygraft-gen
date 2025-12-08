#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
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
# -----------------------------------------------------------------------------
# Subgraph matching using the VF2++ algorithm with nt4python (ntlib)
# Created by : Lionel TAILHARDAT and Morgan CHOPIN, 2025-06
# -----------------------------------------------------------------------------

# =============================================================================
# Imports
import argparse
import json
import logging

from nt4python import *

# =============================================================================
# Shared variables
home = os.path.expanduser("~")
destdir_default = home

mappingstorage_default = "memory"

class_prop_default = "class"
iri_prop_default = "iri"

logBaseName = "orange.ntlib.subgraph-matching"
logFileName = logBaseName + ".log"

sUsage_description = "Subgraph matching using the VF2++ algorithm from the NetworkTools / Lemon library."
sUsage_exec = "python3 .\subgraph-matching.py --queryGraph subgraph-matching_simple_kg_query.json --dataGraph subgraph-matching_simple_kg_data.json --destDir ."

parser = argparse.ArgumentParser(
    description=sUsage_description,
    epilog="CLI example: " + "\n" + sUsage_exec + "\n",
)

parser.add_argument(
    "--queryGraph",
    action="store",
    help="Path to the query graph file (NetworkX graph object), the graph to find embeddings for in the data graph.",
)

parser.add_argument(
    "--dataGraph",
    action="store",
    help="Path to the data graph file (NetworkX graph object), the graph that might embed the query graph.",
)

parser.add_argument(
    "--classProp",
    action="store",
    default=class_prop_default,
    help="Property name, in both the query and data graphs, for the node class, i.e. the label to match (default : "
         + class_prop_default
         + ")",
)

parser.add_argument(
    "--iriProp",
    action="store",
    default=iri_prop_default,
    help="Property name, in both the query and data graphs, for the node IRIs (default : "
    + iri_prop_default
    + ")",
)

parser.add_argument(
    "--destDir",
    action="store",
    default=destdir_default,
    help="Folder for saving results (default: "
         + destdir_default
         + ")",
)

parser.add_argument(
    "--mappingStorage",
    choices=['memory', 'file', 'none'],
    action="store",
    default=mappingstorage_default,
    help="Method to save mappings, 'memory' means mappings will be kept in memory until serialized to the vf2pp_mappings.json file; 'file' means a separate file for each mapping found is used (which can help to avoid memory overload); 'none' means no serialization is done. Default: "
         + mappingstorage_default
         + ".",
)

parser.add_argument(
    "--mappingLimit",
    type=int,
    action="store",
    default=-1,
    help="Number of mappings to find and serialize (default: -1, i.e. infinite).",
)

parser.add_argument(
    "--log",
    type=int,
    choices=[10, 20, 30, 40, 50],
    action="store",
    default=20,
    help="Logging verbosity level (default: 20) : DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50",
)

# Fetching CLI arguments
args = parser.parse_args()


# =============================================================================
# Functions

def save_data(data_to_save,
              destination_filename,
              destination_dir):
    """
    Elementary function to save some data to jSon file
    :param data_to_save: the data set, e.g. Dict()
    :param destination_filename: file name, e.g. String()
    :param destination_dir: directory, e.g. String()
    :return: True
    """
    out_filepath = os.path.join(
        destination_dir,
        destination_filename
    )
    out_file = open(
        out_filepath,
        "w")
    json.dump(data_to_save,
              out_file,
              indent=4)
    out_file.close()
    logging.info("SAVE:out_file=%s", out_filepath)
    return True


# =============================================================================
# Initializing loggers
loggingFormatString = (
    "%(asctime)s:%(levelname)s:%(threadName)s:%(funcName)s:%(message)s"
)
logging.basicConfig(
    format=loggingFormatString,
    level=args.log  # DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50"
)

logging.info("INIT")

# =============================================================================
# Preparing for loading the data graph
data_graph = nt_digraph_create()

# Creating graph props accessors
# Note that we need to set maps in query_graph_props before calling nt_digraph_load_networkx_prop
# as query_graph_props is functioning like a filter: we define which maps we would like to read,
# and then, when the graph is loaded, nt_digraph_load_networkx_prop() queries
# query_graph_props() to know if it must store the property in a map
data_graph_props = nt_gprops_create()
data_graph_nodeprop_class = nt_nodemap_str_create(data_graph)  # NtNodeMapString
nt_gprops_set_nodemap_str(data_graph_props, args.classProp, data_graph_nodeprop_class)

data_graph_nodeprop_iri = nt_nodemap_str_create(data_graph)  # NtNodeMapString
nt_gprops_set_nodemap_str(data_graph_props, args.iriProp, data_graph_nodeprop_iri)

# Loading the data graph
if nt_digraph_load_networkx_prop(args.dataGraph, data_graph, data_graph_props) != NT_SUCCESS:
    logging.error("INIT:Loading the data graph '%s' failed!", args.dataGraph)
    exit()
logging.info(
    "INIT:nt_digraph_node_num(data_graph)=%s:nt_digraph_arc_num(data_graph)=%s",
    nt_digraph_node_num(data_graph),
    nt_digraph_arc_num(data_graph)
)

# -----------------------------------------------------------------------------
# Preparing for loading the query graph
query_graph = nt_digraph_create()

# Creating graph props accessors
query_graph_props = nt_gprops_create()
query_graph_nodeprop_class = nt_nodemap_str_create(query_graph)
nt_gprops_set_nodemap_str(query_graph_props, args.classProp, query_graph_nodeprop_class)

query_graph_nodeprop_iri = nt_nodemap_str_create(query_graph)
nt_gprops_set_nodemap_str(query_graph_props, args.iriProp, query_graph_nodeprop_iri)

# Loading the query graph
if nt_digraph_load_networkx_prop(args.queryGraph, query_graph, query_graph_props) != NT_SUCCESS:
    logging.error("INIT:Loading the query graph '%s' failed!", args.queryGraph)
    exit()
logging.info(
    "INIT:nt_digraph_node_num(query_graph)=%s:nt_digraph_arc_num(query_graph)=%s",
    nt_digraph_node_num(query_graph),
    nt_digraph_arc_num(query_graph)
)

# =============================================================================
# Subgraph matching query => data
# the fastest way, but it doesn't allow for browsing results
# Should return True
# iso = nt_vf2pp_iso(query_graph, data_graph)
# iso_count = nt_vf2pp_count(query_graph, data_graph)
# logging.info("query/data:nt_vf2pp_iso=%s:iso_count=%s", iso, iso_count)

# Subgraph matching data => query
# the fastest way, but it doesn't allow for browsing results
# Should return False
# iso = nt_vf2pp_iso(data_graph, query_graph)
# iso_count = nt_vf2pp_count(data_graph, query_graph)
# logging.info("data/query:nt_vf2pp_iso=%s:iso_count=%s", iso, iso_count)

# =============================================================================
# Subgraph matching query => data using a VF2++ object to browse results

# Initializing a vf2pp object
vf2pp_object = nt_vf2pp_create(query_graph, data_graph)

# To allow vf2pp to compute node label-based embeddings, we need to tag
# both the query and data graphs with a class label value index.
# In the following, we first build a property name <=> property index table.
# Then we tag the nodes with a class index using the two following vf2pp methods
# // Set the label 'l' to node 'u' in the graph g1 to be embedded
# NTAPI void nt_vf2pp_set_label1(NtVf2pp vf2pp, NtNode u, int l);
# // Set the label 'l' to node 'u' in the graph g2 that g1 will be embedded into
# NTAPI void nt_vf2pp_set_label2(NtVf2pp vf2pp, NtNode u, int l);

# --- Build the class list by iterating over the query_graph nodes ---

node_class_list = []

logging.info("LABELING:Building the class list from the query_graph: starting...")
node = nt_digraph_first_node(query_graph)
while nt_digraph_is_node_valid(query_graph, node):
    # Get the class of the current node
    # nt_nodemap_str_get retuns an object of type bytes
    node_class = nt_nodemap_str_get(query_graph_nodeprop_class, node).decode("utf-8")
    # Add the class to the list if relevant
    if node_class not in node_class_list:
        node_class_list.append(node_class)
    # Go to the next node
    node = nt_digraph_next_node(query_graph, node)
logging.debug(
    "LABELING:Building the class list from the query_graph: done, node_class_list = %s",
    node_class_list
)

# --- Build the class list by iterating over the data_graph nodes ---

logging.info("LABELING:Building the class list from the data_graph: starting...")
node = nt_digraph_first_node(data_graph)
while nt_digraph_is_node_valid(data_graph, node):
    # Get the class of the current node
    # nt_nodemap_str_get retuns an object of type bytes
    node_class = nt_nodemap_str_get(data_graph_nodeprop_class, node).decode("utf-8")
    # Add the class to the list if relevant
    if node_class not in node_class_list:
        node_class_list.append(node_class)
    # Go to the next node
    node = nt_digraph_next_node(data_graph, node)
logging.debug(
    "LABELING:Building the class list from the data_graph: done, node_class_list = %s",
    node_class_list
)

# --- Labeling the nodes by iterating over the query_graph nodes ---

logging.info("LABELING:Labeling the query_graph nodes based on the node_class_list: starting...")
node_iter_count = 0
node = nt_digraph_first_node(query_graph)
while nt_digraph_is_node_valid(query_graph, node):
    node_iter_count += 1

    # Get the current node class and its corresponding class index
    node_class = nt_nodemap_str_get(query_graph_nodeprop_class, node).decode("utf-8")
    node_class_index = node_class_list.index(node_class)

    # Set the class index on the current node
    nt_vf2pp_set_label1(
        vf2pp_object,
        node,
        node_class_index
    )

    node_iri = nt_nodemap_str_get(query_graph_nodeprop_iri, node).decode("utf-8")
    logging.debug(
        "LABELING:vf2pp_set_label1(query_graph):NtNode(%s)=%s:node_iri='%s':node_class='%s':node_class_index=%s",
        node_iter_count,
        node,
        node_iri,
        node_class,
        node_class_index
    )

    # Go to the next node
    node = nt_digraph_next_node(query_graph, node)

logging.debug(
    "LABELING:Labeling the query_graph nodes based on the node_class_list: done, node_iter_count=%s",
    node_iter_count
)

# --- Labeling the nodes by iterating over the data_graph nodes ---

logging.info("LABELING:Labeling the data_graph nodes based on the node_class_list: starting...")
node_iter_count = 0
node = nt_digraph_first_node(data_graph)
while nt_digraph_is_node_valid(data_graph, node):
    node_iter_count += 1

    # Get the current node class and its corresponding class index
    node_class = nt_nodemap_str_get(data_graph_nodeprop_class, node).decode("utf-8")
    node_class_index = node_class_list.index(node_class)

    # Set the class index on the current node
    nt_vf2pp_set_label2(
        vf2pp_object,
        node,
        node_class_index
    )

    node_iri = nt_nodemap_str_get(data_graph_nodeprop_iri, node).decode("utf-8")
    logging.debug(
        "LABELING:nt_vf2pp_set_label2(data_graph):NtNode(%s)=%s:node_iri='%s':node_class='%s':node_class_index=%s",
        node_iter_count,
        node,
        node_iri,
        node_class,
        node_class_index)

    # Go to the next node
    node = nt_digraph_next_node(data_graph, node)

logging.debug(
    "LABELING:Labeling the data_graph nodes based on the node_class_list: done, node_iter_count=%s",
    node_iter_count
)

# =============================================================================
# Running the VF2++ algorithm to find mappings (subgraph matching)

# vf2pp is both an object allowing for calling vf2pp algorithms
# and an iterator for the mappings found
logging.info("VF2PP(query=>data):Running the algorithm ...")
mapping_found = nt_vf2pp_run(vf2pp_object)
logging.info("VF2PP(query=>data):mapping_found=%s", mapping_found)

mappings_dict = {}  # To save mappings found
mappings_dict["mappings"] = {}

# Types of mapping used in the Vf2++ algorithm
#
# Subgraph isomorphism
#   NT_VF2PP_SUBGRAPH = 0,
# Induced subgraph isomorphism
#   NT_VF2PP_INDUCED = 1,
#
# Graph isomorphism
# If the two graphs have the same number of nodes, then it is
#  equivalent to INDUCED, and if they also have the same
#  number of edges, then it is also equivalent to SUBGRAPH.
# However, using this setting is faster than the other two options.
#   NT_VF2PP_ISOMORPH = 2
mapping_type = nt_vf2pp_get_mapping_type(vf2pp_object)
logging.info("VF2PP(query=>data):mapping_type=%s", mapping_type)
mappings_dict["type"] = mapping_type

# =============================================================================
# Browsing mappings
logging.info("VF2PP(query=>data):Browsing mappings limit=%s", args.mappingLimit)
logging.info("VF2PP(query=>data):Browsing mappings:starting ...")
mapping_iter_count = 0  # For logging and saving mappings

# For each mapping found
while mapping_found:
    mapping_iter_count += 1
    logging.info("MAPPING(%s):Browsing query graph nodes ...", mapping_iter_count)

    node_iter_count = 0
    node = nt_digraph_first_node(query_graph)

    mapping_details = []

    # For each query graph node
    while nt_digraph_is_node_valid(query_graph, node):
        node_iter_count += 1

        query_graph_node_iri = nt_nodemap_str_get(query_graph_nodeprop_iri, node).decode("utf-8")
        query_graph_node_class = nt_nodemap_str_get(query_graph_nodeprop_class, node).decode("utf-8")

        # Fetching mapped node in the data graph for the current mapping
        data_graph_mapped_to_node = nt_vf2pp_get_mapped(vf2pp_object, node)

        data_graph_node_iri = nt_nodemap_str_get(data_graph_nodeprop_iri, data_graph_mapped_to_node).decode("utf-8")
        data_graph_node_class = nt_nodemap_str_get(data_graph_nodeprop_class, data_graph_mapped_to_node).decode("utf-8")

        # Making user-friendly representation
        VqVd_class_representation = "({})-->({})".format(query_graph_node_class, data_graph_node_class)
        VqVd_iri_representation = "({})-->({})".format(query_graph_node_iri, data_graph_node_iri)

        # For storing results
        mapping_details.append({
            "query_graph_node_class": query_graph_node_class,
            "query_graph_node_iri": query_graph_node_iri,
            "data_graph_node_class": data_graph_node_class,
            "data_graph_node_iri": data_graph_node_iri,
            "VqVd_class": VqVd_class_representation,
            "VqVd_iri": VqVd_iri_representation,
        })

        logging.debug(
            "MAPPING(%s):node_iter_count=%s:nt_vf2pp_get_mapped:VqVd=%s||%s",
            mapping_iter_count,
            node_iter_count,
            VqVd_class_representation,
            VqVd_iri_representation
        )

        # Get the next query graph node
        node = nt_digraph_next_node(query_graph, node)

    # Store the mapping for post-processing
    if args.mappingStorage == "memory":
        mappings_dict["mappings"][mapping_iter_count] = mapping_details
    elif args.mappingStorage == "file":
        mapping_details_filename = "vf2pp_mappings_{}.json".format(mapping_iter_count)
        save_data(mapping_details, mapping_details_filename, args.destDir)
    elif args.mappingStorage == "none":
        pass
    else:
        raise ValueError

    # Leave loop if mappingLimit is reached
    if args.mappingLimit > 0 and mapping_iter_count == args.mappingLimit:
        logging.info(
            "MAPPING(%s):mappingLimit reached, leaving the mapping browsing loop ...",
            mapping_iter_count
        )
        break

    # Get the next mapping. Return false if no mapping is found.
    # nt_vf2pp_run() MUST be called before this function.
    mapping_found = nt_vf2pp_next(vf2pp_object)

logging.info("VF2PP(query=>data):Browsing mappings:done, mapping_iter_count=%s", mapping_iter_count)

# Saving mapping results
mappings_dict["queryGraph"] = args.queryGraph
mappings_dict["dataGraph"] = args.dataGraph
mappings_dict["classProp"] = args.classProp
mappings_dict["iriProp"] = args.iriProp
mappings_dict["classList"] = node_class_list
mappings_dict["mappingCount"] = mapping_iter_count
save_data(mappings_dict, "vf2pp_mappings.json", args.destDir)

# =============================================================================
nt_vf2pp_free(vf2pp_object)
nt_digraph_free(data_graph)
nt_digraph_free(query_graph)

logging.info("END")

# === EOF ======================================================================
