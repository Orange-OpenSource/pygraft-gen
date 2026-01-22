# PyGraft-gen / evaluation

This folder contains the datasets and tooling for exemplifying or testing the **PyGraft-gen** project.

## Reference ontologies

The ontologies used for evaluating the PyGraft-gen framework are available in the [ontologies/](./ontologies) subfolder.

## Subgraph matching techniques

Four families of subgraph matching techniques typically emerge:

1. Retrieval-based : SPARQL
2. Graph traversal-based : SHACL
3. Label matching: VF2++
4. Model-based: graph embeddings, graph neural networks

For 1, 2, and 3, generally speaking, we provide a *query graph* (pattern, shape, graph instance) to look for in the *data graph*.
We go with these three families as the query graph directly corresponds to the use case.

For 4, it needs training. As we rely on very few example, we don't go with this family of techniques.

### Patterns

Patterns (use cases) for subgraph matching are implemented in the [./patterns](./patterns) subfolder.
We provide the three patterns depicted in the following figure. We refer to these as *resilience*, *topology*, and *unreach*.
![uc_full.png](../docs/assets/images/uc_full.png)

#### Editing patterns

Within each [./patterns](./patterns) subfolder,

- Edit the `uc_diagram.drawio` file using the draw.io tool and the Chowlk notation
- Convert the `uc_diagram.drawio` to an RDF knowledge graph using the Chowlk converter, using one of the following options:
  - Web-based conversion: browse to the Chowlk converter and drop the `uc_diagram.drawio` file, then save the result in the `uc.ttl` file.
  - API call: send the `uc_diagram.drawio` file using the `make convert-diagram` CLI command.
- Apply any post-processing of your will on the `uc.ttl` file.

Useful Links:

- draw.io : https://www.drawio.com/
- Chowlk notation : https://chowlk.linkeddata.es/notation
- Chowlk converter : https://chowlk.linkeddata.es/
- Chowlk git : https://github.com/oeg-upm/Chowlk
- JQ : https://www.baeldung.com/linux/jq-command-json


### Subgraph matching using SPARQL

For each pattern, preparing the query:

- Implement a SPARQL query matching the pattern in a `uc_query.sparql` and `uc_query_count.sparql` file (e.g. [patterns/resilience/uc_query.sparql](patterns/resilience/uc_query.sparql)).
- Convert the query into its graphical form to check it corresponds to the pattern: in the pattern subfolder, run `make sparql-query-viz` and open the resulting `uc_query.html` file in your favorite Web browser.
- Test the query on both the pattern in RDF form and the extended pattern in RDF form: in the pattern subfolder, run `make eval-sparql` and analyse result files to check for the correct presence and number of pattern occurrence.
- Implement an "evaluation goal" in the current folder's [makefile](makefile), e.g. `eval-sparql-pattern-resilience`.

Running the evaluation:

- Assuming that you already have generated graphs to test, add their filepath to the `EVAL_GRAPH_LIST` in the current folder's [makefile](makefile).
- Call the "evaluation goal", e.g. `make eval-sparql-pattern-resilience` for a specific pattern, or `make eval-sparql-pattern` to evaluate all patterns.
- Analyse results in the [./out/patterns](./out/patterns) subfolder.
- (optional) Call complementary queries using the same process by editing the `EVAL_SPARQL_QUERY_FILE` in the current folder's [makefile](makefile).

### Subgraph matching using SHACL

In the SHACL case, SHACL’s limited recursion (nodes of nodes, etc.) hampers expressing use cases.
For the "*topology*" UC, capturing patterns notably requires constraints that violate SHACL’s [well-formed property path](https://www.w3.org/TR/shacl/#property-paths) requirement.
Further we observe limitations with sh:rule which cannot handle precise indication of the matching entity.

Consequently, SHACL detection is set aside for future work.

### Subgraph matching using VF2++

For this technique, we will first use SPARQL and Python for KG2PG translation of the patterns, then use Python and VF2++/LEMON/NTLib for subgraph matching.

KG2PG:

- Translate the **query graph** (use case graph pattern) to a NetworkX format: within the pattern subfolder, call `make kg2pg-query-graph`
  - This yields to a `nx_query_graph.json` file in the pattern folder
  - See [scripts/kg2pg_querygraph_shell_example.sh](scripts/kg2pg_querygraph_shell_example.sh) for a shell script example of the process.
- (optional) Translate the *extended* data graph to a NetworkX format: within the pattern subfolder, call `make kg2pg-data-graph-extended`
  - This yields to a `nx_data_graph_extended.json` file in the pattern folder
  - It can be used to test the subgraph matching technique: call `make eval-vf2pp` within the pattern subfolder and then browse results in the `vf2pp_mappings.json` file. 
- Assuming that you already have generated graphs to test, add their filepath to the `EVAL_GRAPH_LIST` in the current folder's [makefile](makefile).
- Translate the **data graphs** (generated graphs) to a NetworkX format: call `make kg2pg-data-graph` from the current folder
  - See [scripts/kg2pg_datagraph_shell_example.sh](scripts/kg2pg_datagraph_shell_example.sh) for a shell script example of the process.

Subgraph matching:

- Install the NTLIB tools.
- Call the "evaluation goal", e.g. `make eval-vf2pp-pattern-resilience` for a specific pattern, or `make eval-vf2pp-pattern` to evaluate all patterns.
- Analyse results in the [./out/patterns](./out/patterns) subfolder.

