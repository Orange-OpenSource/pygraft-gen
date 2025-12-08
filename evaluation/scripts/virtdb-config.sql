/*
 *  Software Name: PyGraft-gen
 *  SPDX-FileCopyrightText: Copyright (c) Orange SA
 *  SPDX-License-Identifier: MIT
 *
 *  This software is distributed under the MIT license,
 *  the text of which is available at https://opensource.org/license/MIT/
 *  or see the "LICENSE" file for more details.
 *
 *  Authors: See CONTRIBUTORS.txt
 *  Software description: A RDF Knowledge Graph stochastic generation solution.
 *
 */

-- Install the FCT module
vad_install ('../vad/fct_dav.vad', 0);

-- Enable SPONGE for the SPARQL account (see: http://localhost:8890/sparql?help=enable_sponge)
USER_GRANT_ROLE ('SPARQL', 'SPARQL_SPONGE');
USER_GRANT_ROLE ('SPARQL', 'SPARQL_UPDATE');

-- Declare common namespaces
DB.DBA.XML_SET_NS_DECL ('noria', 'https://w3id.org/noria/ontology/', 2);
