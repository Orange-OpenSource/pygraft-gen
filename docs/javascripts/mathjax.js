/*
 * Software Name: PyGraft-gen
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
 *
 * Authors: See CONTRIBUTORS.txt
 * Software description: A RDF Knowledge Graph stochastic generation solution.
 *
 */

window.MathJax = {
  tex: {
    inlineMath: [["$", "$"]],
    displayMath: [["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
