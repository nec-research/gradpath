# coding: utf-8
#        Gradient Rollback
#
#   File:     gradpath_explanations.py
#
# NEC Laboratories Europe GmbH, Copyright (c) 2024, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#
"""Generate path-based explanations."""
from collections import defaultdict
import numpy as np
from gr.utils.utils import softmax

def find_paths_and_triples(lookup_table, start, end, nodelimit=4, path=[], triples_start=[],
                           triples_path=[]):
    path = path + [start]
    if len(triples_start) > 0:
        triples_path = triples_path + [triples_start]

    if len(path) >= nodelimit and start != end:
        return [], []
    if start == end:
        return [path], [triples_path]

    paths = []
    triples_paths = []

    # Get next nodes connecting start, direction is not considered.
    # This means start can be subject or object of a triple.
    # There might be multiple triples between start and another node.
    triples_relate_start = lookup_table['e:' + start]
    next_nodes = defaultdict(list)
    for triple in triples_relate_start:
        tem = triple.split(':')
        e = set([tem[0], tem[2]]) - {start}
        if len(e) > 0:
            next_nodes[next(iter(e))].append(triple)

    for node in next_nodes.keys():
        if node not in path:
            # newpaths = find_simple_paths(lookup_table, node, end, nodelimit, path)
            newpaths, newtriples_paths = find_paths_and_triples(lookup_table, node, end, nodelimit,
                                                                path, next_nodes[node],
                                                                triples_path)
            for newpath in newpaths:
                paths.append(newpath)
            for newtriples_path in newtriples_paths:
                triples_paths.append(newtriples_path)
    return paths, triples_paths


def get_paths_from_triple_sequence(triple_sequence, start=None, idx=0, path=[]):
    """
    Find all paths connecting two given nodes. The number of all nodes in each path, including start and end, can not be larger than a given limit.
    A simple path is a path that any node can not appear more than one time.
    """
    if idx > 0:
        path = path + [start]
    if idx >= len(triple_sequence):
        return [path]

    paths = []

    # Get next nodes connecting start
    for node in triple_sequence[idx]:
        newpaths = get_paths_from_triple_sequence(triple_sequence, node, idx + 1, path)
        for newpath in newpaths:
            paths.append(newpath)
    return paths


def get_all_paths(lookup_table, start, end, nodelimit=4):
    """
    Find all paths connecting two given nodes. The number of all nodes in each path, including start and end, can not be larger than a given limit.
    A simple path is a path that any node can not appear more than one time.
    """
    paths, triples_paths = find_paths_and_triples(lookup_table, start, end, nodelimit)
    res = defaultdict(list)
    for triple_sequence in triples_paths:
        k = 'length_' + str(len(triple_sequence))
        ps = get_paths_from_triple_sequence(triple_sequence)
        res[k].extend(ps)
    return res


def get_explanations(model_holder, influence_map, test_tripe, paths_all, length=3):
    """Get explanations of path length = length.
    """
    explanations = defaultdict(float)

    ps = paths_all['length_' + str(length)]

    # get original probability of test triple
    node_idx_test, relation_idx_test, gold_idx_test = model_holder.triple_to_index(test_tripe)
    logits_test_orig = model_holder.model.predict_tail((node_idx_test, relation_idx_test)).numpy()
    probs_test_orig = softmax(logits_test_orig, axis=1)

    for path in ps:
        # for a path, get influence of each triple, as well as indexes of the involved s,r,o
        indexes_list = []
        influence_list = []
        for triple in path:
            triple_elements = triple.split(':')
            influence_triple = influence_map[triple]
            influence_triple_head = influence_triple['e:' + triple_elements[0]]
            influence_triple_rel = influence_triple['r:' + triple_elements[1]]
            influence_triple_tail = influence_triple['e:' + triple_elements[2]]

            triple_head_idx, triple_relations_idx, triple_tail_idx = model_holder.triple_to_index(
                triple_elements)

            indexes_list.append((triple_head_idx, triple_relations_idx, triple_tail_idx))
            influence_list.append(
                (influence_triple_head, influence_triple_rel, influence_triple_tail))

        # get original prob. of each triple
        probs_triples_orig = []
        for triple in indexes_list:
            logits_triple = model_holder.model.predict_tail(triple[:-1]).numpy()
            probs_triple = softmax(logits_triple, axis=1)
            probs_triples_orig.append(probs_triple[0][triple[-1]])

        # get original prob. of test triple
        probs_triples_orig.append(probs_test_orig[0][gold_idx_test])

        # remove influence of the path (all triples in the path)
        model_holder.model.update_according_to_influence_list(indexes_list, influence_list)

        # get new prob. of each triple
        probs_triples_new = []
        for triple in indexes_list:
            logits_triple = model_holder.model.predict_tail(triple[:-1]).numpy()
            probs_triple = softmax(logits_triple, axis=1)
            probs_triples_new.append(probs_triple[0][triple[-1]])

        # get new prob. of test triple
        logits_test_new = model_holder.model.predict_tail(
            (node_idx_test, relation_idx_test)).numpy()
        probs_test_new = softmax(logits_test_new, axis=1)
        probs_triples_new.append(probs_test_new[0][gold_idx_test])

        # revert to original weights, ready for the next for loop step
        model_holder.model.revert_weights()

        # if new probs smaller than the original ones, then the path is an explanation
        # the score of the explanation is sum of (log(p_orig) - log(p_new)) over all triples
        tem1 = np.array(probs_triples_orig)
        tem2 = np.array(probs_triples_new)
        if np.all(np.greater_equal(tem1, tem2)):
            score_path = (np.sum(np.log(tem1)) - np.sum(np.log(tem2))) / length
            explanations[' '.join(path)] = [length, score_path.item(), tem1.tolist(), tem2.tolist()]

    return explanations
