#!/usr/bin/env python3
#

"""
MeerKAT Proposal Reviewer Assignment
Richard Armstrong 
February 2023

Reviewer assignment following Taylor, 2008 for a set of MeerKAT Proposals with some
specific requirements:

* Reviewers must have between 10 and 20 proposals to review. 
* Some reviewers have requested a maximum of 10 proposals.
* Proposals are reviewed exactly 4 times each.
* Exclude reviewers from reviewing proposals they are involved in.
"""

import argparse, logging, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

from numpy import genfromtxt


def main(args, loglevel):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)

    # load the list of proposals and their categories into a dataframe
    df_cat = pd.read_csv("csv/proposal_scientific_categories_MKT-22.csv", sep=";")
    df_cat["Category_List"] = df_cat["CATEGORY LIST"].apply(lambda x: eval(x))
    n_p = n_proposals = len(df_cat)

    # load the reviewer conflict dataframe
    df_conflict = pd.read_csv("csv/science_cross_check_MKT-22.csv", sep=";")
    df_conflict["Proposals"] = df_conflict["PROPOSAL LIST"].apply(
        lambda x: x.strip("][").split(",")
    )

    # load observation categories for proposals and reviewers
    roe = np.genfromtxt(
        "csv/reviewer_observation_expertise_MKT-22.csv", delimiter=",", skip_header=1
    )
    poc = np.genfromtxt(
        "csv/proposal_observation_categories_MKT-22.csv", delimiter=",", skip_header=1
    )

    # load the self-identified reviewer competency scores per catagory into a dataframe
    df_rev_score = pd.read_csv("csv/reviewer_scientific_expertise_MKT-22.csv")
    # .. and also a numpy array
    rev_score = genfromtxt(
        "csv/reviewer_scientific_expertise_MKT-22.csv", delimiter=","
    )

    # Create vector with max number of reviews per reviewer
    n_r = n_reviewers = len(df_rev_score)
    reviewers_props = 20 * np.ones(n_r)
    df_rev_score[
        "N_max"
    ] = reviewers_props  # Make column with max number of reviews for each reviewer

    scientific_categories = pd.read_csv("csv/scientific_categories_MKT-22.csv")
    scientific_categories.index = scientific_categories["CATEGORY ID"]

    df_rev_score.columns = (
        ["CATEGORY ID"] + list(range(1, scientific_categories.shape[0] + 1)) + ["N_max"]
    )

    og_rev_idx = np.array(df_rev_score["CATEGORY ID"])

    # ---- DERIVED
    # create a binary numpy array(/mask) from the reviewer conflict dataframe:
    # i.e. set to 0 if conflict, else 1
    conflict_mask = np.ones((n_r, n_p))
    conflict_counter = 0
    logging.info(
        f"total number of conflicts in mask before: {conflict_mask.flatten().shape[0] - np.count_nonzero(conflict_mask)}"
    )

    rev_idx = np.array(df_rev_score["CATEGORY ID"].index)

    for row_idx, row in df_conflict.iterrows():
        rev_pos = int(*np.where(og_rev_idx == row[0]))
        for col_idx, prop in enumerate(df_conflict["Proposals"][row_idx]):
            prop_pos = df_cat[df_cat["PSS ID"] == prop].index[0]
            # print(prop, col_idx, prop_pos, rev_pos)
            conflict_mask[int(rev_pos)][prop_pos] = 0
            conflict_counter += 1

    logging.info(
        f"total number of conflicts in mask after: {conflict_mask.flatten().shape[0] - np.count_nonzero(conflict_mask)}"
    )
    logging.debug(f"conflict_counter: {conflict_counter}")
    logging.debug(f"sum of conflicts: {int(np.sum(n_p-np.sum(conflict_mask, axis=1)))}")

    # create a binary array (/mask) of reviewer expertise > .0
    zeroes_mask = np.where(np.array(df_rev_score.transpose()) > 0.0, 1, 0)[1:-1]
    np.count_nonzero(zeroes_mask), zeroes_mask.shape

    # create a numerical array of the reviewer scores dataframe
    rev_scores = np.array(df_rev_score.transpose())[1:-1]
    np.count_nonzero(rev_scores), rev_scores.shape[0], rev_scores.shape[1]

    '''# ------ PLOTS
    # Make a bar plot of all the proposal categories and numerical reviewer expertise
    all_topics_list = []
    all_topics_count = np.zeros(df_rev_score.transpose()[1:-1].shape[0])

    for idx, topic_set in enumerate(df_cat["Category_List"]):
        for topic in topic_set:
            all_topics_count[topic - 1] += 1
            all_topics_list.append(topic)

    fig, ax = plt.subplots()
    ax.bar(
        range(1, rev_scores.shape[0] + 1),
        [row.sum() for row in rev_scores],
        alpha=0.7,
        label="reviewer expertise",
        color="b",
    )
    ax.bar(
        range(1, rev_scores.shape[0] + 1),
        all_topics_count,
        label="category demand",
        color="r",
        width=0.4,
    )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.bar(
        range(1, rev_scores.shape[0] + 1),
        5
        * 10**2
        * np.array(all_topics_count)[0:13]
        / [row.sum() for row in rev_scores][0:13],
        width=0.2,
        label="relative stress",
        color="k",
    )
    fig.legend()
    plt.xlabel("category label")
    plt.savefig("png/relative_stress.png")

    # Make a bar plot of all the proposal categories and binary reviewer expertise
    fig, ax = plt.subplots()
    ax.bar(
        range(1, 14),
        [row.sum() for row in rev_scores],
        alpha=0.7,
        label="reviewer (total numerical) expertise",
        color="b",
    )
    ax.bar(
        range(1, rev_scores.shape[0] + 1),
        [row.sum() for row in zeroes_mask],
        label="reviewer 'non-zeros'",
        color="c",
    )
    ax.bar(
        range(1, rev_scores.shape[0] + 1),
        all_topics_count,
        label="category demand",
        color="r",
        width=0.4,
    )
    plt.xlabel("category label")
    fig.legend()
    plt.savefig("png/rev_expertise.png")
    '''

    # ---------------
    # AFFINITY MATRIX
    # ---------------

    # create the affinity matrix
    affinity = np.zeros((len(df_rev_score), len(df_cat)))

    logging.debug(
        "affinity matrix creation",
        affinity.shape,
        "zeros:",
        affinity.flatten().shape[0] - np.count_nonzero(affinity),
        "of",
        affinity.flatten().shape[0],
    )

    def logarithmic_weights(length, base=np.e, offset=1):
        raw_weights = np.array([1 / np.log(i + offset + 1) / np.log(base) for i in range(length)])
        normalized_weights = raw_weights / np.sum(raw_weights)
        return normalized_weights
    
    # set the affinity to the MAXimum of ANY keyword
    for row_idx, row in enumerate(affinity):
        # print(row_idx, row)
        # for cell_idx, cell in enumerate(row):
        for cell_idx, topic_set in enumerate(df_cat["Category_List"]):
            # print(topic_set, cell_idx)
            weights = logarithmic_weights(len(topic_set), base=np.e, offset=1)
            max = 0
            sum = 0
            for idx, topic in enumerate(topic_set):
                # print(topic)
                val = rev_scores[topic - 1][row_idx]
                max = np.maximum(max, val)
                sum += weights[idx] * val
                # if ((topic != 14) and (topic != 15)):
                # max = np.maximum(max, df_rev_score[topic][row_idx+1])
                # NOTE only consider the 13 subject categories
            print(max, sum)
            affinity[row_idx, cell_idx] = sum / 10  # normalise to between 0 and 1
            

    logging.debug(
        "set up affinities done",
        affinity.shape,
        "zeros:",
        affinity.flatten().shape[0] - np.count_nonzero(affinity),
        "of",
        affinity.flatten().shape[0],
    )

    # -----------

    # Set constraints in the affinity matrix :
    # i.e. 1. Maximum over observation categories
    #      2. Make sure there are not zeros in EVERY category,
    #      3. reviewer/proposal conflicts (science_cross_check)
    # by setting affinity to 0 (which is actually a soft constraint, but wusually suffices)

    # Observation Category
    rev_obs_exp = np.genfromtxt(
        "csv/reviewer_observation_expertise_MKT-22.csv", delimiter=",", skip_header=1
    )
    prop_obs_cat = np.genfromtxt(
        "csv/proposal_observation_categories_MKT-22.csv", delimiter=",", skip_header=1
    )[:, 1]

    for row_idx, row in enumerate(rev_obs_exp):
        for col_idx, col in enumerate(prop_obs_cat):
            if row[int(col)] == 0:
                affinity[row_idx][col_idx] = 0.0
    logging.debug(
        "Obs cat done",
        affinity.shape,
        "zeros:",
        affinity.flatten().shape[0] - np.count_nonzero(affinity),
        "of",
        affinity.flatten().shape[0],
    )

    # mask out conflicts of interest:
    affinity = affinity * conflict_mask
    logging.debug(
        "set conflicts done",
        affinity.shape,
        "zeros:",
        affinity.flatten().shape[0] - np.count_nonzero(affinity),
        "of",
        affinity.flatten().shape[0],
    )
    logging.info("affinity matrix creation done")

    # --------------------------------------------------------------------------------------------
    # convert arrays to a set of stored numpy arrays for the Linear Program (LP) optimisation code
    # --------------------------------------------------------------------------------------------

    # assign the maximum number of proposals per reviewer
    loads = np.genfromtxt("csv/science_max_reviews.csv", skip_header=1, delimiter=",")[
        :, 1
    ]

    # -------------
    n_rev = np.size(affinity, axis=0)
    n_pap = np.size(affinity, axis=1)

    a = affinity.flatten(order="C")
    b = np.ones(a.shape[0])

    Np = np.zeros((n_pap, n_pap * n_rev))
    Nr = np.zeros((n_rev, n_pap * n_rev))

    for pdx, p in enumerate(affinity.transpose()):
        for rdx, val in enumerate(p):
            if val > 0.0:
                Np[pdx][rdx * n_pap + pdx] = 1

    for rdx, r in enumerate(affinity):
        for pdx, val in enumerate(r):
            if val > 0.0:
                Nr[rdx][rdx * n_pap + pdx] = 1

    I = np.identity(n_pap * n_rev)
    N = np.vstack((Np, Nr, -Nr))
    K = np.vstack((Np, Nr, -Nr, I, -I))

    cp = args.COVERAGE * np.ones(n_pap)
    cr = loads
    crlb = args.LOADS_LB * np.ones(n_rev)
    c = np.concatenate((cp, cr, crlb))

    zeroes = np.zeros(I.shape[0])
    ones = np.ones(I.shape[0])
    d = np.concatenate((cp, cr, crlb, ones, zeroes))

    assert (Np @ a).shape == cp.shape
    assert (Nr @ a).shape == cr.shape
    assert (N @ a).shape == c.shape

    assert (I @ a).shape == ones.shape
    assert (-I @ a).shape == zeroes.shape

    # -- LP
    res = scipy.optimize.linprog(
        -a, A_ub=K, b_ub=d, bounds=(0, 1), options={"disp": True}
    )
    # res = scipy.optimize.linprog(-a, A_ub=N, b_ub=c, bounds=(0, 1), options={"disp": True}, integrality=3)

    # The assignment matrix is a binary array of dim n_reviewers * n_proposals
    assignment = res.x.reshape(affinity.shape)
    assignment = assignment.astype(int)

    ## Sanity check -- check that the trace of the affinity and assignment match that produced by LP
    logging.debug(
        f"Sanity check: trace of the affinity and assignment matrices match that \
         produced by LP: {np.sum(np.diag(np.matmul(np.transpose(affinity), assignment)))} \
         should be == {np.trace(np.matmul(np.transpose(affinity), assignment))}"
    )

    # Gene Matrix
    ################

    # The assignment matrix may be expressed as a 'gene matrix' (an inherited term): a dense array
    # of reviewers assigned to each proposal, (as opposed to the assignment matrix, which is a sparse binary
    # array of dim n_reviewers * n_proposals with nonzero entries for a positive assignment)
    gene_matrix = np.zeros((n_proposals, args.COVERAGE))
    gene_matrix_indexes = np.zeros((n_proposals, args.COVERAGE))

    # The reviewer_assignment is the transpose of the gene matrix: for each reviewer,
    # it is a list of their associated proposals
    reviewer_assignment = np.empty((n_reviewers), dtype=object)

    # set the first element of the reviewer assignment vector to be the original reviewer
    # index, as a string
    for idx, r in enumerate(reviewer_assignment):
        reviewer_assignment[idx] = str(og_rev_idx[idx])

    for col_idx, col in enumerate(assignment.transpose()):
        # set the gene matrix to the indices of each non-zero element in the assignment
        gene_matrix[col_idx] = np.nonzero(col)[0]
        # set up the correct original reviewer IDs
        for rev_idx, rev in enumerate(gene_matrix[col_idx]):
            gene_matrix_indexes[col_idx][rev_idx] = og_rev_idx[int(rev)]
        # augment the string with each fo the {n_min, n_max} proposals that reviewer should review
        for i in np.nonzero(col)[0]:
            reviewer_assignment[i] = (
                reviewer_assignment[i] + ", " + df_cat["PSS ID"][col_idx]
            )

    gene_matrix = gene_matrix.astype(int)
    gene_matrix_indexes = gene_matrix_indexes.astype(int)

    logging.debug(
        f'LP result, expressed as a set of (4) reviewers (by original ID) \
        per proposal: \n {np.column_stack((df_cat["PSS ID"], gene_matrix_indexes))}'
    )

    # Save proposal assignments and reviewer assignemnts to CSV files
    logging.info("Saving proposal assignments and reviewer assignments to CSV")
    pd.DataFrame(
        np.column_stack((df_cat["PSS ID"], gene_matrix_indexes.astype(int)))
    ).to_csv("csv/Proposal_assignment.csv")
    pd.DataFrame(np.column_stack((og_rev_idx, reviewer_assignment))).to_csv(
        "csv/Reviewer_assignment.csv"
    )
    '''
    # -----------
    # Fernando's Matrix and plots
    # -----------

    # max reviewer score matrix ('Fernando's Matrix')
    rev_max_arr = np.zeros((n_p, 4))
    hi_cont_arr = np.zeros((n_p, 4))
    # all reviewer score list ('Extended Fernando's List')
    all_assigned_affinities = []
    all_hicont_scores = []

    for cell_idx, topic_set in enumerate(
        df_cat["Category_List"]
    ):  # iterate over proposals
        rev_a = np.zeros(4)
        hi_a = np.zeros(4)
        for topic in topic_set:  # iterate over topic proposals
            if topic <= 13:
                for rev_idx, rev in enumerate(
                    gene_matrix[cell_idx]
                ):  # iterate over reviewers
                    rev_a[rev_idx] = np.maximum(
                        rev_a[rev_idx], rev_scores[topic - 1][rev]
                    )
                    all_assigned_affinities.append(rev_scores[topic - 1][rev])
            else:
                for rev_idx, rev in enumerate(
                    gene_matrix[cell_idx]
                ):  # iterate over reviewers
                    hi_a[rev_idx] = np.maximum(
                        hi_a[rev_idx], rev_scores[topic - 1][rev]
                    )
                    all_hicont_scores.append(rev_scores[topic - 1][rev])

        rev_max_arr[cell_idx] = rev_a
        hi_cont_arr[cell_idx] = hi_a

    logging.debug(
        "Fernando's matrix zeros: ",
        rev_max_arr.flatten().shape[0] - np.count_nonzero(rev_max_arr),
    )
    logging.debug(
        "Fernando's Extended zeros: ",
        len(all_assigned_affinities)
        - np.count_nonzero(np.array(all_assigned_affinities)),
    )

    fig, ax = plt.subplots()
    ax.hist(
        all_assigned_affinities,
        bins=int(np.max(all_assigned_affinities) - np.min(all_assigned_affinities)),
        histtype="stepfilled",
        linewidth=5,
        label="all category matches",
        color="k",
    )
    ax.hist(
        rev_max_arr.astype(int).flatten(),
        bins=np.max(rev_max_arr.astype(int).flatten())
        - np.min(rev_max_arr.astype(int).flatten()),
        histtype="step",
        linewidth=5,
        label="best category match",
        color="w",
    )
    fig.legend()

    logging.debug(
        "Extended: mean, median:",
        np.mean(all_assigned_affinities),
        np.median(all_assigned_affinities),
    )
    logging.debug(
        np.sum(all_assigned_affinities),
        len(all_assigned_affinities),
        np.sum(all_assigned_affinities) / len(all_assigned_affinities),
    )

    rev_max_arr = rev_max_arr.astype(int)
    logging.info(
        f"Fernando's matrix: mean, median, minimum sum: {np.median(rev_max_arr)}, {np.mean(rev_max_arr)}, {np.min(rev_max_arr.sum(axis=1))}"
    )
    logging.debug(
        "argmin: ",
        np.argmin(rev_max_arr.sum(axis=1)),
        ",sum: ",
        rev_max_arr.sum(axis=1),
    )
    logging.debug(np.column_stack((rev_max_arr, df_cat["PSS ID"])))
    plt.savefig("png/Fernandos_hist.png")

    plt.hist(
        rev_max_arr.sum(axis=1),
        histtype="step",
        bins=range(
            np.min(rev_max_arr.sum(axis=1)), 2 + np.max(rev_max_arr.sum(axis=1))
        ),
        linewidth=4,
        color="k",
        label="overall reviewer expertise per proposal",
    )
    plt.legend()
    plt.savefig("png/total_reviewer_expertise_per_proposal_hist.png")

    # column sums of the assignment: number of reviews per reviewer
    col_sum = []
    for el in assignment:
        col_sum.append(np.count_nonzero(el))

    plt.hist(
        col_sum,
        histtype="step",
        bins=range(np.min(col_sum), 2 + np.max(col_sum)),
        linewidth=4,
        color="k",
        label="number of reviewers with this many allocations",
    )
    plt.legend()
    plt.savefig("png/reviewers_hist.png")
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign reviewers to proposals.",
        epilog="As an alternative to the commandline, params can be placed in a file, one per line, and specified on the commandline like '%(prog)s @params.conf'.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-ll",
        "--loads-lb",
        help="Loads Lower Bound: minimum number of reviews per reviewer",
        dest="LOADS_LB",
        default=10,
    )
    parser.add_argument(
        "-c",
        "--coverage",
        help="Coverage: number of reviews per proposal",
        dest="COVERAGE",
        default=4,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true",
    )
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    main(args, loglevel)
