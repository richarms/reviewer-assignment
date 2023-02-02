#!/usr/bin/env python3
#

"""
# MeerKAT Proposal Reviewer Allocation
Richard Armstrong 
February 2023

Reviewer Assignment Problem following (1) for a set of MeerKAT Proposals with some
specific requirements:

* Reviewers must have between 10 and 20 proposals to review 
* Some reviewers have requested a maximum of 10 proposals
* Proposals are reviewed 4 times each
* Must exclude reviewers from reviewing proposals they are involved in

1. https://www.cis.upenn.edu/~cjtaylor/PUBLICATIONS/pdfs/TaylorTR08.pdf

"""

import sys, argparse, logging
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt

from numpy import genfromtxt

COVERAGE = 4  # Number of reviews per proposal
LOADS_LB = 10  # Minimum number of reviews per reviewer


def main(args, loglevel):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)

    # ---- LOAD
    # load the list of proposals and their categories into a dataframe
    df_cat = pd.read_csv("csv/proposal_scientific_categories_MKT-22.csv", sep=";")
    df_cat["Category_List"] = df_cat["CATEGORY LIST"].apply(lambda x: eval(x))
    n_p = n_proposals = len(df_cat)

    # load the reviewer conflict dataframe
    df_conflict = pd.read_csv("csv/science_cross_check_MKT-22.csv", sep=";")
    df_conflict["Proposals"] = df_conflict["PROPOSAL LIST"].apply(
        lambda x: x.strip("][").split(",")
    )
    df_conflict

    # load observation categories for proposals and reviewers
    roe = np.genfromtxt(
        "csv/reviewer_observation_expertise_MKT-22.csv", delimiter=",", skip_header=1
    )
    poc = np.genfromtxt(
        "csv/proposal_observation_categories_MKT-22.csv", delimiter=",", skip_header=1
    )

    # load the self-identified reviewer competency scores per catagory into a dataframe
    df_rev_score = pd.read_csv("csv/reviewer_scientific_expertise_MKT-22.csv")
    # df_rev_score.index = df_rev_score['REVIEWER ID']
    rev_score = genfromtxt(
        "csv/reviewer_scientific_expertise_MKT-22.csv", delimiter=","
    )

    # Create vector with max number of reviews per reviewer
    n_r = n_reviewers = len(df_rev_score)
    reviewers_props = 20 * np.ones(n_r)
    # limit10_list = [28,29]
    # reviewers_props[limit10_list] = 10
    df_rev_score[
        "N_max"
    ] = reviewers_props  # Make column with max number of reviews for each reviewer

    scientific_categories = pd.read_csv("csv/scientific_categories_MKT-22.csv")
    scientific_categories.index = scientific_categories["CATEGORY ID"]

    df_rev_score.columns = [
        "CATEGORY ID",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        "N_max",
    ]

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
    logging.info(f"conflict_counter: {conflict_counter}")
    # print(conflict_mask[n_r-3], n_r-3, conflict_mask.shape)

    # print(f'number of conflicts for each reviewer: \n{np.column_stack((og_rev_idx,n_p-np.sum(conflict_mask, axis=1))).astype(int)}')
    logging.info(f"sum of conflicts: {int(np.sum(n_p-np.sum(conflict_mask, axis=1)))}")

    # create a binary array (/mask) of reviewer expertise > .0
    zeroes_mask = np.where(np.array(df_rev_score.transpose()) > 0.0, 1, 0)[1:-1]
    np.count_nonzero(zeroes_mask), zeroes_mask.shape

    # create a numerical array of the reviewer scores dataframe
    rev_scores = np.array(df_rev_score.transpose())[1:-1]
    np.count_nonzero(rev_scores), rev_scores.shape[0], rev_scores.shape[1]

    # ------ PLOTS

    # Make a bar plot of all the proposal categories and numerical reviewer expertise
    all_topics_list = []
    all_topics_count = np.zeros(df_rev_score.transpose()[1:-1].shape[0])

    for idx, topic_set in enumerate(
        df_cat["Category_List"]
    ):  # run through each proposal
        # print(idx, topic_set)
        for topic in topic_set:  # run through each topic in each proposal
            all_topics_count[topic - 1] += 1  # add one to the demand per mention
            # print(topic, df_rev_score[topic][1])
            all_topics_list.append(topic)

    fig, ax = plt.subplots()  # figsize=(16,8))

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
    # ax.hist(np.array(all_topics_list), histtype='step', linewidth=3, align='mid', bins=1000, label = 'category demand')

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

    # set the affinity to the MAXimum of ANY keyword
    max = 0
    for row_idx, row in enumerate(affinity):
        # print(row_idx, row)
        # for cell_idx, cell in enumerate(row):
        for cell_idx, topic_set in enumerate(df_cat["Category_List"]):
            # print(topic_set, cell_idx)
            for topic in topic_set:
                # print(topic)
                max = np.maximum(max, rev_scores[topic - 1][row_idx])
                # if ((topic != 14) and (topic != 15)):
                # max = np.maximum(max, df_rev_score[topic][row_idx+1])
                # NOTE only consider the 13 subject categories
            affinity[row_idx, cell_idx] = max / 10  # normalise to between 0 and 1
            max = 0

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
    # print(prop_obs_cat)

    for row_idx, row in enumerate(rev_obs_exp):
        # print (rev_obs_exp[row_idx][1:])
        for col_idx, col in enumerate(prop_obs_cat):
            # print (row_idx, col_idx, col, row[1:])
            if row[int(col)] == 0:
                # print (affinity[row_idx][col_idx])
                affinity[row_idx][col_idx] = 0.0
    logging.debug(
        "Obs cat done",
        affinity.shape,
        "zeros:",
        affinity.flatten().shape[0] - np.count_nonzero(affinity),
        "of",
        affinity.flatten().shape[0],
    )

    # check there is at least one non-zero
    """set_zero = False
    for row_idx, row in enumerate(affinity):#row_idx is reviewer ID
        for cell_idx, topic_set in enumerate(df_cat['Category_List']):#cell_idx is the proposal id, topic_set is the list of topics
            for topic in topic_set:
                if df_rev_score[topic][row_idx]==0.: set_zero = True # if there are ANY '0' experience ratings
            if set_zero == True: affinity[row_idx, cell_idx] = 0.
            set_zero = False

    print('at least one nonzero done', affinity.shape, 'zeros:', affinity.flatten().shape[0] - np.count_nonzero(affinity), 'of', affinity.flatten().shape[0])
    """

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

    """
    # apply a non-linear scaling to the affinity 
    affinity = affinity + np.where(affinity>0.3, 0.4, 0.)
    print('non_linear scaling done', affinity.shape, 'zeros:', affinity.flatten().shape[0] - np.count_nonzero(affinity), 'of', affinity.flatten().shape[0])
    """

    # --------------------------------------------------------------------------------------------
    # convert arrays to a set of stored numpy arrays for the Linear Program (LP) optimisation code
    # --------------------------------------------------------------------------------------------

    # assign the maximum number of proposals per reviewer
    loads = np.genfromtxt("csv/science_max_reviews.csv", skip_header=1, delimiter=",")[
        :, 1
    ]

    # set the lower bound of proposals to review to be 10
    loads_lb = LOADS_LB * np.ones(affinity.shape[0])

    # set the coverage (i.e. number of reviews per paper) to be exactly 4
    covs = COVERAGE * np.ones(affinity.shape[1])

    # -------------
    # following https://www.cis.upenn.edu/~cjtaylor/PUBLICATIONS/pdfs/TaylorTR08.pdf

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
    # K = np.vstack((Np, Nr, I, -I))
    # I use a lower bound on number of reviews per reviewer, so add another term (-Nr) to
    # the totally unimodular matrix K
    K = np.vstack((Np, Nr, -Nr, I, -I))
    N = np.vstack((Np, Nr))

    cp = COVERAGE * np.ones(n_pap)
    cr = loads
    crlb = LOADS_LB * np.ones(n_rev)

    c = np.concatenate((cp, cr))
    # c = np.concatenate((cp, cr, crlb))

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
    # res = scipy.optimize.linprog(
    #    -a, A_ub=N, b_ub=c, bounds=(0, 1), options={"disp": True}, integrality=3)

    # plt.imshow(assignment)

    # The assignment matrix is a binary array of dim n_reviewers * n_proposals
    assignment = res.x.reshape(affinity.shape)
    assignment = assignment.astype(int)
    # assignment[np.where(og_rev_idx==40)].nonzero()

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
    gene_matrix = np.zeros((n_proposals, COVERAGE))
    gene_matrix_indexes = np.zeros((n_proposals, COVERAGE))

    # original reviewer index
    # og_rev_idx = np.array(df_rev_score['CATEGORY ID'])

    # The reviewer_assignment is the logical opposite of the gene matrix: for each reviewer,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign reviewers to proposals.",
        epilog="As an alternative to the commandline, params can be placed in a file, one per line, and specified on the commandline like '%(prog)s @params.conf'.",
        fromfile_prefix_chars="@",
    )
    # parser.add_argument("argument", help="pass ARG to the program", metavar="ARG")
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    main(args, loglevel)
