from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import sys
import time
from datetime import datetime
from operator import itemgetter

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn import metrics

from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.deep.model import DecagonModel
from decagon.deep.optimizer import DecagonOptimizer
from decagon.utility import rank_metrics, preprocessing
# NOTE utility.py needs to be copied up from polypharmacy to current directory
from utility import *

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(0)

###########################################################
#
# Global variables
#
###########################################################

FLAGS = None

placeholders = None
minibatch_iterator = None
sess = None
optimizer = None
adj_mats_orig = None
num_edge_types = 0


###########################################################
#
# Functions
#
###########################################################


def cross_entropy(y_hat, y):
    """
    :param y_hat: estimate/prediction of label (score)
    :param y: actual label
    """
    if y == 1:
        return -np.log(y_hat)
    else:
        return -np.log(1 - y_hat)


def get_accuracy_scores(edges_pos, edges_neg, edge_type, feed_dict):
    """
    Measure AUROC, AUPRC, and AP@50 for the given edge type using the provided list of positive and negative examples.
    :param edges_pos: Positive examples to measure accuracy over (need not be only edge_type type edges)
    :param edges_neg: Negative examples to measure accuracy over (need not be only edge_type type edges)
    :param edge_type: Edge type to filter by
    :param feed_dict: feed dictionary which should contain placeholders etc. See EdgeMiniBatchIterator#update_feed_dict
    :return: auroc, auprc, apk@50
    """
    actual, predicted, all_scores, all_labels, _, _, _ = get_predictions(edge_type, edges_neg, edges_pos, feed_dict)

    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    auroc = metrics.roc_auc_score(all_labels, all_scores)
    auprc = metrics.average_precision_score(all_labels, all_scores)
    apk_score = rank_metrics.apk(actual, predicted, k=50)

    return auroc, auprc, apk_score


def get_predictions(edge_type, edges_neg, edges_pos, feed_dict):
    """
    Get the predictions for the given edges.
    :param edge_type: Edge type to filter by. list/tuple with three elements
    :type edge_type: tuple
    :param edges_pos: Positive examples to measure accuracy over (need not be only edge_type type edges)
    :param edges_neg: Negative examples to measure accuracy over (need not be only edge_type type edges)
    :param feed_dict: feed dictionary which should contain placeholders etc. See EdgeMiniBatchIterator#update_feed_dict
    :return:
    """
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch_iterator.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(optimizer.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    actual = []
    predicted = []
    edge_index = 0

    subjects = []
    objects = []
    predicates = []

    scores = []

    # There are 4 possible edge categories:
    # (0,0) protein-protein edge
    # (0,1) protein-drug edge
    # (1,0) drug-protein edge
    # (1,1) drug-drug edge
    edge_category = edge_type[:2]
    for u, v in edges_pos[edge_category][edge_type[2]]:
        score = sigmoid(rec[u, v])
        scores.append(score)
        # Make sure this positive edge really did occur in the original data
        assert adj_mats_orig[edge_category][edge_type[2]][u, v] == 1, 'Problem 1'

        subjects.append(u)
        objects.append(v)
        predicates.append(edge_type)

        actual.append(edge_index)
        predicted.append((score, edge_index))
        edge_index += 1

    scores_neg = []
    for u, v in edges_neg[edge_category][edge_type[2]]:
        score = sigmoid(rec[u, v])
        scores_neg.append(score)
        # Make sure this negative edge really did not occur in the original data
        assert adj_mats_orig[edge_category][edge_type[2]][u, v] == 0, 'Problem 0'

        subjects.append(u)
        objects.append(v)
        predicates.append(edge_type)

        predicted.append((score, edge_index))
        edge_index += 1

    all_scores = np.hstack([scores, scores_neg])
    all_scores = np.nan_to_num(all_scores)
    all_labels = np.hstack([np.ones(len(scores)), np.zeros(len(scores_neg))])

    return actual, predicted, all_scores, all_labels, subjects, predicates, objects


def get_overall_auc(edges_pos, edges_neg, feed_dict):
    labels_all, preds_all = get_all_labels_and_predictions(edges_neg, edges_pos, feed_dict)

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    return roc_sc


def get_all_labels_and_predictions(edges_pos, edges_neg, feed_dict):
    all_labels = []
    all_scores = []
    for edge_type in range(num_edge_types):
        scores, labels = get_predicted_labels(
            edges_pos=edges_pos, edges_neg=edges_neg, edge_type=minibatch_iterator.idx2edge_type[edge_type],
            feed_dict=feed_dict)
        all_labels = np.hstack([all_labels, labels])
        all_scores = np.hstack([all_scores, scores])
    return all_labels, all_scores


def get_validation_loss(edges_pos, edges_neg, feed_dict):
    """ Mean cross entropy """
    all_labels, all_scores = get_all_labels_and_predictions(edges_pos=edges_pos, edges_neg=edges_neg,
                                                            feed_dict=feed_dict)
    loss_sum = 0.0
    for index in range(len(all_labels)):
        loss_sum = loss_sum + cross_entropy(all_scores[index], all_labels[index])
    # determine average cross entropy
    return loss_sum / len(all_labels)


def get_predicted_labels(edges_pos, edges_neg, edge_type, feed_dict):
    actual, predicted, all_scores, all_labels, _, _, _ = get_predictions(edge_type, edges_neg=edges_neg,
                                                                         edges_pos=edges_pos,
                                                                         feed_dict=feed_dict)
    return all_scores, all_labels


def construct_placeholders(edge_types):
    global placeholders
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i, j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})


###########################################################
#
# Load and preprocess data
#
###########################################################

####
#
# All preprocessed datasets used in the drug combination study are at: http://snap.stanford.edu/decagon:
# (1) Download datasets from http://snap.stanford.edu/decagon to your local machine.
# (2) Supply path to bio-decagon directory (with trailing slash!) to main.py as argument
# (3) Train & test the model.
####

# mappings:
# combo2stitch maps drug combinations to a tuple of stitch IDs,
#     e.g. 'CID000005206_CID000009433': ['CID000005206', 'CID000009433']
# combo2se maps combo (e.g. 'CID000005206_CID000009433') to list of side effects
# se2name maps side effect ids that occur in the poly/combo file to side effect name


def main(args):
    decagon_data_file_directory = args.decagon_data_file_directory
    verbose = args.verbose
    script_start_time = datetime.now()

    # create pre-processed file that only has side effect with >=500 occurrences
    if not os.path.isfile('%sbio-decagon-combo-over500only.csv' % decagon_data_file_directory):
        all_combos_arr = np.genfromtxt('%sbio-decagon-combo.csv' % decagon_data_file_directory, delimiter=',',
                                       encoding="utf8", dtype='str', skip_header=1)
        unique, counts = np.unique(all_combos_arr[:, 2], return_counts=True)
        counts_dict = dict(zip(unique, counts))
        has_500_or_more = []
        for i in all_combos_arr:
            has_500_or_more.append(counts_dict[i[2]] >= 500)
        over500 = all_combos_arr[np.where(has_500_or_more)]
        np.savetxt('%sbio-decagon-combo-over500only.csv' % decagon_data_file_directory, over500, delimiter=',',
                   fmt='%s',
                   encoding='utf8', header='STITCH 1,STITCH 2,Polypharmacy Side Effect,Side Effect Name', comments='')

    # use pre=processed file that only contains the most common side effects (those with >= 500 drug pairs)
    drug_drug_net, combo2stitch, combo2se, se2name = load_combo_se(
        fname=('%sbio-decagon-combo-over500only.csv' % decagon_data_file_directory))
    # net is a networkx graph with genes(proteins) as nodes and protein-protein-interactions as edges
    # node2idx maps node id to node index
    gene_net, node2idx = load_ppi(fname=('%sbio-decagon-ppi.csv' % decagon_data_file_directory))
    # stitch2proteins maps stitch ids (drug) to protein (gene) ids
    drug_gene_net, stitch2proteins = load_targets(
        fname=('%sbio-decagon-targets-all.csv' % decagon_data_file_directory))

    # this was 0.05 in the original code, but the paper says that 10% each are used for testing and validation
    val_test_size = 0.1
    n_genes = gene_net.number_of_nodes()
    if n_genes == 0:
        gene_adj = sp.coo_matrix([])
    else:
        gene_adj = nx.adjacency_matrix(gene_net)
    gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

    ordered_list_of_drugs = list(drug_drug_net.nodes.keys())
    ordered_list_of_side_effects = list(se2name.keys())
    ordered_list_of_proteins = list(gene_net.nodes.keys())

    n_drugs = len(ordered_list_of_drugs)

    drug_gene_adj = sp.lil_matrix(np.zeros((n_drugs, n_genes)))
    for drug in stitch2proteins:
        for protein in stitch2proteins[drug]:
            # there are quite a few drugs in here that aren't in our list of 645,
            # and proteins that aren't in our list of 19081
            if drug in ordered_list_of_drugs and protein in ordered_list_of_proteins:
                drug_index = ordered_list_of_drugs.index(drug)
                gene_index = ordered_list_of_proteins.index(protein)
                drug_gene_adj[drug_index, gene_index] = 1

    drug_gene_adj = drug_gene_adj.tocsr()

    # needs to be drug vs. gene matrix (645x19081)
    gene_drug_adj = drug_gene_adj.transpose(copy=True)

    drug_drug_adj_list = []
    if not os.path.isfile("%sadjacency_matrices/sparse_matrix0000.npz" % args.saved_files_directory):
        # pre-initialize all the matrices
        print("Initializing drug-drug adjacency matrix list")
        start_time = datetime.now()
        print("Starting at %s" % str(start_time))

        n = len(ordered_list_of_side_effects)
        for i in range(n):
            drug_drug_adj_list.append(sp.lil_matrix(np.zeros((n_drugs, n_drugs))))
            if verbose:
                print("%s percent done" % str(100.0 * i / n))
        print("Done initializing at %s after %s" % (datetime.now(), datetime.now() - start_time))

        start_time = datetime.now()
        combo_finish_time = start_time
        print("Creating adjacency matrices for side effects")
        print("Starting at %s" % str(start_time))
        combo_count = len(combo2se)
        combo_counter = 0

        # for side_effect_type in ordered_list_of_side_effects:
        # for drug1, drug2 in combinations(list(range(n_drugs)), 2):

        for combo in combo2se.keys():
            side_effect_list = combo2se[combo]
            for present_side_effect in side_effect_list:
                # find the matrix we need to update
                side_effect_number = ordered_list_of_side_effects.index(present_side_effect)
                # find the drugs for which we need to make the update
                drug_tuple = combo2stitch[combo]
                drug1_index = ordered_list_of_drugs.index(drug_tuple[0])
                drug2_index = ordered_list_of_drugs.index(drug_tuple[1])
                # update
                drug_drug_adj_list[side_effect_number][drug1_index, drug2_index] = 1

            if verbose and combo_counter % 1000 == 0:
                print("Finished combo %s after %s . %d percent of combos done" %
                      (combo_counter, str(combo_finish_time - start_time), (100.0 * combo_counter / combo_count)))
            combo_finish_time = datetime.now()
            combo_counter = combo_counter + 1

        print("Done creating adjacency matrices at %s after %s" % (datetime.now(), datetime.now() - start_time))

        start_time = datetime.now()
        print("Saving matrices to file")
        print("Starting at %s" % str(start_time))

        # save matrices to file
        if not os.path.isdir("%sadjacency_matrices" % args.saved_files_directory):
            os.mkdir("%sadjacency_matrices" % args.saved_files_directory)
        for i in range(len(drug_drug_adj_list)):
            sp.save_npz('%sadjacency_matrices/sparse_matrix%04d.npz' % (args.saved_files_directory, i,),
                        drug_drug_adj_list[i].tocoo())
        print("Done saving matrices to file at %s after %s" % (datetime.now(), datetime.now() - start_time))
    else:
        print("Loading adjacency matrices from file.")
        for i in range(len(ordered_list_of_side_effects)):
            drug_drug_adj_list.append(
                sp.load_npz('%sadjacency_matrices/sparse_matrix%04d.npz' % (args.saved_files_directory, i, )))

    for i in range(len(drug_drug_adj_list)):
        drug_drug_adj_list[i] = drug_drug_adj_list[i].tocsr()

    start_time = datetime.now()
    print("Setting up for training")
    print("Starting at %s" % str(start_time))

    drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]

    # data representation
    global adj_mats_orig
    adj_mats_orig = {
        (0, 0): [gene_adj, gene_adj.transpose(copy=True)],  # protein-protein interactions (and inverses)
        (0, 1): [gene_drug_adj],  # protein-drug relationships (inverse of targets)
        (1, 0): [drug_gene_adj],  # drug-protein relationships (targets)
        # This creates an "inverse" relationship for every polypharmacy side effect, using the transpose of the
        # relationship's adjacency matrix, resulting in 2x the number of side effects (and adjacency matrices).
        (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
    }
    degrees = {
        0: [gene_degrees, gene_degrees],
        1: drug_degrees_list + drug_degrees_list,
    }

    # featureless (genes)
    gene_feat = sp.identity(n_genes)
    gene_nonzero_feat, gene_num_feat = gene_feat.shape
    gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

    # features (drugs)
    drug_feat = sp.identity(n_drugs)
    drug_nonzero_feat, drug_num_feat = drug_feat.shape
    drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

    # data representation
    num_feat = {
        0: gene_num_feat,
        1: drug_num_feat,
    }
    nonzero_feat = {
        0: gene_nonzero_feat,
        1: drug_nonzero_feat,
    }
    feat = {
        0: gene_feat,
        1: drug_feat,
    }

    edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
    edge_type2decoder = {
        (0, 0): 'bilinear',
        (0, 1): 'bilinear',
        (1, 0): 'bilinear',
        (1, 1): 'dedicom',
    }

    edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
    global num_edge_types
    num_edge_types = sum(edge_types.values())
    print("Edge types:", "%d" % num_edge_types)

    ###########################################################
    #
    # Settings and placeholders
    #
    ###########################################################

    # Important -- Do not evaluate/print validation performance every iteration as it can take
    # substantial amount of time
    PRINT_PROGRESS_EVERY = 10000

    print("Defining placeholders")
    construct_placeholders(edge_types)

    ###########################################################
    #
    # Create minibatch iterator, model and optimizer
    #
    ###########################################################

    global minibatch_iterator
    iterator_pickle_file_name = args.saved_files_directory + "minibatch_iterator.pickle"
    if os.path.isfile(iterator_pickle_file_name):
        print("Load minibatch iterator pickle")
        with open(iterator_pickle_file_name, 'rb') as pickle_file:
            minibatch_iterator = pickle.load(pickle_file)
    else:
        print("Create minibatch iterator")
        minibatch_iterator = EdgeMinibatchIterator(
            adj_mats=adj_mats_orig,
            feat=feat,
            edge_types=edge_types,
            batch_size=FLAGS.batch_size,
            val_test_size=val_test_size,
            negatives_sampling_strategy=args.negatives_sampling_strategy,
            saved_files_directory=args.saved_files_directory)
        print("Pickling minibatch iterator")
        with open(iterator_pickle_file_name, 'wb') as pickle_file:
            pickle.dump(minibatch_iterator, pickle_file)

    print("Create model")
    model = DecagonModel(
        placeholders=placeholders,
        num_feat=num_feat,
        nonzero_feat=nonzero_feat,
        edge_types=edge_types,
        decoders=edge_type2decoder,
    )

    print("Create optimizer")
    global optimizer
    with tf.name_scope('optimizer'):
        optimizer = DecagonOptimizer(
            embeddings=model.embeddings,
            latent_inters=model.latent_inters,
            latent_varies=model.latent_varies,
            degrees=degrees,
            edge_types=edge_types,
            edge_type2dim=edge_type2dim,
            placeholders=placeholders,
            batch_size=FLAGS.batch_size,
            margin=FLAGS.max_margin
        )

    print("Done setting up at %s after %s" % (datetime.now(), datetime.now() - start_time))

    print("Initialize session")
    global sess
    sess = tf.Session()

    decagon_model_file_name = args.saved_files_directory + "decagon_model.ckpt"
    saved_model_available = os.path.isfile(decagon_model_file_name + ".index")
    if saved_model_available:
        saver = tf.train.Saver()
        saver.restore(sess, decagon_model_file_name)
        print("Model restored.")
    if not saved_model_available:
        print("Training model")
        start_time = datetime.now()
        print("Starting at %s" % str(start_time))

        sess.run(tf.global_variables_initializer())
        feed_dict = {}

        ###########################################################
        #
        # Train model
        #
        ###########################################################

        saver = tf.train.Saver()

        print("Train model")
        epoch_losses = []
        for epoch in range(FLAGS.epochs):

            minibatch_iterator.shuffle()
            itr = 0
            while not minibatch_iterator.end():
                # Construct feed dictionary
                feed_dict = minibatch_iterator.next_minibatch_feed_dict(placeholders=placeholders)
                feed_dict = minibatch_iterator.update_feed_dict(
                    feed_dict=feed_dict,
                    dropout=FLAGS.dropout,
                    placeholders=placeholders)

                t = time.time()

                # Training step: run single weight update
                outs = sess.run([optimizer.opt_op, optimizer.cost, optimizer.batch_edge_type_idx], feed_dict=feed_dict)
                train_cost = outs[1]
                batch_edge_type = outs[2]

                if itr % PRINT_PROGRESS_EVERY == 0:
                    val_auc, val_auprc, val_apk = get_accuracy_scores(
                        minibatch_iterator.val_edges, minibatch_iterator.val_edges_false,
                        minibatch_iterator.idx2edge_type[minibatch_iterator.current_edge_type_idx],
                        feed_dict)

                    print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:",
                          "%04d" % batch_edge_type,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                          "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))

                itr += 1
            validation_loss = get_validation_loss(edges_pos=minibatch_iterator.val_edges,
                                                  edges_neg=minibatch_iterator.val_edges_false, feed_dict=feed_dict)
            print("Epoch:", "%04d" % (epoch + 1), "Validation loss (average cross entropy): {}".format(validation_loss))

            epoch_losses.append(validation_loss)
            if len(epoch_losses) >= 3:
                if round(epoch_losses[-1], 3) >= round(epoch_losses[-2], 3) >= round(epoch_losses[-3], 3):
                    break

            print("Saving model after epoch:", epoch)
            save_path = saver.save(sess, args.saved_files_directory + "decagon_model" + str(epoch) + ".ckpt")
            print("Model saved in path: %s" % save_path)

        print("Optimization finished!")
        print("Done training model %s after %s" % (datetime.now(), datetime.now() - start_time))

        print("Saving model")
        save_path = saver.save(sess, decagon_model_file_name)
        print("Model saved in path: %s" % save_path)

        print("Pickling minibatch iterator")
        with open(iterator_pickle_file_name, 'wb') as pickle_file:
            pickle.dump(minibatch_iterator, pickle_file)

    start_time = datetime.now()
    print("Evaluating model")
    print("Starting at %s" % str(start_time))

    side_effect_result_lines = []
    header = "subject\tpredicate\tobject\tpredicted\tactual"
    side_effect_result_lines.append(header)
    print(header)

    for edge_type in range(num_edge_types):
        # get all edges in test set with this type
        feed_dict = minibatch_iterator.test_feed_dict(edge_type, placeholders=placeholders)
        feed_dict = minibatch_iterator.update_feed_dict(feed_dict, FLAGS.dropout, placeholders)
        edge_tuple = minibatch_iterator.idx2edge_type[edge_type]

        if args.predict_side_effects_only and edge_tuple[:2] != (1, 1):
            continue

        _, _, all_scores, all_labels, subjects, predicates, objects = get_predictions(
            edges_pos=minibatch_iterator.test_edges, edges_neg=minibatch_iterator.test_edges_false,
            edge_type=edge_tuple, feed_dict=feed_dict)

        header = "subject\tpredicate\tobject\tpredicted\tactual"
        side_effect_result_lines.append(header)
        print(header)

        for i in range(len(all_scores)):
            subject = subjects[i]
            if edge_tuple[0] == 1:
                subject = ordered_list_of_drugs[subject]
            else:
                subject = ordered_list_of_proteins[subject]

            object = objects[i]
            if edge_tuple[1] == 1:
                object = ordered_list_of_drugs[object]
            else:
                object = ordered_list_of_proteins[object]

            predicate = predicates[i]
            if edge_tuple[:2] == (1, 1):
                side_effect_index = edge_tuple[2]
                is_inverse = False
                if side_effect_index >= 963:
                    side_effect_index = side_effect_index - 963
                    is_inverse = True
                predicate = ordered_list_of_side_effects[side_effect_index]
                if is_inverse:
                    predicate = predicate + "_2"

            line = "{}\t{}\t{}\t{}\t{}".format(subject, predicate, object, all_scores[i], all_labels[i])
            if edge_tuple[:2] == (1, 1):
                side_effect_result_lines.append(line)
            print(line)

    with open(args.score_output_file, "w") as outfile:
        outfile.write("\n".join(side_effect_result_lines))

    print()

    print("Done evaluating at %s after %s" % (datetime.now(), datetime.now() - start_time))

    print("Script running time: %s" % (datetime.now() - script_start_time))


if __name__ == '__main__':
    # allow specification of command line flags
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("Tensorflow arguments")
    group.add_argument('--neg_sample_size', type=int, default=1, help='Negative sample size.')
    group.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    group.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    group.add_argument('--hidden1', type=int, default=64, help='Number of units in hidden layer 1.')
    group.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
    group.add_argument('--weight_decay', type=float, default=0, help='Weight for L2 loss on embedding matrix.')
    group.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    group.add_argument('--max_margin', type=float, default=0.1, help='Max margin parameter in hinge loss')
    group.add_argument('--batch_size', type=int, default=512, help='minibatch size.')
    group.add_argument('--bias', type=bool, default=True, help='Bias term.')

    tf_flags, unparsed = parser.parse_known_args()

    # put command line flags into tf flags object
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('neg_sample_size', tf_flags.neg_sample_size, 'Negative sample size.')
    flags.DEFINE_float('learning_rate', tf_flags.learning_rate, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', tf_flags.epochs, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', tf_flags.hidden1, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', tf_flags.hidden2, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('weight_decay', tf_flags.weight_decay, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', tf_flags.dropout, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('max_margin', tf_flags.max_margin, 'Max margin parameter in hinge loss')
    flags.DEFINE_integer('batch_size', tf_flags.batch_size, 'minibatch size.')
    flags.DEFINE_boolean('bias', tf_flags.bias, 'Bias term.')

    # Remove flags to keep from confusing tensorflow. This prevents an UnrecognizedFlagError later on.
    sys.argv = sys.argv[:1]

    parser = argparse.ArgumentParser(parents=[parser])
    group = parser.add_argument_group("More arguments")
    group.add_argument("--decagon_data_file_directory", type=str,
                       help="path to directory where bio-decagon-*.csv files are located, with trailing slash. "
                            "Default is current directory",
                       default='./')
    group.add_argument("--saved_files_directory", type=str,
                       help="path to directory where saved files files are located, with trailing slash. "
                            "Default is current directory. If a decagon_model.ckpt* exists in this directory, it will "
                            "be loaded and evaluated, and no training will be done.",
                       default='./')
    group.add_argument("--verbose", help="increase output verbosity", action="store_true", default=False)
    group.add_argument("--predict_side_effects_only",
                       help="output predictions for side effect triples only (not for PPI or drug target triples)",
                       action="store_true", default=False)
    group.add_argument("--negatives_sampling_strategy",
                       help="'naive' or 'known_pairs' (default naive). False edges for drug pairs will be "
                            "sampled as follows: "
                            "- naive: for each side effect in the positive testing examples, make a negative triple with that side "
                            "effect and 2 random drugs (independently sampled), confirming that the resulting triple is not "
                            "a positive triple. This strategy has a high chance of resulting in never-seen-before drug "
                            "pairings. "
                            "- known_pairs: for each side effect in the positive testing examples, make a negative triple with that "
                            "side effect and a randomly selected pair taken from the list of pairs that are present in the "
                            "positive examples. This way, only known drug pairings will be represented in the negative set.",
                       default="naive", type=str, choices=['naive', 'known_pairs'])
    group.add_argument("--score_output_file",
                       help="name of file to write test set predictions to", default="scores.tsv", type=str)
    args = parser.parse_args(unparsed)

    main(args)
