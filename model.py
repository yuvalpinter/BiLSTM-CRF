from __future__ import division
from collections import Counter
from _collections import defaultdict
from evaluate_morphotags import Evaluator
from sys import maxint

import collections
import argparse
import random
import cPickle
import logging
import progressbar
import os
import math
import dynet as dy
import numpy as np

import utils

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
POS_KEY = "POS"
PADDING_CHAR = "<*>"

DEFAULT_WORD_EMBEDDING_SIZE = 64

class LSTMTagger:

    def __init__(self, tagset_sizes, num_lstm_layers, hidden_dim, word_embeddings, no_we_update, use_char_rnn, charset_size, lowercase_words, att_props=None, vocab_size=None, word_embedding_dim=None):
        self.model = dy.Model()
        self.tagset_sizes = tagset_sizes
        self.attributes = tagset_sizes.keys()
        self.we_update = not no_we_update
        self.lowercase_words = lowercase_words
        if att_props is not None:
            self.att_props = defaultdict(float, {att:(1.0-p) for att,p in att_props.iteritems()})
        else:
            self.att_props = None

        if word_embeddings is not None: # Use pretrained embeddings
            vocab_size = word_embeddings.shape[0]
            word_embedding_dim = word_embeddings.shape[1]

        self.words_lookup = self.model.add_lookup_parameters((vocab_size, word_embedding_dim))

        if word_embeddings is not None:
            self.words_lookup.init_from_array(word_embeddings)

        # Char LSTM Parameters
        self.use_char_rnn = use_char_rnn
        if use_char_rnn:
            self.char_lookup = self.model.add_lookup_parameters((charset_size, 20))
            self.char_bi_lstm = dy.BiRNNBuilder(1, 20, hidden_dim, self.model, dy.LSTMBuilder)

        # Word LSTM parameters
        if use_char_rnn:
            input_dim = word_embedding_dim + hidden_dim
        else:
            input_dim = word_embedding_dim
        self.word_bi_lstm = dy.BiRNNBuilder(num_lstm_layers, input_dim, hidden_dim, self.model, dy.LSTMBuilder)

        # Matrix that maps from Bi-LSTM output to num tags
        self.lstm_to_tags_params = {}
        self.lstm_to_tags_bias = {}
        self.mlp_out = {}
        self.mlp_out_bias = {}
        for att, set_size in tagset_sizes.items():
            self.lstm_to_tags_params[att] = self.model.add_parameters((set_size, hidden_dim))
            self.lstm_to_tags_bias[att] = self.model.add_parameters(set_size)
            self.mlp_out[att] = self.model.add_parameters((set_size, set_size))
            self.mlp_out_bias[att] = self.model.add_parameters(set_size)


    def word_rep(self, word):
        '''
        :param word: index of word in lookup table
        '''
        if self.lowercase_words:
            lower_word_form = i2w[word].lower()
            if lower_word_form in w2i:
                word_in_ds = w2i[lower_word_form]
            else:
                word_in_ds = word
        else:
            word_in_ds = word
        wemb = dy.lookup(self.words_lookup, word_in_ds, update=self.we_update)
        if self.use_char_rnn:
            pad_char = c2i[PADDING_CHAR]
            # Note: use original casing ("word") for characters
            char_ids = [pad_char] + [c2i[c] for c in i2w[word]] + [pad_char]
            char_embs = [self.char_lookup[cid] for cid in char_ids]
            char_exprs = self.char_bi_lstm.transduce(char_embs)
            return dy.concatenate([ wemb, char_exprs[-1] ])
        else:
            return wemb


    def build_tagging_graph(self, sentence):
        dy.renew_cg()

        embeddings = [self.word_rep(w) for w in sentence]

        lstm_out = self.word_bi_lstm.transduce(embeddings)

        H = {}
        Hb = {}
        O = {}
        Ob = {}
        scores = {}
        for att in self.attributes:
            H[att] = dy.parameter(self.lstm_to_tags_params[att])
            Hb[att] = dy.parameter(self.lstm_to_tags_bias[att])
            O[att] = dy.parameter(self.mlp_out[att])
            Ob[att] = dy.parameter(self.mlp_out_bias[att])
            scores[att] = []
            for rep in lstm_out:
                score_t = O[att] * dy.tanh(H[att] * rep + Hb[att]) + Ob[att]
                scores[att].append(score_t)

        return scores


    def loss(self, sentence, tags_set):
        observations_set = self.build_tagging_graph(sentence)
        errors = {}
        for att, tags in tags_set.iteritems():
            err = []
            for obs, tag in zip(observations_set[att], tags):
                err_t = dy.pickneglogsoftmax(obs, tag)
                err.append(err_t)
            errors[att] = dy.esum(err)
        if self.att_props is not None:
            for att, err in errors.iteritems():
                prop_vec = dy.inputVector([self.att_props[att]] * err.dim()[0])
                err = dy.cmult(err, prop_vec)
        return errors


    def tag_sentence(self, sentence):
        observations_set = self.build_tagging_graph(sentence)
        tag_seqs = {}
        for att, observations in observations_set.iteritems():
            observations = [ dy.softmax(obs) for obs in observations ]
            probs = [ obs.npvalue() for obs in observations ]
            tag_seq = []
            for prob in probs:
                tag_t = np.argmax(prob)
                tag_seq.append(tag_t)
            tag_seqs[att] = tag_seq
        return tag_seqs


    def set_dropout(self, p):
        self.word_bi_lstm.set_dropout(p)


    def disable_dropout(self):
        self.word_bi_lstm.disable_dropout()

    def save(self, file_name):
        members_to_save = []
        members_to_save.append(self.words_lookup)
        if (self.use_char_rnn):
            members_to_save.append(self.char_lookup)
            members_to_save.append(self.char_bi_lstm)
        members_to_save.append(self.word_bi_lstm)
        members_to_save.extend(utils.sortvals(self.lstm_to_tags_params))
        members_to_save.extend(utils.sortvals(self.lstm_to_tags_bias))
        members_to_save.extend(utils.sortvals(self.mlp_out))
        members_to_save.extend(utils.sortvals(self.mlp_out_bias))
        self.model.save(file_name, members_to_save)

        with open(file_name + "-atts", 'w') as attdict:
            attdict.write("\t".join(sorted(self.attributes)))

    @property
    def model(self):
        return self.model

def get_att_prop(instances):
    logging.info("Calculating attribute proportions for proportional loss margin or proportional loss magnitude")
    total_tokens = 0
    att_counts = Counter()
    for instance in instances:
        total_tokens += len(instance.sentence)
        for att, tags in instance.tags.items():
            t2i = t2is[att]
            att_counts[att] += len([t for t in tags if t != t2i.get(NONE_TAG, -1)])
    return {att:(1.0 - (att_counts[att] / total_tokens)) for att in att_counts}

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds (if not supplied, will be random)")
parser.add_argument("--num-epochs", default=20, dest="num_epochs", type=int, help="Number of full passes through training set (default - 20)")
parser.add_argument("--num-lstm-layers", default=2, dest="lstm_layers", type=int, help="Number of LSTM layers (default - 2)")
parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="Size of LSTM hidden layers (default - 128)")
parser.add_argument("--training-sentence-size", default=maxint, dest="training_sentence_size", type=int, help="Instance count of training set (default - unlimited)")
parser.add_argument("--token-size", default=maxint, dest="token_size", type=int, help="Token count of training set (default - unlimited)")
parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate (default - 0.01)")
parser.add_argument("--dropout", default=-1, dest="dropout", type=float, help="Amount of dropout to apply to LSTM part of graph (default - off)")
parser.add_argument("--no-we-update", dest="no_we_update", action="store_true", help="Word Embeddings aren't updated")
parser.add_argument("--loss-prop", dest="loss_prop", action="store_true", help="Proportional loss magnitudes")
parser.add_argument("--use-char-rnn", dest="use_char_rnn", action="store_true", help="Use character RNN (default - off)")
parser.add_argument("--lowercase-words", dest="lowercase_words", action="store_true", help="Words are all in lowercased form (characters stay the same)")
parser.add_argument("--log-dir", default="log", dest="log_dir", help="Directory where to write logs / serialized models")
parser.add_argument("--no-model", dest="no_model", action="store_true", help="Don't serialize models")
parser.add_argument("--dynet-mem", help="Ignore this external argument")
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
options = parser.parse_args()


# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
if not os.path.exists(options.log_dir):
    os.mkdir(options.log_dir)
logging.basicConfig(filename=options.log_dir + "/log.txt", filemode="w", format="%(message)s", level=logging.INFO)
train_dev_cost = utils.CSVLogger(options.log_dir + "/train_dev.log", ["Train.cost", "Dev.cost"])


# ===-----------------------------------------------------------------------===
# Log run parameters
# ===-----------------------------------------------------------------------===

logging.info(
"""
Dataset: {}
Pretrained Embeddings: {}
Num Epochs: {}
LSTM: {} layers, {} hidden dim
Training set size limit: {} sentences or {} tokens
Initial Learning Rate: {}
Dropout: {}
LSTM loss weights proportional to attribute frequency: {}
Lowercasing words: {}

""".format(options.dataset, options.word_embeddings, options.num_epochs, options.lstm_layers, options.hidden_dim,
           options.training_sentence_size, options.token_size, options.learning_rate, options.dropout, options.loss_prop, options.lowercase_words))

if options.debug:
    print "DEBUG MODE"

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = cPickle.load(open(options.dataset, "r"))
w2i = dataset["w2i"]
t2is = dataset["t2is"]
c2i = dataset["c2i"]
m2i = None
i2w = { i: w for w, i in w2i.items() } # Inverse mapping
i2ts = { att: {i: t for t, i in t2i.items()} for att, t2i in t2is.items() }
i2c = { i: c for c, i in c2i.items() }

training_instances = dataset["training_instances"]
training_vocab = dataset["training_vocab"]
dev_instances = dataset["dev_instances"]
dev_vocab = dataset["dev_vocab"]
test_instances = dataset["test_instances"]

# trim training set for size evaluation (sentence based)
if len(training_instances) > options.training_sentence_size:
    random.shuffle(training_instances)
    training_instances = training_instances[:options.training_sentence_size]

# trim training set for size evaluation (token based)
training_corpus_size = sum(training_vocab.values())
if training_corpus_size > options.token_size:
    random.shuffle(training_instances)
    cumulative_tokens = 0
    cutoff_index = -1
    for i,inst in enumerate(training_instances):
        cumulative_tokens += len(inst.sentence)
        if cumulative_tokens >= options.token_size:
            training_instances = training_instances[:i+1]
            break

# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===
if options.word_embeddings is not None:
    word_embeddings = utils.read_pretrained_embeddings(options.word_embeddings, w2i)
else:
    word_embeddings = None

tag_set_sizes = { att: len(t2i) for att, t2i in t2is.items() }

if options.loss_prop:
    att_props = get_att_prop(training_instances)
else:
    att_props = None

model = LSTMTagger(tagset_sizes=tag_set_sizes,
                   num_lstm_layers=options.lstm_layers,
                   hidden_dim=options.hidden_dim,
                   word_embeddings=word_embeddings,
                   no_we_update = options.no_we_update,
                   use_char_rnn=options.use_char_rnn,
                   charset_size=len(c2i),
                   lowercase_words=options.lowercase_words,
                   vocab_size=len(w2i),
                   att_props=att_props,
                   word_embedding_dim=DEFAULT_WORD_EMBEDDING_SIZE)

trainer = dy.MomentumSGDTrainer(model.model, options.learning_rate, 0.9, 0.1)
logging.info("Training Algorithm: {}".format(type(trainer)))

logging.info("Number training instances: {}".format(len(training_instances)))
logging.info("Number dev instances: {}".format(len(dev_instances)))

for epoch in xrange(int(options.num_epochs)):
    bar = progressbar.ProgressBar()
    random.shuffle(training_instances)
    train_loss = 0.0
    train_correct = Counter()
    train_total = Counter()

    if options.dropout > 0:
        model.set_dropout(options.dropout)

    if options.debug:
        train_instances = training_instances[0:int(len(training_instances)/20)]
    else:
        train_instances = training_instances

    for idx,instance in enumerate(bar(train_instances)):
        if len(instance.sentence) == 0: continue

        gold_tags = instance.tags
        for att in model.attributes:
            if att not in instance.tags:
                gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
        loss_exprs = model.loss(instance.sentence, gold_tags)
        loss_expr = dy.esum(loss_exprs.values())
        loss = loss_expr.scalar_value()

        # Bail if loss is NaN
        if math.isnan(loss):
            assert False, "NaN occured"

        train_loss += (loss / len(instance.sentence))

        # Do backward pass and update parameters
        loss_expr.backward()
        trainer.update()

    logging.info("\n")
    logging.info("Epoch {} complete".format(epoch + 1))
    trainer.update_epoch(1)
    print trainer.status()

    train_loss = train_loss / len(train_instances)

    # Evaluate dev data
    model.disable_dropout()
    dev_loss = 0.0
    dev_correct = Counter()
    dev_total = Counter()
    dev_oov_total = Counter()
    bar = progressbar.ProgressBar()
    total_wrong = Counter()
    total_wrong_oov = Counter()
    f1_eval = Evaluator(m = 'att')
    if options.debug:
        d_instances = dev_instances[0:int(len(dev_instances)/10)]
    else:
        d_instances = dev_instances
    with open("{}/devout_epoch-{:02d}.txt".format(options.log_dir, epoch + 1), 'w') as dev_writer:
        for instance in bar(d_instances):
            if len(instance.sentence) == 0: continue
            gold_tags = instance.tags
            for att in model.attributes:
                if att not in instance.tags:
                    gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
            losses = model.loss(instance.sentence, gold_tags)
            total_loss = sum([l.scalar_value() for l in losses.values()])
            out_tags_set = model.tag_sentence(instance.sentence)

            gold_strings = utils.morphotag_strings(i2ts, gold_tags)
            obs_strings = utils.morphotag_strings(i2ts, out_tags_set)
            for g, o in zip(gold_strings, obs_strings):
                f1_eval.add_instance(utils.split_tagstring(g, has_pos=True), utils.split_tagstring(o, has_pos=True))
            for att, tags in gold_tags.items():
                out_tags = out_tags_set[att]
                correct_sent = True

                oov_strings = []
                for word, gold, out in zip(instance.sentence, tags, out_tags):
                    if gold == out:
                        dev_correct[att] += 1
                    else:
                        # Got the wrong tag
                        total_wrong[att] += 1
                        correct_sent = False
                        if i2w[word] not in training_vocab:
                            total_wrong_oov[att] += 1

                    if i2w[word] not in training_vocab:
                        dev_oov_total[att] += 1
                        oov_strings.append("OOV")
                    else:
                        oov_strings.append("")

                dev_total[att] += len(tags)

            dev_loss += (total_loss / len(instance.sentence))

            dev_writer.write(("\n"
                             + "\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
                                                                         gold_strings, obs_strings, oov_strings)])
                             + "\n").encode('utf8'))


    dev_loss = dev_loss / len(d_instances)

    # logging this epoch
    logging.info("POS Dev Accuracy: {}".format(dev_correct[POS_KEY] / dev_total[POS_KEY]))
    logging.info("POS % OOV accuracy: {}".format((dev_oov_total[POS_KEY] - total_wrong_oov[POS_KEY]) / dev_oov_total[POS_KEY]))
    if total_wrong[POS_KEY] > 0:
        logging.info("POS % Wrong that are OOV: {}".format(total_wrong_oov[POS_KEY] / total_wrong[POS_KEY]))
    for attr in t2is.keys():
        if attr != POS_KEY:
            logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
    logging.info("Total attribute F1s: {} micro, {} macro, POS included = {}".format(f1_eval.mic_f1(), f1_eval.mac_f1(), False))

    logging.info("Total dev tokens: {}, Total dev OOV: {}, % OOV: {}".format(dev_total[POS_KEY], dev_oov_total[POS_KEY], dev_oov_total[POS_KEY] / dev_total[POS_KEY]))

    logging.info("Train Loss: {}".format(train_loss))
    logging.info("Dev Loss: {}".format(dev_loss))
    train_dev_cost.add_column([train_loss, dev_loss])

    if epoch > 1 and epoch % 10 != 0: # leave outputs from epochs 1,10,20, etc.
        old_devout_file_name = "{}/devout_epoch-{:02d}.txt".format(options.log_dir, epoch)
        os.remove(old_devout_file_name)

    # Serialize model
    if not options.no_model:
        new_model_file_name = "{}/model_epoch-{:02d}.bin".format(options.log_dir, epoch + 1)
        logging.info("Saving model to {}".format(new_model_file_name))
        model.save(new_model_file_name) # TODO also save non-internal model stuff like mappings (?)
        if epoch > 1 and epoch % 10 != 0: # leave models from epochs 1,10,20, etc.
            logging.info("Removing files from previous epoch.")
            old_model_file_name = "{}/model_epoch-{:02d}.bin".format(options.log_dir, epoch)
            os.remove(old_model_file_name)
            os.remove(old_model_file_name + ".pym")
            os.remove(old_model_file_name + ".pyk")
            os.remove(old_model_file_name + "-atts")

# Evaluate test data (once)
logging.info("\n")
logging.info("Number test instances: {}".format(len(test_instances)))
model.disable_dropout()
test_correct = Counter()
test_total = Counter()
test_oov_total = Counter()
bar = progressbar.ProgressBar()
total_wrong = Counter()
total_wrong_oov = Counter()
f1_eval = Evaluator(m = 'att')
if options.debug:
    t_instances = test_instances[0:int(len(test_instances)/10)]
else:
    t_instances = test_instances
with open("{}/testout.txt".format(options.log_dir), 'w') as test_writer:
    for instance in bar(t_instances):
        if len(instance.sentence) == 0: continue
        gold_tags = instance.tags
        for att in model.attributes:
            if att not in instance.tags:
                gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
        out_tags_set = model.tag_sentence(instance.sentence)

        gold_strings = utils.morphotag_strings(i2ts, gold_tags)
        obs_strings = utils.morphotag_strings(i2ts, out_tags_set)
        for g, o in zip(gold_strings, obs_strings):
            f1_eval.add_instance(utils.split_tagstring(g, has_pos=True), utils.split_tagstring(o, has_pos=True))
        for att, tags in gold_tags.items():
            out_tags = out_tags_set[att]
            correct_sent = True

            oov_strings = []
            for word, gold, out in zip(instance.sentence, tags, out_tags):
                if gold == out:
                    test_correct[att] += 1
                else:
                    # Got the wrong tag
                    total_wrong[att] += 1
                    correct_sent = False
                    if i2w[word] not in training_vocab:
                        total_wrong_oov[att] += 1

                if i2w[word] not in training_vocab:
                    test_oov_total[att] += 1
                    oov_strings.append("OOV")
                else:
                    oov_strings.append("")

            test_total[att] += len(tags)
        test_writer.write(("\n"
                         + "\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
                                                                     gold_strings, obs_strings, oov_strings)])
                         + "\n").encode('utf8'))


logging.info("POS Test Accuracy: {}".format(test_correct[POS_KEY] / test_total[POS_KEY]))
logging.info("POS % Test OOV accuracy: {}".format((test_oov_total[POS_KEY] - total_wrong_oov[POS_KEY]) / test_oov_total[POS_KEY]))
if total_wrong[POS_KEY] > 0:
    logging.info("POS % Test Wrong that are OOV: {}".format(total_wrong_oov[POS_KEY] / total_wrong[POS_KEY]))
for attr in t2is.keys():
    if attr != POS_KEY:
        logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
logging.info("Total attribute F1s: {} micro, {} macro, POS included = {}".format(f1_eval.mic_f1(), f1_eval.mac_f1(), False))

logging.info("Total test tokens: {}, Total test OOV: {}, % OOV: {}".format(test_total[POS_KEY], test_oov_total[POS_KEY], test_oov_total[POS_KEY] / test_total[POS_KEY]))
