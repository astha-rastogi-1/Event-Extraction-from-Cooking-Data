# mongodb+srv://df-1203:<password>@event-extraction.m5zil.mongodb.net/test
import pymongo
import spacy
from spacy.lang.en.examples import sentences 
from spacy.lang.en import English
import random
from tqdm import tqdm
from collections import defaultdict
from nltk.corpus import wordnet as wn
from config import *
from datetime import datetime
import json
import collections
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances_argmin_min
import hdbscan
from tabulate import tabulate
import itertools
import copy
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.manifold import TSNE
# import seaborn as sns
import re

#TODO remember to change to the correct client
# client = pymongo.MongoClient("")  # 27017/semantic_reading
# # Apparently if you remove the commented section then the authentication works
# db =  client["test"]
# db_c = db["test_c"]
nlp = spacy.load("en_core_web_sm")
# dirname = os.getcwd()

class LostAndFound(Exception):
    pass

class ClusteringActions:
    def get_collection(self):
        mon_db = get_mongodb_database()
        return mon_db[self.coll]
    
    def __init__(self, collection, *, file_ext, al_round=0, n_cluster_span=(2, 50)):
        # self.device_name = device_name
        self.file_ext = file_ext
        self.al_round = str(al_round)
        self.coll = collection
        self.n_cluster_span = n_cluster_span
        self.sentences = [] 
        self.sentences_test = []
        self.test_mapping = defaultdict(list)
        self.embeddings_np = None
        self.sentence_scoring = dict()
        self.cluster_model_b_clusters = None
        self.info = {}
        # self.test = test
        
    def process_train_sentences(self):
        self.create_sentences()
        # print(self.sentences)
        processing_list = [self.assign_hypernyms, self.remove_duplicate_sentences,
                           self.small_caps_trail_spaces]
        for pro in tqdm(processing_list):
            sentences = list()
            for recipe_id, sentence in pro():
                sentences.append(sentence)
            self.sentences = sentences
            # print(self.sentences)
            print(len(self.sentences))
        #     # print(sentences)
        self.preprocess()
        print(self.sentences)
        self.save_sentences()


    def create_sentences(self):
            # For sentence splitting
            # https://github.com/explosion/spaCy/issues/2708#issuecomment-425912970
            nlp = English()
            nlp.add_pipe('sentencizer')
            len_ = self.get_collection().find().count()
            # Have an 85/15 train/test
            # int(len_ * 0.15)
            test = random.sample(range(0, len_), int(len_ * 0.15))
            self.info["test_ids"] = list()
            # print(test)
            for idx, recipe in tqdm(enumerate(self.get_collection().find()), total=len_):
                # print(recipe)
                if idx in test:
                    print(idx)
                    self.info["test_ids"].append(recipe["_id"])
                    continue
                doc = nlp(recipe['Method'])
                # print(type(doc))
                for sent in doc.sents:
                    if sent.text and len(sent.text) > 2:  # ToDo I changed that from 10
                        self.sentences.append(sent.text)
            # print(self.info)
            self.save_info()
    
    def preprocess(self, test=False):
        # recipe = self.get_collection().find()
        # doc = nlp(recipe['Method'])
        # print(doc)
        # # doc = nlp()
        # sentences = self.sentences
        # for i in range(len(sentences)):
        #     trial_sent = sentences[i]
        #     trial_sent = re.sub(r"step\s\d","", trial_sent)
        #     trial_sent = re.sub(r"advertisement","", trial_sent)
        #     trial_sent = re.sub(r"^\s+","", trial_sent)
        #     trial_sent = re.sub(r"\s+,",",", trial_sent)
        #     trial_sent = re.sub(r"\s+"," ", trial_sent)
        #     sentences[i] = trial_sent
        # self.sentences = sentences
        self.remove_step_advt(test)
        self.sentence_split(test)
        self.reduce_sent(test)


    def remove_step_advt(self, test=False):
        ## TRAINING
        # sentences = self.sentences
        # for i in range(len(sentences)):
        #         trial_sent = sentences[i]
        #         trial_sent = re.sub(r"step\s\d","", trial_sent)
        #         trial_sent = re.sub(r"advertisement","", trial_sent)
        #         trial_sent = re.sub(r"^\s+","", trial_sent)
        #         trial_sent = re.sub(r"\s+,",",", trial_sent)
        #         trial_sent = re.sub(r"\s+"," ", trial_sent)
        #         sentences[i] = trial_sent
        # self.sentences = sentences        
        
        ## TESTING
        if test:
            for recipe_id in self.test_mapping.keys():
                sentences = []
                for sent in self.test_mapping[recipe_id]:
                    trial_sent = sent
                    trial_sent = re.sub(r"step\s\d","", trial_sent)
                    trial_sent = re.sub(r"advertisement","", trial_sent)
                    trial_sent = re.sub(r"elise bauer ", "",trial_sent)
                    trial_sent = re.sub(r"sally vargas ","", trial_sent)
                    trial_sent = re.sub(r"^\s+","", trial_sent)
                    trial_sent = re.sub(r"\s+,",",", trial_sent)
                    trial_sent = re.sub(r"\s+"," ", trial_sent)
                    sent = trial_sent
                    sentences.append(sent)
                self.test_mapping[recipe_id] = sentences

    
    def sentence_split(self, test=False):
        # # TRAINING
        # # self.read_sentences()
        # sentences = self.sentences
        # f_sent = []
        # for sent in sentences:
        #     spit_list = []
        #     split_list = sent.split(';')
            
        #     if(len(split_list)>1):
        #         # print(split_list)
        #         for i in range(len(split_list)):
        #             split_list[i] = re.sub(r'^\s+','', split_list[i])
        #             split_list[i] = re.sub(r'\s+$','.',split_list[i])
        #             f_sent.append(split_list[i])
        #     else:
        #         f_sent.append(sent)
        # self.sentences = f_sent
        # # self.sentences = f_sent
        # # self.save_sentences(ext='split')

        # TESTING
        # sentences = self.test_iterator() if test else self.train_iterator()
        if test:
            
            for recipe_id in self.test_mapping.keys():
                f_sent = []
                for sent in self.test_mapping[recipe_id]:
                    split_list = []
                    split_list = sent.split(';')

                    if(len(split_list)>1):
                        for i in range(len(split_list)):
                            split_list[i] = re.sub(r'^\s+','', split_list[i])
                            split_list[i] = re.sub(r'\s+$','.',split_list[i])
                            f_sent.append(split_list[i])
                    else:
                        f_sent.append(sent)
                self.test_mapping[recipe_id] = f_sent

    
    def reduce_sent(self, test=False):
        should_not_contain = ['spoon','repeat','stand','pour','dip','sift','mix','beat','stir','knead','sprinkle','whisk','cut','ahead','set','add','fold','refrigerate',
                    'roll','pat','rub','freeze','stiff','freez','chop','place','cool','split','half','turn','dough','toss','combine','scoop','spread','pinch','cover','divide','freezer','serve','season','fill','transfer','top','over','break']

        should_contain = ['bake','skillet','baking','pre-heat','preheat','oven','overproof','ovenproof','tin','heat','boil','blanch','parboil','scald','gas','stove',
                    'saucepan','steep','water','microwave','pressure-cook','pressure cook','instant-pot','instant pot','simmer','poach','stew','slow cooker',
                    'cook','temperature','pot','broil','roast','burn','charcoal','sear','grill','brown','roast','rotisserie','saute','sauté','fry',
                    'oil','fat','melt','rise','braise','fried','bbq','barbeque','barbecue','simmer','poach','stew', 'sous vide','sous-vide','steam','toast']
        # # TRAINING
        # f_list = []
        # sentences = self.sentences
        # words_s = re.compile("|".join(should_contain))
        # words_sn = re.compile("|".join(should_not_contain))
        # for sent in sentences:
        #     if words_sn.search(sent):
        #         if words_s.search(sent):
        #             f_list.append(sent)
        #         else:
        #             print(sent)
        #         continue
        #     else:
        #         f_list.append(sent)
        # # print('\n\n')
        # # print(f_list) 
        # self.sentences = f_list

        # TESTING
        if test:
            
            words_s = re.compile("|".join(should_contain))
            words_sn = re.compile("|".join(should_not_contain))
            for recipe_id in self.test_mapping.keys():
                f_list = []
                for sent in self.test_mapping[recipe_id]:
                    if words_sn.search(sent):
                        if words_s.search(sent):
                            f_list.append(sent)
                        else:
                            print(sent)
                        continue
                    else:
                        f_list.append(sent)
                self.test_mapping[recipe_id] = f_list

    def save_info(self):
            now = datetime.now()
            fp = os.path.join(dirname, "data", "interim", "AL",
                            now.strftime("%Y_%m_%d") + "_final.json")
            print(fp)
            with open(fp, 'w') as f:
                json.dump(self.info, f, indent=4, default = str)
    
    def save_sentences(self, ext=None):
        """ Save to file the sentences"""

        now = datetime.now()
        if not ext:
            ext = now.strftime("%Y_%m_%d_%H")
        fp = os.path.join(dirname, "data", "interim", "AL",
                          "sentences_" + self.coll + ext + "_final.json")
        with open(fp, 'w') as f:
            json.dump(self.sentences, f, indent=4)    

    def train_iterator(self):
            for sentence in self.sentences:
                yield None, sentence

    @staticmethod
    def recursive_tag(ss, tag="food"):
        if ss.hypernyms():
            for hyp in ss.hypernyms():
                for lemma in hyp.lemmas():
                    if lemma.name() == tag:
                        raise LostAndFound
            ClusteringActions.recursive_tag(hyp)
        else:
            return False

    ## TO CONVERT ALL INGREDIENTS TO 'FOOD'
    def assign_hypernyms(self, test=False):
        """ Replace any mention of a target word with its parent name.
            Uses wordNet.

        May need nltk.download('wordnet')

        Returns
        -------
        """
        global nlp
        exclusions_food = {"rack", "c", "center", "centre", "broiler", "roast", "medium","grill","barbecue","bbq"}
        d_foods = defaultdict(lambda :0)
        print(d_foods)
        sentences = self.test_iterator() if test else self.train_iterator()
        for recipe_id, sentence in sentences:
            doc = nlp(sentence)
            for id, token in enumerate(doc):
                if token.is_stop or token.is_digit or token.is_bracket or token.is_punct or token.pos_ == "NUM":
                    continue
                if token.pos_ == "NOUN":
                    if token.lemma_ in exclusions_food:
                        continue
                    for ss in wn.synsets(token.lemma_):
                        try:
                            ClusteringActions.recursive_tag(ss, tag="food")
                        except LostAndFound:
                            d_foods[token.lemma_] += 1
                            sentence = sentence.replace(token.text, "food")
            yield recipe_id, sentence

    def remove_duplicate_sentences(self, test=False):
            sentences_ = set()
            sentences = self.test_iterator() if test else self.train_iterator()
            for _, sentence in sentences:
                if sentence not in sentences_:
                    sentences_.add(sentence)
                    yield _, sentence

    def small_caps_trail_spaces(self, test=False):
            sentences = self.test_iterator() if test else self.train_iterator()
            for recipe_id, sentence in sentences:
                sentence = sentence.rstrip()
                sentence = sentence.lstrip()
                sentence = sentence.lower()
                yield recipe_id, sentence


    def sentence_length(self, cutoff_freq=5):
        self.read_sentences()
        sent_len = defaultdict(lambda :0)
        for sentence in self.sentences:
            doc = nlp(sentence)
            sent_len[len(doc)] += 1

        od = collections.OrderedDict(sorted(sent_len.items()))
        pprint(od)
        df = pd.DataFrame(od, index=[0])
        import matplotlib.pyplot as plt
        ax = df.plot.bar()
        plt.show()
        plt.savefig(os.path.join(dirname, "data", "interim", "AL", "sentence_length_barplot.png"), dpi=100)
        sentences = list()
        for sentence in self.sentences:
            doc = nlp(sentence)
            if sent_len[len(doc)] < cutoff_freq:
                continue
            sentences.append(sentence)
        # Print the new sentence length
        sent_len = defaultdict(lambda :0)
        for sentence in sentences:
            doc = nlp(sentence)
            sent_len[len(doc)] += 1
        od = collections.OrderedDict(sorted(sent_len.items()))
        df = pd.DataFrame(od, index=[0])
        ax = df.plot.bar()
        plt.show()

        self.sentences = sentences
        self.save_sentences(ext="2021_09_07_cut_1")

    def read_sentences(self):
        # Read from file
        fp = None
        fp = os.path.join(dirname, "data", "interim", "AL",
                              "sentences_" + self.coll + "2021_09_07_18.json")
        with open(fp, 'r', encoding="utf-8") as f:
            self.sentences = json.load(f)
    
    def read_embeddings(self):
        fp = os.path.join(dirname, "data", "interim", "AL",self.file_ext + "_arrays",
                          f"corpus_embeddings__150_round_{self.al_round}_final.npy")  # ToDo remember to change
        self.embeddings_np = np.load(fp)

    def create_embeddings(self, sentences):
        fp = os.path.join(dirname, "models", "training_sem_read-roberta-large-nli-stsb-mean-tokens-round_6")
        embedder = SentenceTransformer(fp)

       
        print('Create embeddings')

        corpus_embeddings = embedder.encode(sentences, show_progress_bar=True)
        print('Saving embeddings')
        fp = os.path.join(dirname, "data", "interim", "AL", self.file_ext + "_arrays",
                          "corpus_embeddings_" + "_2021_09_07_round_0.npy")
        np.save(fp, np.array(corpus_embeddings))
        return np.array(corpus_embeddings)

    def process(self):
        self.read_sentences()
        self.read_embeddings()
        self.sentences = pd.Series(self.sentences) 
        print('Distance')
        dist = cosine_distances(self.embeddings_np)
        print('Clustering')
        cluster_model_b = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=4, min_samples=2,
                                          cluster_selection_epsilon=0.01, leaf_size=100)
        cluster_model_b = cluster_model_b.fit(dist.astype(np.double))

        with open(f"E:\Astha PC\Documents\BITS\Thesis\IML_project_export\project_export\semantic_reading\data\interim\AL\clusters_final_{self.al_round}.csv", "w", encoding="utf-8") as f:
            for label in tqdm(range(cluster_model_b.labels_.max() + 1)):
                for idx, sentence in enumerate(self.sentences):
                    if idx in np.where(cluster_model_b.labels_ == label)[0]:
                        to_print = str(label) + "\t" + sentence + "\n"
                        f.write(to_print)
        print('Metrics')
        print(metrics.silhouette_score(dist, cluster_model_b.labels_, metric="precomputed"))
        unique, counts = np.unique(cluster_model_b.labels_, return_counts=True)


        print(tabulate(np.asarray((unique, counts)).T, headers=["Cluster", "Frequency"], tablefmt="github"))
        self.active_learning_HDBScan(self.embeddings_np, cluster_model_b,
                                     threshold_amongst_clusters=24, quota=80, quota_incons=10, max_sample=1_000)
        
    def save_sentence_scoring(self):
        """ Save to sentence scoring by appending to an JSON file.
        """
        fp = os.path.join(dirname, "data", "interim", "AL", "AL_annotations",
                          "sentence_scoring_")
        fp_info = fp + "info_round_" + self.al_round + "_final.json"
        try:
            with open(fp_info, "r") as js:
                d = json.load(js)
        except OSError:
            d_sents = {}
            d = {"round":  self.al_round, "sents": d_sents}
        d["sents"].update(ClusteringActions.frozen2nested(self.sentence_scoring))
        with open(fp_info, "w", encoding="utf-8") as js:
            json.dump(d, js, indent=4)

        fp_sents = fp + "sents_round_" + self.al_round + "_final.tsv"
        with open(fp_sents, "w", encoding="utf-8") as f:
            for k, v in self.sentence_scoring.items():
                to_print = str(v) + '\t' + '\t'.join([self.sentences[i] for i in k])  + '\t' +\
                           '\t'.join([str(i) for i in k]) + '\n'
                f.write(to_print)
    
    @staticmethod
    def frozen2nested(d):
        d_out = dict()
        for k, v in d.items():
            t = [i for i in k]
            d_out[t[0]] = dict()
            d_out[t[0]][t[1]] = v
        return d_out
    
    def active_learning_HDBScan(self, embeddings, cluster_model_b, threshold_amongst_clusters=40, quota=106,
                                quota_incons=None, max_sample=1_000):
        """

        :param embeddings:  Numpy array
                            the vector representation for each sentence
        :param cluster_model_b: HDBScan model

        :param threshold_amongst_clusters:  int
                                            The number of consistent clusters to compare
        :param quota:  int
                       The number of random samples to evaluate with the closest distance consistent cluster
        :param quota_incons: int
                             Evaluate randomly sentences between inconsistent clusters.
                            Set None to skip this process.
        :param max_sample: int
                           A constraint for oversampling purposes.
        :return:
        """
        embeddings = pd.DataFrame(embeddings)
        # -------------------------------------------------------------------------------------------------------------
        # Evaluate each cluster's consistency
        consistent_clusters = list()
        for label in tqdm(range(cluster_model_b.labels_.max() + 1)):
            print("___ Cluster " + str(label) + " ___")
            buffer = []
            for idx, sentence in enumerate(self.sentences):
                if idx in np.where(cluster_model_b.labels_ == label)[0]:
                    print(sentence)
                    buffer.append(idx)
            while True:
                answer = input("Am I consistent y/n?")
                if answer == "y":
                    answer = True
                    break
                elif answer =="n":
                    answer = False
                    break
            # Permutate buffer
            if answer:
                combinations  = list(itertools.combinations(buffer, 2))
                to_ann_1 = None
                if len(combinations) > max_sample:
                    to_ann_1 = random.sample(range(0, len(combinations)), max_sample)
                for counter, comb in enumerate(combinations):
                    try:
                        if counter not in to_ann_1:
                            continue
                    except TypeError:
                        pass
                    if answer:
                        self.sentence_scoring[frozenset([int(comb[0]), int(comb[1])])] = float(5)
                self.save_sentence_scoring()
                consistent_clusters.append(label)
        print("Consistent Clusters")
        print(consistent_clusters)
        print(len(consistent_clusters))
        print(cluster_model_b.labels_.max() + 1)
        print("sentence_scoring 1")
        pprint(len(self.sentence_scoring))
        # -------------------------------------------------------------------------------------------------------------
        # Annotate between the inconsistent clusters
        if quota_incons:
            incons_sentences = self.sentences[np.in1d(cluster_model_b.labels_, consistent_clusters, assume_unique=False,
                                                      invert=True)]
            s_ids = random.sample(incons_sentences.index.to_list(), quota_incons)
            iters = list(itertools.combinations(s_ids, 2))
            for id1, id2 in tqdm(iters, total=len(iters)):
                print()
                print(self.sentences[id1])
                print(self.sentences[id2])
                while True:
                    try:
                        score = input("\nSimilar from 0 to 5: ")
                        if 0 <= float(score) <= 5:
                            break
                    except:
                        continue
                self.sentence_scoring[frozenset([int(id1), int(id2)])] = float(score)
            self.save_sentence_scoring()
        print("sentence_scoring 2")
        pprint(len(self.sentence_scoring))
        # -------------------------------------------------------------------------------------------------------------
        if len(consistent_clusters) > 1:
            # Evaluate each cluster vs cluster
            combs = list(itertools.combinations(consistent_clusters, 2))
            to_ann_2 = None  
            if len(combs) > threshold_amongst_clusters:
                to_ann_2 = random.sample(range(0, len(combs)), threshold_amongst_clusters)
            for counter, comb in tqdm(enumerate(combs), total=len(combs)):
                try:
                    if counter not in to_ann_2:
                        continue
                except TypeError:
                    pass
                print("\n _^_ Cluster A: _^_")
                clust_a_ids = []
                for idx, sentence in enumerate(self.sentences):
                        if idx in np.where(cluster_model_b.labels_ == comb[0])[0]:
                            print(sentence)
                            clust_a_ids.append(idx)

                print("\n _^_ Cluster B: _^_")
                clust_b_ids = []
                for idx, sentence in enumerate(self.sentences):
                        if idx in np.where(cluster_model_b.labels_ == comb[1])[0]:
                            print(sentence)
                            clust_b_ids.append(idx)
                while True:
                    try:
                        score = input("\nSimilar from 0 to 5: ")
                        if 0 <= float(score) <= 5:
                            break
                    except:
                        continue
                to_ann_3 = None
                combinations = list(itertools.product(clust_a_ids, clust_b_ids))
                if len(combinations) > max_sample:
                    to_ann_3 = random.sample(range(0, len(combinations)), max_sample)
                for counter_2, comb_2 in enumerate(combinations):
                    try:
                        if counter_2 not in to_ann_3:
                            continue
                    except TypeError:
                        pass
                    self.sentence_scoring[frozenset([int(comb_2[0]), int(comb_2[1])])] = float(score)
                self.save_sentence_scoring()
        print("sentence_scoring 3")
        pprint(len(self.sentence_scoring))
        # -------------------------------------------------------------------------------------------------------------
        if len(consistent_clusters) > 0:
            cons_embeddings = embeddings[np.in1d(cluster_model_b.labels_, consistent_clusters, assume_unique=False)]
            incons_embeddings = embeddings[np.in1d(cluster_model_b.labels_, consistent_clusters, assume_unique=False,
                                                   invert=True)]
            s_ids = random.sample(incons_embeddings.index.to_list(), quota)
            for id in tqdm(s_ids):
                id_con, _ = pairwise_distances_argmin_min(incons_embeddings.loc[id, :].to_numpy().reshape(1,-1),
                                                          cons_embeddings.to_numpy(), metric="cosine")
                id_con = cons_embeddings.index[id_con][0]
                # Find the its consistent cluster
                label = cluster_model_b.labels_[id_con]
                # Get all its mates
                embeddings_label = embeddings.loc[cluster_model_b.labels_ == label ,:]
                print('\n_^_ Below are cons cluster_^_ ')
                buffer_ids = []
                for con_id, sentence in self.sentences[embeddings_label.index].iteritems():
                    buffer_ids.append(con_id)
                    print(sentence)
                print("_^_ Below is the outlier _^_")
                print(self.sentences[id])
                while True:
                    try:
                        score = input("Score from 0 to 5?")
                        # score = 5
                        if 0 <= float(score) <= 5:
                            break
                    except:
                        continue
                for con_id in buffer_ids:
                    self.sentence_scoring[frozenset([int(id), int(con_id)])] = float(score)
                self.save_sentence_scoring()
        print("sentence_scoring 4")
        pprint(len(self.sentence_scoring))

## TESTING PURPOSES
    def read_info(self, date_):
        fp = os.path.join(dirname, "data", "interim", "AL",
                          date_ + ".json")
        with open(fp, 'r') as f:
            self.info = json.load(f)


    def create_sentences_test(self):
        nlp = English()
        nlp.add_pipe('sentencizer')
        len_ = self.get_collection().find().count()
        print(len_)
        self.read_info("2021_09_07")    ## Saves to self.info
        print(self.info)
        self.test_mapping = defaultdict(list)
        count = 0
        for idx, recipe in tqdm(enumerate(self.get_collection().find()), total=len_):
            print(type(recipe["_id"]))
            print(type(self.info["test_ids"]))
            if str(recipe["_id"]) not in self.info["test_ids"]:
                count = count+1
                continue
            doc = nlp(recipe['Method'])

            for sent in doc.sents:
                if sent.text and len(sent.text) > 2:
                    self.sentences_test.append(sent.text)
                    self.test_mapping[str(recipe["_id"])].append(sent.text)
        print("passed ",count)

    def save_sentences_test(self):
        """ Save to file the test sentences.
        """
        fp = os.path.join(dirname, "data", "interim", "AL",
                          "sentences_2021_09_07_test_final.json")
        with open(fp, 'w') as f:
            json.dump(self.sentences_test, f, indent=4)

    def save_mapping_test(self):
        fp = os.path.join(dirname, "data", "interim", "AL",
                          "sentences_" + "_2021_09_07_test_mapping_final.json")
        with open(fp, 'w') as f:
            json.dump(self.test_mapping, f, indent=4)

    def read_mapping_test(self):
        fp = os.path.join(dirname, "data", "interim", "AL",
                          "test_mapping.json")
        with open(fp, 'r') as f:
            self.test_mapping = json.load(f)
    
    def test_iterator(self):
        for recipe_id in self.test_mapping.keys():
            for sentence in self.test_mapping[recipe_id]:
                yield recipe_id, sentence

    def process_test_sentences(self):
        # # Create data & save
        self.create_sentences_test()
        self.save_sentences_test()
        self.save_mapping_test()
        self.read_mapping_test()  
        processing_list = [self.assign_hypernyms, self.small_caps_trail_spaces]
        for pro in tqdm(processing_list):
            sentences = defaultdict(list)
            for recipe_id, sentence in pro(test=True):
                sentences[recipe_id].append(sentence)
            self.test_mapping = sentences
        self.preprocess(test=True)
        self.save_mapping_test()

    def test(self):
        print("Doing test...")
        self.read_mapping_test()
        test_mapping_vector = dict()
        # Use the last most trained round
        # for recipe_id in tqdm(self.test_mapping.keys()):
        #     print(self.test_mapping[recipe_id])
        #     test_mapping_vector[recipe_id] = self.create_embeddings(self.test_mapping[recipe_id])
            # print(test_mapping_vector[recipe_id])

        # fp = os.path.join(dirname, "data", "interim", "AL","test_vectors.npy")
        # np.save(fp, test_mapping_vector)

        test_mapping_vector = np.load(fp, allow_pickle=True)[()]  # Conversion from numpy to dict
        pprint(test_mapping_vector)
        self.read_embeddings()  # Load latest Round 6 embeddings
        # Re-produce the best clustering result
        dist = cosine_distances(self.embeddings_np)
        cluster_model_b = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5, min_samples=2, cluster_selection_epsilon=0.01, allow_single_cluster=True)
        cluster_model_b.fit(dist.astype(np.double))
        print(metrics.silhouette_score(dist, cluster_model_b.labels_, metric="precomputed"))
        unique, counts = np.unique(cluster_model_b.labels_, return_counts=True)
        print(tabulate(np.asarray((unique, counts))))

        # # Train knn algorithm
        neigh = KNeighborsClassifier(n_neighbors=3, metric="cosine")

        # Remove the unlabelled
        y = copy.copy(cluster_model_b.labels_)
        X = copy.copy(self.embeddings_np)
        X = np.delete(X, np.where(y == -1)[0], 0)
        X = np.delete(X, np.where(y == 79)[0], 0)
        X = np.delete(X, np.where(y == 95)[0], 0)
        X = np.delete(X, np.where(y == 124)[0], 0)
        y = np.delete(y, np.where(y == -1)[0])
        y = np.delete(y, np.where(y == 79)[0])
        y = np.delete(y, np.where(y == 95)[0])
        y = np.delete(y, np.where(y == 124)[0])
        neigh.fit(X, y)

        for recipe_id in test_mapping_vector.keys():
            for counter, (vector, sentence) in enumerate(zip(test_mapping_vector[recipe_id], self.test_mapping[recipe_id])):
                 # Find the labels
                cluster = neigh.predict(vector.reshape(1, -1))
                print("_ Command " + str(counter) + " _")
                print("sentence: " + sentence)
                print(int(cluster[0]))
            print("\n")
            input()

if __name__ == '__main__':
    # ClusteringActions('cooking_methods_150', file_ext='HDBScan', al_round=0).process_train_sentences()
    # ClusteringActions('cooking_methods_500', file_ext='HDBScan', al_round=0).sentence_length()
    # ClusteringActions('cooking_methods_150', file_ext='HDBScan', al_round=0).create_embeddings()
    # ClusteringActions('cooking_methods_150', file_ext='HDBScan', al_round=6).process()

    # ClusteringActions('cooking_methods_150', file_ext='HDBScan', al_round=6,test=True).process_test_sentences()
    ClusteringActions('cooking_methods_150', file_ext='HDBScan', al_round=2).process()
    # print(c.find_one())

