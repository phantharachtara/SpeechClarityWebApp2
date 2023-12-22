# Imports and downloads from NLTK
import nltk
from nltk.tree import Tree
from nltk.tokenize import word_tokenize
from nltk import pos_tag,ne_chunk
from nltk.corpus import brown
from nltk.corpus import reuters
nltk.download('brown')
nltk.download('reuters')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# Spacy imports
import pickle
import spacy
from spacy import displacy
# nlp = spacy.load("en_core_web_sm")
nlp = pickle.load(open('models/en_core_web_sm.pkl','rb'))

# Other imports
import numpy as np
import pandas as pd
import textstat
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

class SentenceAnalyzer():
    def __init__(self,base_corpus='brown'):
        self.grammar = "NP: {<DT>?<JJ>*<NN>}"
        self.base_corpus = base_corpus
        self.pipe = None
        self.corpus_trans_feats = None
        self.corpus_base_feats = None
    def get_corpus_sents(self):
        if self.base_corpus == 'brown':
            return brown.sents()
        elif self.base_corpus == 'reuters':
            return reuters.sents()
    def count_subtrees(self,sentence,draw_tree=False):
        doc = nlp(sentence)
        syntactic_tree = []
        for token in doc:
            syntactic_tree.append((token.text, token.dep_, token.head.text))
        num_subtrees = sum(1 for chunk in doc.noun_chunks)
        if draw_tree:
            # if len(sentence.split(' '))>10:
            options={"compact": True, "distance": 100}
            syntax_tree = displacy.render(doc, style="dep", options=options)
            return num_subtrees, syntax_tree
        return num_subtrees
    
    def get_dep_distances(self,sentence,plot_graph=False,return_doc=True):
        doc = nlp(sentence)
        distances = []
        for token in doc:
            if token.dep_ != 'punct':  # Exclude punctuation
                distances.append(token.head.i - token.i)

    
        abs_mean_dist = np.mean(np.abs(distances))
        abs_max_dist = np.max(np.abs(distances))
        mean_dist = np.mean(distances)
        max_dist = np.max(distances)
        
        if plot_graph:
            if len(sentence.split(' '))>10:
                options={"compact": True, "distance": 100}
            else:
                options = {"distance": 100}
            dep_svg = displacy.render(doc,style="dep",jupyter=False,options=options)
            if return_doc ==True:
                return distances,mean_dist,max_dist,abs_mean_dist,abs_max_dist, doc
            else:
                return distances,mean_dist,max_dist,abs_mean_dist,abs_max_dist, dep_svg
            
        return distances,mean_dist,max_dist,abs_mean_dist,abs_max_dist
    
    def compute_flesch_reading_ease(self,sentence):
        readability_score = textstat.flesch_reading_ease(sentence)
        return readability_score
    
    def get_complexity(self,sentence):
        if isinstance(sentence,list):
            sentence = [w.lower() for w in sentence if w.isalpha()]
            sentence = ' '.join(sentence)
        num_subtrees = self.count_subtrees(sentence)
        distances,mean_dist,max_dist,abs_mean_dist,abs_max_dist = self.get_dep_distances(sentence)
        # vocab_level = self.get_vocab_level(sentence)
        readability_score = self.compute_flesch_reading_ease(sentence)
        
        return num_subtrees,mean_dist,max_dist,abs_mean_dist,abs_max_dist,100-readability_score
    
    def get_baseline_stats(self,num_sent=5000):
        if self.pipe is None or self.corpus_trans_feats is None:
            sentences_features = np.empty((0,6))
            sentences=self.get_corpus_sents()
            p = np.random.permutation(len(sentences))
            sentence_indices = []
            for i in tqdm(p[:num_sent]):
                sentence = sentences[i]
                if len(sentence)>10:
                    sentence_indices.append(i)
                    sentences_features = np.row_stack((sentences_features,self.get_complexity(sentence)))
                    
            pca = PCA(n_components=1)
            scaler = StandardScaler()
            pipe = Pipeline([('scale',scaler),('pca',pca)])
            pipe.fit(sentences_features)
            self.pipe = pipe
            self.corpus_trans_feats = self.pipe.transform(sentences_features)
            self.corpus_base_feats = sentences_features
            return self

        
    
    def compare_sentence(self,sentence,get_feat_val=False):
        ''' Takes a text sentence and an optional parameter to indicate whether the sentence feature should be returned
            Returns complexity relative to the baseline dataset (Brown)'''
        features = self.get_complexity(sentence)
        features = np.array(features).reshape(1,-1)
        transformed_features = self.pipe.transform(features)
        # calculate percentage of sentences in corpus more complex
        relative_complexity = sum(self.corpus_trans_feats>transformed_features)/len(self.corpus_trans_feats)
        if get_feat_val:
            return relative_complexity, transformed_features
        else:
            return relative_complexity
        
    
    def make_hist_plot(self,sentence):
        # Creates a histogram plot to visualize complexity of a text
        data = np.array(self.corpus_trans_feats).flatten()
        relative_complexity, threshold = self.compare_sentence(sentence,get_feat_val=True)
        threshold = threshold.flatten()[0]
        relative_complexity = str(round(relative_complexity[0]*100,2))+'%'

        fig,ax = plt.subplots()

        N,bins,patches = ax.hist(data,bins=50,alpha=0.5)

        for bin,patch in zip(bins,patches):
            if bin>threshold:
                patch.set_facecolor('green')
            else:
                patch.set_facecolor('red')

        ax.vlines(threshold,ymin=0,ymax=np.max(N)*1.1)
        ax.annotate('Your input sentence',xy=(threshold+0.1,np.max(N)*1.1),
                    xytext=(threshold+0.2,np.max(N)*1.1),
                    size=10
                    )
        
        ax.set_title(f'Relative Sentence Understandability: {relative_complexity}')
        ax.set_xlabel('Ease of Comprehension')
        ax.set_ylabel('Frequency')
        ax.set_ylim((0,np.max(N)*1.3))

        return fig
            
    

class SentenceRefiner():
    def __init__(self):
        pass

    def reduce_mean_distances(self,doc):
        new_sentence = []
        for token in doc:
            if token.pos_ == 'ADJ' and token.head.pos_=='ADV':
                text = token.text
                print(f'removing {text}')
                continue
            else:
                new_sentence.append(token.text)
        new_sentence = " ".join(new_sentence)
        return new_sentence
    
    def reduce_mean_complexity(self,doc):
        new_sentence = ''
        for chunk in doc.noun_chunks:
            root_text = chunk.head.text
            noun_text = chunk.text
        print(noun_text,'-->',root_text)
        simplified = ''
        for token in chunk:
            # Ignore anything that is describing or expanding on an adjective
            if token.head.pos_ == 'ADJ':
                continue
            elif token.dep_ == 'advmod' and token.pos_ == 'ADV':
                print(token.text,token.head.text,token.head.pos_)
                continue
            elif token.pos_ == 'PUNCT':
                continue
            else:
                simplified+=token.text+' '
        print(simplified)
        if chunk.root.head.i<chunk.root.i:
            print('last word')
            last_word = new_sentence.split(' ')[-2]
            if last_word !=root_text:
                new_sentence += root_text+' '+ simplified
            else:
                new_sentence += simplified
        else:
            new_sentence += simplified + root_text+' '
        return new_sentence
    
    def traverse_and_collect_phrases_with_position(self,token,depth=0,result=None):
        
        if result is None:
            result = []
        if token.pos_ == 'DET':
            result.append((depth-1, token.i, token.text))
        else:
            # Add the token's text and position in the original text to the current depth level
            result.append((depth, token.i, token.text))

        # Recursive traversal
        for child in token.children:
            self.traverse_and_collect_phrases_with_position(child, depth + 1, result)

        return result
    
    def reduce_max_complexity(self,doc):
        # find root
        root = [token for token in doc if token.head == token][0]
        result = self.traverse_and_collect_phrases_with_position(root)
        return self.sentence_level(result,depth=2)


    def sentence_level(self,result,depth=3):
        depth_arr = np.array(result)[:,0].astype(int)
        order_arr = np.array(result)[:,1].astype(int)
        sentence_arr = np.array(result)[:,-1]

        filter_depth = depth_arr<=depth
        filtered_order = order_arr[filter_depth].argsort()
        filtered_words = sentence_arr[filter_depth]

        shortened_sentence = ' '.join(filtered_words[filtered_order])
        return shortened_sentence
    

