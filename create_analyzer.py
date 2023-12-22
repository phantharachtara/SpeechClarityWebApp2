from utils.SentenceAnalysis import *
import pickle

nlp = spacy.load('en_core_web_sm')
# dump the nlp model into models folder
pickle.dump(nlp,open('models/en_core_web_sm.pkl','wb'))

analyzer = SentenceAnalyzer(base_corpus='reuters')

# sentences = brown.sents()
analyzer.get_baseline_stats(5000)
pickle.dump(analyzer,open('models/small_reuters_analyzer.pkl','wb'))
# analyzer.get_baseline_stats(10000)
# pickle.dump(analyzer,open('models/medium_brown_analyzer.pkl','wb'))