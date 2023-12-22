from utils.SentenceAnalysis import *
import pickle

analyzer = SentenceAnalyzer(base_corpus='reuters')

# sentences = brown.sents()
analyzer.get_baseline_stats(5000)
pickle.dump(analyzer,open('models/small_reuters_analyzer.pkl','wb'))
# analyzer.get_baseline_stats(10000)
# pickle.dump(analyzer,open('models/medium_brown_analyzer.pkl','wb'))