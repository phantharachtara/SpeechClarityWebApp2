from utils.AudioTranscription import *
from utils.SentenceAnalysis import *
import os
import streamlit as st
from audiorecorder import audiorecorder
import pickle
from spacy_streamlit import visualize_parser
st.set_page_config(layout='wide')
import ffmpeg

# Generate big title
big_title = '<b style="font-family:sans-serif; color:Black; font-size:40px;">How good are your explanations?</b>'

st.markdown(big_title,unsafe_allow_html=True)
# st.title("How good is your explanation?")
st.write('---')

st.markdown('<b style="font-family:sans-serif; color:Black; font-size:30px;">Record yourself explaining something</b>',unsafe_allow_html=True)
print(os.listdir())
audio = audiorecorder("Click to record", "Click to stop recording")

transcriber = AudioTranscription()
# analyzer = pickle.load(open('models/medium_brown_analyzer.pkl','rb'))
analyzer = pickle.load(open('models/small_reuters_analyzer.pkl','rb'))
advisor = SentenceRefiner()

if len(audio) > 0:
    st.audio(audio.export().read())  
    data_path = "audio.wav"
    # data_path = "temp_data/audio.wav"
    audio.export(data_path, format="wav",bitrate=16000)
    text = transcriber.process_audio_file(data_path)

  
    if len(text) == 0:
        st.write("No words detected by the algorithm. Please try again")
    else:
        st.write('\n\n\n\n\n\n\n\n')
        sentence_score,feature_value = analyzer.compare_sentence(text,get_feat_val=True)
        sentence_score = sentence_score[0]
        sentence_score_text = "Sentence Score: "+str(round(sentence_score,2)*100)+'%'
        st.subheader(sentence_score_text)
        st.subheader("Transcribed Sentence: "+text[0].capitalize()+text[1:])
        # st.write('---')
        # Create tabs
        base_tab, analysis_tab, suggestion_tab = st.tabs(['Performance', 'Analysis','Recommendations (BETA)'])
        css_tab = '''
                    <style>
                        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                        font-size:20px;
                        }
                    </style>
                    '''
        st.markdown(css_tab,unsafe_allow_html=True)
        
        # Generate Transcript
        with base_tab:
            # Generate Relative Complexity Plot
            fig1 = analyzer.make_hist_plot(text)
            fig1.set_figwidth(5)
            fig1.set_figheight(3)
            st.pyplot(fig1,use_container_width=False)
        with analysis_tab:        
            # Calculate statistics for plot
            # distances,mean_dist,max_dist,abs_mean_dist,abs_max_dist, dep_svg = analyzer.get_dep_distances(text,plot_graph=True)
            distances,mean_dist,max_dist,abs_mean_dist,abs_max_dist, doc = analyzer.get_dep_distances(text,plot_graph=True,return_doc=True)
            complexity_features = analyzer.get_complexity(text)
            features = ['num_subtrees','mean_dist','max_dist','abs_mean_dist','abs_max_dist','inv_ease']
            mean_base_features = analyzer.corpus_base_feats.mean(axis=0)
            std_base_features = analyzer.corpus_base_feats.std(axis=0)
            print(mean_base_features)
            feature_diff = (complexity_features-mean_base_features)/std_base_features
            print(complexity_features)
            print(std_base_features)
            print(analyzer.corpus_base_feats.max(axis=0))
                        
            # Generate Base Feature plot
            st.header('Metrics')
            color = np.repeat('',len(feature_diff))
            color[feature_diff<0] = 'green'
            color[feature_diff>=0] = 'red'
            fig2,ax2 = plt.subplots(figsize=(5,3))
            ax2.bar(features,feature_diff,color=color,alpha=0.5)
            ax2.set_title('Input sentence complexity vs Baseline average')
            lim_thresh = np.max(abs(feature_diff))*1.2
            ax2.set_ylim((-lim_thresh,lim_thresh))
            ax2.tick_params(axis='x',labelrotation=45)
            st.pyplot(fig2,use_container_width=False)
            # Explanation for features
            # st.subheader('')
            st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:35px;">
                        Syntax Tree:</b>""",unsafe_allow_html=True)
            # Syntatic Tree
            st.markdown("""<p style="font-family:sans-serif; color:Black; font-size:18px;">
                        Syntax tree gives us a way to represent a sentence structure hierarchically. This type of representation allows us to identify the subcomponents in a sentence.</p>""",unsafe_allow_html=True)
            # Subsubheader
            st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:20px;">
                        Number of Subtrees:</b>""",unsafe_allow_html=True)
            # Text
            st.markdown("""<p style="font-family:sans-serif; color:Black; font-size:18px;">
                        This metric describe the depth of the syntatx structure. A sentence with more subtrees contains more information at different levels.</p>""",unsafe_allow_html=True)
            
            # ============================================================= 
            # Dependency distance
            # st.subheader('Dependency Distance')
            st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:35px;">
                        Dependency Distance:</b>""",unsafe_allow_html=True)
            #Text
            st.markdown("""<p style="font-family:sans-serif; color:Black; font-size:18px;">
                        Dependency distance allows us to quantify how far apart words that describe an idea are from each other.</p>""",unsafe_allow_html=True)
            #-------------------------------------------
            # Mean Dependency
            st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:20px;">
                        Mean Dependency Distance:</b>""",unsafe_allow_html=True)
            st.markdown("""<p style="font-family:sans-serif; color:Black; font-size:18px;">
                        This is the average time a concept is required to be held. A sentence with high mean dependency distance requires a lot of memory.</p>""",unsafe_allow_html=True)
            #-------------------------------------------
            # Max Dependency
            st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:20px;">
                        Max Dependency Distance:</b>""",unsafe_allow_html=True)
            # Text
            st.markdown("""<p style="font-family:sans-serif; color:Black; font-size:18px;">
                        This metric is similar to the mean distance, but it captures sentences with simple but extensive appositives or nonessential clause. These sentences may require longer sustained attention.
                        </p>""",unsafe_allow_html=True)
            #-------------------------------------------
            # Abs Mean
            st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:20px;">
                        Mean Absolute Dependency Distance:</b>""",unsafe_allow_html=True)
            st.markdown("""<p style="font-family:sans-serif; color:Black; font-size:18px;">
                        By taking the absolute value of distance, we give more weight to transitive verbs with both forward and backward connections.
                        </p>""",unsafe_allow_html=True)
            #-------------------------------------------
            # Abs Max
            st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:20px;">
                        Max Absolute Dependency Distance:</b>""",unsafe_allow_html=True)
            st.markdown("""<p style="font-family:sans-serif; color:Black; font-size:18px;">
                        This metric describes the longest time required to conceptualize a relationship in the sentence.
                        </p>""",unsafe_allow_html=True)
            
            # ============================================================= 
            # Ease of Reading
            # st.subheader('Dependency Distance')
            st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:35px;">
                        Readability Metrics:</b>""",unsafe_allow_html=True)
            #Text
            st.markdown("""<p style="font-family:sans-serif; color:Black; font-size:18px;">
                        The inverse readability score describes how difficult a text is to read. The specific metric we use is the Flesch-Kincade reading ease. 
                        By inverse scaling the metric, we are able to represent complexity of sentence.</p>""",unsafe_allow_html=True)
            
            #===============================================================
        with suggestion_tab:   
            # DISPLAY SENTENCE STRUCTURE FORMAT
            # st.header('')
            st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:35px;">
                        Sentence Structure</b>""",unsafe_allow_html=True)
            # Generate Dependency Tree    
            # st.image(dep_svg,use_column_width=False,width=1080)
            visualize_parser(doc,title='')
            print(doc)
            if sentence_score>0.9:
                st.markdown("""<p style="font-family:sans-serif; color:Black; font-size:18px;">
                        Your sentence is very clear!</p>""",unsafe_allow_html=True)
            # ['num_subtrees','mean_dist','max_dist','abs_mean_dist','abs_max_dist','inv_ease']
            
            else:
                possible_sentence_1 = advisor.reduce_max_complexity(doc)
                possible_sentence_1 = f'<p style="font-family:sans-serif; color:Black; font-size:20px;">{possible_sentence_1}</p>'
                possible_sentence_2 = advisor.reduce_mean_distances(doc)
                possible_sentence_2 = f'<p style="font-family:sans-serif; color:Black; font-size:20px;">{possible_sentence_2}</p>'
                    
                print(sentence_score)
                if feature_diff[0]>0.5 or feature_diff[2]>0.5 or feature_diff[4]>0.5: # SENTENCE IS TOO LONG
                    st.markdown("""<p style="font-family:sans-serif; color:Black; font-size:20px;">
                        It seems your sentence requires a lot of memory and attention to understand. Consider shortening the sentence or bringing the subject and verb closer to each other
                                </p>""",unsafe_allow_html=True)
                    
                    st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:35px;">
                        Here is an example:
                                </b>""",unsafe_allow_html=True)
                    st.markdown(possible_sentence_1,unsafe_allow_html=True)

                elif feature_diff[3]>0.5: # Too many adejectives?
                    st.markdown("""<p style="font-family:sans-serif; color:Black; font-size:20px;">
                                It seems your sentence has a lot of details. This could potentially take away from the main point. Consider removing some adjectives or adverbs that is not needed.
                                </p>""",unsafe_allow_html=True)
                    st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:35px;">
                        Here is an example:
                                </b>""",unsafe_allow_html=True)
                    st.markdown(possible_sentence_2,unsafe_allow_html=True)

                else:
                    st.markdown("""<b style="font-family:sans-serif; color:Black; font-size:35px;">
                        Here are some ideas for improvements:
                                </b>""",unsafe_allow_html=True)
                    st.markdown(possible_sentence_1,unsafe_allow_html=True)
                    st.markdown(possible_sentence_2,unsafe_allow_html=True)

                    
                    



               