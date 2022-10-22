Once = 0
if Once ==0:

    import snscrape.modules.twitter as sntwitter
    import requests
    import matplotlib.pyplot as plt
    import nltk
    import streamlit as st
    import pandas as pd
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import re
    import contractions
    import pickle
    import numpy as np
    from collections import Counter


    moods = {0:'sadness',
         1:'anger',
         2:'love',
         3:'surprise',
         4:'fear',
         5:'joy'}

    vectorizer = pickle.load(open("C:/Users/Shady/Downloads/vector.pickel", "rb"))
    LR = pickle.load(open("C:/Users/Shady/Downloads/LRModel.sav", "rb"))


    def Lemm(sentence):
        stop_words = set(stopwords.words('english'))
        sentence = contractions.fix(sentence)
        lemmatizer = WordNetLemmatizer()
        sentence = re.sub('[^A-z]', ' ', sentence)
        negative = ['not', 'neither', 'nor', 'but', 'however',
                    'although', 'nonetheless', 'despite', 'except',
                            'even though', 'yet','unless']
        stop_words = [z for z in stop_words if z not in negative]
        preprocessed_tokens = [lemmatizer.lemmatize(contractions.fix(temp.lower())) for temp in sentence.split() if temp not in stop_words]
        output = ' '.join([x for x in preprocessed_tokens]).strip()
        return output
        



        
    def get_tweets(username):
        global attributes_container
        attributes_container = []
        global A_L 
        A_L = []
        try:
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:'+ username).get_items()):
                
                if i>100:
                    break
                emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U00002500-\U00002BEF"  # chinese char
                    u"\U00002702-\U000027B0"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    u"\U0001f926-\U0001f937"
                    u"\U00010000-\U0010ffff"
                    u"\u2640-\u2642" 
                    u"\u2600-\u2B55"
                    u"\u200d"
                    u"\u23cf"
                    u"\u23e9"
                    u"\u231a"
                    u"\ufe0f"  # dingbats
                    u"\u3030"
                                "]+", re.UNICODE)
                r = tweet.content
                r = re.sub('@\w+ +', '', r)
                r = re.sub(emoji_pattern, '', r)
                r = re.sub(r'http\S+', '', r, flags=re.MULTILINE)
                r = str(r)
                A_L.append(r)
                try:
                    r = requests.post("https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=auto&tl=en&q="+ r).json()[0][0]
                except:
                    print('here')
                    pass
                my_bar.progress(int(i) )
                
                print(r)
                attributes_container.append([r])

                
        except:
            print('No More tweets')
        return attributes_container


    Once = 1




st.title(' > ODC and [i]NSTANT NLP Project ! :city_sunset:')
st.markdown('### Done by: *Shady Mohamed* & *Mostafa Kamal* ', 5)
st.write(':male-technologist: *Sup. Eng/ Mahmoud Bustami* & *Eng/ Rewan Hesham*')
st.markdown('# > *MOOD Detector* :full_moon_with_face:')
st.write('##### This project is a sentiment analysis project used to define the current mood according to the text given, it also takes the url of a social media account and defines the majority of the posts tends to which mood.') # Size of a sub header
st.write("")

st.write("### Moods Available:")
st.markdown("- Sadness")
st.markdown("- Joy")
st.markdown("- Fear")
st.markdown("- Anger")
st.markdown("- Love")
st.markdown("- Surprise")

st.write('')

txt = st.text_input("Enter your text:")

if txt != '' and txt !=' ':
    st.success(moods[LR.predict(vectorizer.transform([Lemm(txt)]))[0]].capitalize())


url = st.text_input("Enter the Username of the twitter Profile:", key="url")

my_bar = st.progress(0)

if url !='' and url !=' ':


    f= get_tweets(url)
    
    preds = []

    for i in range(len(f)):
        preds.append(moods[LR.predict(vectorizer.transform([Lemm(f[i][0])]))[0]].capitalize())

    dicA = {'Arabic Tweets': A_L, 'English Tweets': f}
    dfA = pd.DataFrame(dicA)
    st.dataframe(dfA)


    s = Counter(preds)

    labels = s.keys()
    sizes = s.values()

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)




    
st.write('')
st.write('')
st.write('Thank you! :heart:')
