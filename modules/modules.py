# Again. This is to get you started. Don't change anything.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
import time
from gensim import similarities
import pandas as pd
import numpy as np
from gensim import models
# import custom filters
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric, stem_text, strip_punctuation2, preprocess_string
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short
from gensim import corpora
from gensim.test.utils import common_corpus, common_dictionary
from gensim.similarities import MatrixSimilarity
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
from pathlib import Path

def pre_process_dataframe(df, numeric, categorical, missing=np.nan, missing_num='mean', missing_cat = 'most_frequent'):
    """This will use a data imputer to fill in missing values and standardize numeric features.
       Save this one for the future. Will save you a lot of time.
    """
    #Create a data imputer for numeric values
    imp_num = SimpleImputer(missing_values=missing, strategy=missing_num)
    #Create a pipeline which imputes values and then usese the standard scaler.
    pipe_num = make_pipeline(imp_num ) # StandardScaler()
    #Create a different imputer for categorical values.
    imp_cat = SimpleImputer(missing_values=missing, strategy=missing_cat)
    enc_cat=  OneHotEncoder(drop= 'first').fit(df[categorical])
    pipe_cat = make_pipeline(imp_cat,enc_cat)
    #Use column transformer to
    preprocessor = make_column_transformer((pipe_cat, categorical),(pipe_num, numeric))
    #Get feature names
    cat_features=list(enc_cat.get_feature_names(categorical))
    #combine categorical features
    cols= cat_features+numeric
    #generate new dataframe
    df=pd.DataFrame(preprocessor.fit_transform(df), columns=cols)
    df[cat_features]=df[cat_features].astype(int)
    return df



    # define custom filters
    CUSTOM_FILTERS = [lambda x: x.encode('utf-8').strip(),
                      lambda x: x.lower(), #lowercase
                      strip_multiple_whitespaces,# remove repeating whitespaces
                      strip_numeric, # remove numbers
                      strip_punctuation, #remove punctuation
                      remove_stopwords,# remove stopwords
                      strip_short, # remove words less than minsize=3 characters long
                      stem_text # return porter-stemmed text,
                     ]

    # process text using custom filters
    def preprocess(x, filters):
        results=preprocess_string(x, filters )
        return results

    def bow(x, dictionary):
        return dictionary.doc2bow(x)

    def transform(x, model):
        return model[x]

    def transform_doc(x):
        out= []
        for i in range(len(x)):
            out.append((i,x[i]))
        return out

    def create_gensim_lsa_model(doc_clean, number_of_topics, words):
        """
        Input  : clean document, number of topics and number of words associated with each topic
        Purpose: create LSA model using gensim
        Output : return LSA model
        """

        # generate LSA model
        lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
        return lsamodel

      #https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
      def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
          """
          Input   : dictionary : Gensim dictionary
                    corpus : Gensim corpus
                    texts : List of input texts
                    stop : Max num of topics
          purpose : Compute c_v coherence for various number of topics
          Output  : model_list : List of LSA topic models
                    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
          """
          coherence_values = []
          model_list = []
          for num_topics in range(start, stop, step):
              # generate LSA model
              model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
              model_list.append(model)
              coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
              coherence_values.append(coherencemodel.get_coherence())
          return model_list, coherence_values

      def similarity_test(a , b, dictionary, doc_col, debug = False):
          #This creates a index across a corpus b.
          if len(b)<2:
              return pd.Series([np.nan for item in range(len(a))]).values
          if doc_col in ['tfidf_norm']:
              index = similarities.MatrixSimilarity(b,num_features=len(dictionary))
          elif doc_col in ['docvecs','lsi_norm','lda_norm']:
              index = similarities.MatrixSimilarity(b)
          #elif doc_col in ['lsi_norm','lda_norm']:
          #    index = similarities.MatrixSimilarity(b,num_features=50) #warning shouldn't be hardcoded
          else:
              print("doc_col not in known list")
          all_a=[]

          for x in a:
              similarity=index[x]
              #This sorts the array from most similar to least similar.
              similarity[::-1].sort()

              #print(len(similarity),type(similarity))
              all_a.append(similarity)
          #This takes the mean of the topK items.
          #print(len(a),len(all_a))

          return pd.Series(all_a).values

    def vectorize(df, text_col,  custom_filters, num_topics):
        print("Vectorizing: ", text_col)
        start = time.time()
        # run your code
        print("Initialize dataframe.")

        print("Applying Custom filters...", str(time.time()-start))
        df[text_col+'pro']=df[text_col].apply(preprocess, filters=custom_filters)

        print("Developing tagged document...", str(time.time()-start))
        df[text_col+'tag']=pd.Series(TaggedDocument(doc, [i]) for i, doc in enumerate(df[text_col+'pro'].to_list()))

        print("Applying dictionary...",str(time.time()-start))
        dict = corpora.Dictionary(df[text_col+'pro'].to_list())

        print("Generating BOW from dictionary...",str(time.time()-start))
        df[text_col+'bow']=df[text_col+'pro'].apply(bow, dictionary= dict)

        print("Creating TFIDF model with normalization.",str(time.time()-start))
        tfidf_norm = models.TfidfModel( df[text_col+'bow'].to_list(),  normalize=True)
        df[text_col+'tfidf_norm']=df[text_col+'bow'].apply(transform, model=tfidf_norm )

        #print("Creating TFIDF model without normalization.",str(time.time()-start))
        #tfidf_nonorm = models.TfidfModel( df['bow'].to_list(),  normalize=False)
        #df['tfidf_nonorm']=df['bow'].apply(transform, model=tfidf_nonorm )

        print("Creating doc2vec model.",str(time.time()-start))
        doc2vec = Doc2Vec(df[text_col+'tag'] , vector_size=100, window=2, min_count=1, workers=4)


        print("Transforming doc2vec model.",str(time.time()-start))
        #df['docvecs']=doc2vec.infer_vector(df['tagged'])
        #df['docvecs']=doc2vec.encode(documents=df['tagged'])
        df[text_col+'docvecs']=pd.Series([doc2vec.docvecs[x] for x in range(len(df))])
        df[text_col+'docvecs']=df[text_col+'docvecs'].apply(transform_doc)

        print("Creating LSI Model. Normalized ",str(time.time()-start))
        lsi_model = models.LsiModel(df[text_col+'tfidf_norm'].to_list(), id2word=dict, num_topics=num_topics)
        print("Transforming LSI Model.",str(time.time()-start))
        df[text_col+'lsi_norm']=df[text_col+'tfidf_norm'].apply(transform, model=lsi_model)

        print("Creating LDA Model. Normalized ",str(time.time()-start))
        lda_model = models.LdaModel(df[text_col+'tfidf_norm'].to_list(), id2word=dict, num_topics=num_topics, minimum_probability=0)
        print("Transforming LDA Model.",str(time.time()-start))
        df[text_col+'lda_norm']=df[text_col+'tfidf_norm'].apply(transform, model=lda_model)

        print("Returning data.",str(time.time()-start))
        #cossim_tfidf = MatrixSimilarity(df['tfidf'], num_features=len(dict))
        #cossim_lsi = MatrixSimilarity(df['lsi'], num_features=num_topics)

        return df, dict #, cossim_tfidf, cossim_lsi
