from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
##### pandas read #######
df=pd.read_csv('ques.csv')
######vector emebdding ###
encoder= SentenceTransformer("all-mpnet-base-v2")
vectors=encoder.encode(df.Question.tolist())
dim=vectors.shape[1]

########### similarity search ###########
index=faiss.IndexFlatL2(dim)
index.add(vectors)

######### search query #########
search_query="How do i change my name?"
search_encode=encoder.encode(search_query)

######### increasing dimension #######3
svec=np.array(search_encode).reshape(1,-1)

#### print similarity indexs #######
distances,Indexes=index.search(svec,k=3)
print(df.loc[Indexes[0]])







