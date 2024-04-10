#%%
import pandas as pd 
import numpy as np 
from bert_utils import *
import tensorflow as tf 
import keras 
from keras.layers import Input, Dense, Dropout
from keras.regularizers import L1L2
import regex as re

uncertainty_related = [
	'could adversely affect',
	'can have an adverse impact on',
	'the impact on us of adverse economic changes',
	'involve risks and uncertainties',
	'Our actual results could differ from',
	'due to certain factors',
	'Our income may suffer',
	'our operating results will suffer.',
	'could subject us to',
	'We cannot be sure that',
	'could be negatively impacted',
	'We are exposed to foreign currency exchange rate risks',
	'may impact the company in the future',
	'Such disruptions could adversely impact our ability to',
	'could materially impact our reported financial results',
	'this may magnify the adverse impact of',
	'the unfavorable currency impact could affect',
	'could be adversely impacted by',
	'uncertain geopolitical conditions',
	'foreign exchange risk',
	'would likely result in a reduction in demand for our products and services.',
	'could harm our',
	'could limit our ability to ',
	'we may be required to',
	'we cannot assure you that',
	'negative rating outlook',
	'If these ratings are not upgraded or are further downgraded', 
	'we would be required to pay a higher interest rate',
	'could have long-term adverse effects on our business.',
	'the risk that we will not be able to ',
	'we may have to undertake further restructuring initiatives',
	'would entail additional charges.',
	'There are several risks inherent in',
	'our results could be materially and adversely impacted',
	'Our income could be harmed if',
	'may infringe our intellectual property',
	'we may expend significant resources enforcing our rights',
	'may suffer competitive injury',
	'We may be required to spend significant resources to',
	'subject to catastrophic loss due to',
    'Potential disruptions could impact operations.'
    'could render our offerings obsolete',
    'could adversely affect our supply chain',
    'Litigation expenses and settlements could drain financial resources',
    'could weaken our competitive position.',
    'could disrupt production',
    'Exposure to volatile commodity prices could impact cost',
    'could lead to lost revenue',
]

random_list =  [
    "The cat sleeps on the windowsill.",
    "A rainbow appears after the rain.",
    "Books are scattered on the table.",
    "The clock ticks steadily in the quiet room.",
    "Leaves rustle in the gentle breeze.",
    "A cup of coffee sits untouched.",
    "Stars twinkle in the night sky.",
    "The sun sets behind the mountains.",
    "Children laugh in the distance.",
    "Birds chirp early in the morning.",
    "A pencil rolls off the desk.",
    "Footsteps echo in the empty hall.",
    "A door creaks open slowly.",
    "The moon casts a pale light.",
    "A dog barks in the night.",
    "The kettle whistles on the stove.",
    "Raindrops tap against the window.",
    "A car honks on the busy street.",
    "A clock strikes midnight.",
    "Shadows dance on the walls."
]

uncertainty_related.extend(random_list)
embeddings = bert_embeding(uncertainty_related)
#%%
df = pd.DataFrame(embeddings)
# %%
Y = np.where((np.array([i for i in range(len(df))])) > len(df) - len(random_list), 0, 1 )


#%%
"""
I have a few options :
- I could train an equivalent of word2vec (using fixed rolling window) whose final output is to predict the topics
    - We could mix full self-supervised learning with mask embedding and supervised manually labeled  
- I can use bert embeddings and do active learning with streaming focusing on text with certain words  
"""

#%%
m_input = Input(len(df.columns))

dense = Dense(10, 'sigmoid')(m_input)
dense = Dense(1, 'sigmoid')(dense)

model = keras.models.Model(inputs=[m_input], outputs=[dense])
model.compile(loss='binary_crossentropy')
model.fit(df, Y, epochs=20)



#%%

# ? Here we define a function to fetch sample targets 
file_idx = np.random.randint(0, len(files))

with open(files[file_idx], 'r') as text:

    text = text.read()
    text = re.sub('[^A-Za-z\. ]+', '', text)
    text = text.split('.')
    print(text[0:10])