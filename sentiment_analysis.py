from transformers import pipeline
def sentiment(query):
 sentiment_analysis = pipeline(
    "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
)
 if(sentiment_analysis(query)[0]['label']=='POSITIVE'):
  return True
 else:
  return False


# print(sentiment_analysis("I like Transformers")[0]['label'])
# # [{'label': 'POSITIVE', 'score': 0.9987214207649231}]

# print(sentiment_analysis("don't have sufficient information. Please provide more context or details about the miniOrange SSO plugin you are referring to.")[0]['label'])
# [{'label': 'NEGATIVE', 'score': 0.9993581175804138}]

