#IMPORTS

from pyspark.sql import SparkSession
from pyspark.sql.functions import array
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import udf, col
from pyspark.sql import Row
from pyspark.sql.types import StringType, StructType, StructField, ArrayType,IntegerType
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer
import random

#STOP WORDS LIST
li =['i','i\'d', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
"you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 
'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
"aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', 
"wouldn't"]

#THE FUNCTION TO RETURN THE RESPONSE BELONGING TO A SUBREDDIT
# input- mapping of label to subreddit, predicted label, intents(contains, subreddits and respective 
#responses), spark object.
#returns a response under the respective predicted class.
def output(mapping,pred,intents,spark):
    subreddit = mapping[pred]
    otpt = list(intents.toPandas()[subreddit])
    return [random.choice(otpt[0]),subreddit]


#A FUNCTION TO REMOVE STOP WORDS
# input- A row of the dataframe
#returns the cleaned parent comment along with the rest of the fields
def remove_stop(x,li):
    tep=[]
    for j in str(x[3]).split(" "):
        if(j.lower() not in li):
            tep.append(j.lower())
    return (x[3],x[5]," ".join(tep))


#A FUNCTION TO REMOVE STOP WORDS FOR THE USER INPUT
# input- sentence and a list of stopwords
#returns the cleaned sentence
def new_stop(x,li):
    tep = []
    for j in x.split(" "):
        if j.lower() not in li :
            tep.append(j.lower())
    return " ".join(tep)


#PROCESSING THE USER INPUT TO THEN PASS THROUGH THE MODEL 
#input is the user input text with all the required preprocessors 
#returns the subreddit class the model predicted
def bag_of_words(inp,hashfunc,idfmod,tokenizer,model):
    x = [(Row(new_stop(inp,li)))]#(remove stop words)
    schema = ["Filtered_data"]
    # # Apply the schema to the RDD.
    sP = spark.createDataFrame(x, schema) #(make it a dataframe)
    yellow = sP.select("Filtered_data")
    yellow.show()
    sP = tokenizer.transform(sP) #(tokenize the input)
    sP = hashfunc.transform(sP) #(find term frequency)
    rDD = idfmod.transform(sP) #(get the inverse document frequency)
    result = model.transform(rDD) #(pass it through the model)
    result.show()
    predictionAndLabels = result.select("prediction")#(get the predicted class)
    return predictionAndLabels.toPandas()['prediction'][0]
    

#MAIN FUNTION
if __name__ == "__main__":
 #Spark session is a unified entry point of a spark application from Spark 2.0.
    spark = SparkSession \
        .builder \
        .appName("Chatbot") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    sc = spark.sparkContext
    
    lines = sc.textFile("/Input/test_file1.tsv")
    parts = lines.map(lambda l: l.split("\t"))
    parts1 = parts.filter(lambda p:p!=None) #(remove empty columns)
    parts2 = parts1.map(lambda x:remove_stop(x,li)) #(remove stop words)
    
    schemaString = "parent_data Subreddit Filtered_data" #(required dataframe columns)
    fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
    schema = StructType(fields)
    # # Apply the schema to the RDD.
    schemaPeople = spark.createDataFrame(parts2, schema)
    yellow = schemaPeople.select("Filtered_data") #(select only the non-stopwords parent comments column)
    yellow.show()
    
    tokenizer = Tokenizer(inputCol="Filtered_data", outputCol="words") #(tokenize the sentences)
    countTokens = udf(lambda words: len(words), IntegerType()) #(function to count the number of words in the sentence)
    schemaPeople = tokenizer.transform(schemaPeople)
    schemaPeople.select("Filtered_data", "words")\
        .withColumn("tokens", countTokens(col("words")))

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1500)
    schemaPeople = hashingTF.transform(schemaPeople)#(gets the term frequency)
    idf = IDF(inputCol="rawFeatures", outputCol="features")#(gets the inverse document frequency)
    
    idfModel = idf.fit(schemaPeople)
    rescaledData = idfModel.transform(schemaPeople)
    rescaledData.select("Subreddit","features").show()
    stringIndexer = StringIndexer(inputCol="Subreddit", outputCol="label") #(label encodes the subreddit classes)
    model = stringIndexer.fit(rescaledData)
    indexed = model.transform(rescaledData)
    
    get_dict_label = list(indexed.toPandas()['label'])
    get_dict_subreddit = list(indexed.toPandas()['Subreddit'])
    new_df = indexed.select('label','features')  #(creates a new dataframe with only the inputs ie the sentence vector and the subreddit label)
    splits = new_df.randomSplit([0.6, 0.4], 1234)  #(split it into train test, 60-30 % split)
    train = splits[0]
    test = splits[1]
    layers = [1500, 250, 184, 120]  #(define the number of neurons in each layer, input is 1500 neurons(the number of features) and output layer is of 120 neurons as we have 120 classes)
    # create the trainer and set its parameters(100 iterations and a block size of 128)
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
    # train the model
    model = trainer.fit(train)
    # try it for the test set
    result = model.transform(test)
    result.show(1600)
    predictionAndLabels = result.select("prediction", "label")
    #as spark doesn’t allow user input, we write our questions in a text file and submit to the model
    questions = sc.textFile("/Input/input.txt")
    le_name_mapping = {}

    for i,j in zip(get_dict_label,get_dict_subreddit): #(maps the subreddit class to its label)
        le_name_mapping[i] = j
    
    path = "/Input/intents.json" #(intents is a file where each subreddit class has a set of responses which we will output given a class)
    intents = spark.read.json(path)
    replies = []
    sub = []
    q= []

    for i,j in enumerate(questions.collect()):
        pred = bag_of_words(j,hashingTF,idfModel,tokenizer,model) #(pass each question to the model)
        q.append(j)
        o = output(le_name_mapping,pred,intents,spark)
        replies.append(o[0])
        sub.append(o[1])

    for i,j,k in zip(replies,q,sub): #(print the output along with its class)
        print(j+" : "+i+"--"+k+"--")
        spark.stop()
