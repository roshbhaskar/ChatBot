# ChatBot using Pyspark

A rule based chatbot built on Pyspark. <br>
Built using Reddit Dataset.<br>

## Team Members
- Sharika Valambatla <br>
- Vishwajeeth Hogale <br>
- Roshini Bhaskar <br>

## Challenges
Spark does not allow user inputs. Hence the questions for this chatbot are submited to the model using a text file 

## NOTE
1. HDFS must be installed in the system and running. (For this project)
2. Pyspark must be installed and running.
3. test_file1.tsv must uploaded to HDFS

<br>

To upload file in HDFS - <br>
1. hdfs dfs –mkdir –p /Input
2. hdfs dfs –put /location/of/test_file1.tsv /Input

To run - <br>
1. spark-submit code.py
