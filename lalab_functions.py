import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import *
import subprocess
import os 
from statistics import mode
from sparknlp.pretrained import PretrainedPipeline
from pyspark.ml.feature import Word2Vec
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Bucketizer
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml.feature import VectorAssembler, MaxAbsScaler, MinMaxScaler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.classification import LogisticRegression, OneVsRest, OneVsRestModel,\
            LogisticRegressionModel, RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel

def doc2vec_embeddings(df, doc2vec_model):
    ''' 
    The doc2vec model is applied on the column "full_text",
    yielding several columns. Then the column "finished_embeddings"
    is used to create the column "doc2vec_embeddings", which is
    compatible with the Vector Assembler.
    '''
    df = doc2vec_model.transform(df)
    
    df = df.withColumn('doc2vec_embeddings', F.col('finished_embeddings')[0])
    
    return df

def vector_assembler(df, columns_for_assembling, outputCol):
#     columns = [ 'doc2vec_embeddings',  'words_fraction',  'nouns_fraction',
#      'verbs_fraction', 'others_fraction', 'uniqwd_fraction', 'toksxsent_fraction']
    assembler = VectorAssembler(inputCols = columns_for_assembling, 
                                      outputCol = outputCol)
    df = assembler.transform(df)
    return df
    
def list_folders_in_folder(folder_path):
    # Initialize an empty list to store folder names
    folder_names = []
    
    # List all items in the folder
    items = os.listdir(folder_path)
    
    # Iterate over each item
    for item in items:
        # Check if the item is a directory
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    
    return folder_names

def list_files_in_folder(folder_path):
    # Initialize an empty list to store file names
    file_names = []
    
    # List all items in the folder
    items = os.listdir(folder_path)
    
    # Iterate over each item
    for item in items:
        # Check if the item is a file
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_names.append(item)
    
    return file_names

@F.udf(IntegerType())
def vector_element_size(vector):
    return len(vector)

def transform_df(df, models_list):
    counter = 0
    columns = ['essay_id', 'score', 'scaled_score', 'features']
    cols_to_sum = []
    for model in models_list:
        predictionCol = 'pred_'+str(counter)
        df = model.transform(df).withColumnRenamed(model.getPredictionCol(), predictionCol)\
                .withColumn(predictionCol, F.col(predictionCol).cast(IntegerType()))
        columns.append(predictionCol)
        cols_to_sum.append(predictionCol)
        df = df.select(columns)
        counter += 1
    df = df.withColumn('mean_score', F.round(F.expr('+'.join(cols_to_sum))/len(cols_to_sum)))\
            .select('essay_id', 'mean_score').withColumnRenamed('mean_score', 'score')
    return df

count_sentences_udf = F.udf(lambda sentences: len(sentences), IntegerType())

@F.udf(IntegerType())
def nouns_in_list(lemma, pos):
    check_list = ['NN', 'NNS', 'NNPS', 'NNP']
    return len([lemma[j] for j in range(len(lemma)) 
                if pos[j] in check_list])

@F.udf(IntegerType())
def verbs_in_list(lemma, pos):
    check_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    return len([lemma[j] for j in range(len(lemma)) 
                if pos[j] in check_list])

@F.udf(IntegerType())
def others_in_list(lemma, pos):
    check_list = ['NN', 'NNS', 'NNPS', 'NNP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    return len([lemma[j] for j in range(len(lemma)) 
                if pos[j] not in check_list])


@F.udf(DoubleType())
def coherence(lemmas, pos, wembs, nouns_number):
    from collections import Counter
    
    def cos_similarity(list1, list2):
        dot_product = sum(x * y for x, y in zip(list1, list2))
        magnitude1 = sum(x ** 2 for x in list1) ** 0.5
        magnitude2 = sum(y ** 2 for y in list2) ** 0.5
        if not magnitude1 or not magnitude2:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
    
    if (len(lemmas) != len(pos)) or (len(pos) != len(wembs)):
        return 0.0
    
    check_list = ['NN', 'NNS', 'NNPS', 'NNP']
    filtered_lemmas = []
    filtered_wembs = []

    for j in range(len(lemmas)):
        if pos[j] in check_list:
            filtered_lemmas.append(lemmas[j])
            filtered_wembs.append(wembs[j])

    if not filtered_lemmas:
        return 0.0

    counts = Counter(filtered_lemmas)
    sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    most_frequent_nouns = []
    j = 0
    while len(most_frequent_nouns) < nouns_number and j < len(sorted_counts):
        most_frequent_nouns.append(sorted_counts[j][0])
        j += 1

    if len(most_frequent_nouns) == 0:
        return 0.0
    # using a dictionary to map nouns to their embeddings directly, 
    # ensuring that each noun gets the correct embedding
    noun_to_embedding = {lemma: wembs[idx] for idx, lemma in enumerate(filtered_lemmas)}
    embeddings_list = [noun_to_embedding[noun] for noun in most_frequent_nouns if noun in noun_to_embedding]

    if len(embeddings_list) < 2:
        return 0.0

    similarities = []
    for i, vec1 in enumerate(embeddings_list):
        for j, vec2 in enumerate(embeddings_list):
            if i < j:
                cos_sim = cos_similarity(vec1, vec2)
                similarities.append(cos_sim)

    return sum(similarities) / len(similarities) if similarities else 0.0

def sentences_and_embeddings(df):
    nouns_number = [5,10,15,20,25]
    for nn in nouns_number:
        df = df.withColumn(f'text_coherence_{nn}', 
                                coherence(   F.col('lemmas').result,
                                             F.col('pos').result,
                                             F.col('word_embeddings').embeddings,
                                             F.lit(nn))
                          )
    df = df.withColumn("num_sentences", count_sentences_udf("sentence"))
    df = df.withColumn('arr_lemmas', F.array_distinct(F.col('lemmas').result))\
                        .withColumn('unique_words', F.size('arr_lemmas'))
    #df = df.withColumn('mean_embedding', mean_embedding(F.col('sentence_embeddings').embeddings))
    #df = df.withColumn('mean_embedding', array_to_vector('mean_embedding'))
    df = df.withColumn('lemmas', F.col('lemmas').result)\
                                .withColumn('pos', F.col('pos').result)\
                                .withColumn('nouns',nouns_in_list('lemmas', 'pos'))
    df = df.withColumn('verbs',verbs_in_list('lemmas', 'pos'))
    df = df.withColumn('others',others_in_list('lemmas', 'pos'))
    df = df.withColumn("num_words", F.size("lemmas"))
    df = df.withColumn('num_tokens', F.size(F.col('token').result))
 
    return df

# to convert sparse vector column into a dense vector 
@F.udf(ArrayType(FloatType()))
def  toDense(v):
    v = DenseVector(v)
    new_array = list([float(x) for x in v])
    return new_array

def create_empty_df(spark, columns):
    # Create an empty list to hold StructField objects
    fields = []
    # Define the schema with the specified number of columns
    for col in columns:
        fields.append(StructField(col, FloatType(), True))
    schema = StructType(fields)
    # Create an empty DataFrame with the defined schema
    empty_df = spark.createDataFrame([], schema)
    return empty_df


def features_for_modeling(df):
#     df = df.withColumn('error_fraction', 
#                 (F.col('spelling_errors')/F.col('num_tokens')).cast(DoubleType()))\
    df = df.withColumn('words_fraction', 
                            (F.col('num_words')/F.col('num_tokens')).cast(DoubleType()))\
                .withColumn('nouns_fraction', 
                            (F.col('nouns')/F.col('num_words')).cast(DoubleType()))\
                .withColumn('verbs_fraction', 
                            (F.col('verbs')/F.col('num_words')).cast(DoubleType()))\
                .withColumn('others_fraction', 
                            (F.col('others')/F.col('num_words')).cast(DoubleType()))\
                .withColumn('uniqwd_fraction', 
                            (F.col('unique_words')/F.col('num_words')).cast(DoubleType()))\
                .withColumn('toksxsent_fraction', 
                        ((F.col('num_tokens')/F.col('num_sentences'))/25.0).cast(DoubleType()))\
                .withColumn('raw_length_fraction', F.length('full_text')/2072.0)
   
    return df

@F.udf(ArrayType(StringType()))
def frequency_count(lst):
    # Count the frequency of each item in the list
    freq_counter = Counter(lst)
    
    # Sort the items based on their frequency in descending order
    sorted_freq = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    #sorted_freq will output a list of tuples, where each tuple contains an item 
    #and its frequency, sorted from higher to lower frequency
    sorted_freq = [z[0] for z in sorted_freq]
    return sorted_freq

@F.udf(ArrayType(FloatType()))
def mean_embedding(embeddings):
    return [float(np.sum(x)/len(embeddings)) for x in zip(*embeddings)]

def binary_models_in_sequence(df, models, classes_to_keep):
    '''
        len(classes_to_keep) = len(models)+1
        Order of classes in classes_to_keep matches the 
        order of models in models, such that the model at
        position j in models determines the class at position j
        in classes_to_keep
    '''
    columns_to_keep = df.columns
    for j in range(len(models)):
        model = models[j]
        df = model.transform(df)
        aux = df.filter(F.col(model.getPredictionCol())==classes_to_keep[j])\
                .select('essay_id',model.getPredictionCol() )\
                .withColumnRenamed(model.getPredictionCol(), 'score')
        if j == len(models)-1:
            aux = aux.withColumn('score', 
                                 F.when(F.col('score')==classes_to_keep[j],classes_to_keep[j] )\
                                  .otherwise(classes_to_keep[j+1])
                                )
        df = df.filter(F.col(model.getPredictionCol())!=classes_to_keep[j])
        df = df.select(columns_to_keep)
        if j == 0:
            df_final = aux
        else:
            df_final = df_final.union(aux)
        if df.count() == 0:
            break
    return df_final

@F.udf(IntegerType())
def combined_result(*columns):
    scores = [3, 2, 4, 1, 5, 6]
    for col in columns:
        if col in scores:
            return col

def apply_binary_models(df, models, classes_to_keep):
    columns_to_keep = df.columns
    cols2work = []
    count = 0
    for count in range(len(models)):
        model = models[count]
        predCol = 'pred_'+str(count)
        df = model.transform(df).withColumnRenamed(model.getPredictionCol(),predCol)
        if count == len(models)-1:
            df = df.withColumn(predCol, F.when(F.col(predCol)==classes_to_keep[count],F.col(predCol))\
                                         .otherwise(classes_to_keep[count+1])
                              )
        df = df.withColumn(predCol, F.col(predCol).cast(IntegerType()))
        columns_to_keep.append(predCol)
        cols2work.append(predCol)
        df = df.select(columns_to_keep)
    df = df.withColumn('result', combined_result(*cols2work).cast(IntegerType()))\
            .select('essay_id','result')\
            .withColumnRenamed('result', 'score')
    return df

@F.udf(returnType=IntegerType())
def unmatches(token, corrected_token):
    return sum([(0,1)[token[j] != corrected_token[j]] for j in range(len(token)) ])

def rforest_classifier_cross_validation(df4model, predictionCol, labelCol, threshold=1.0):
    
    metricName = "accuracy"
    columns = df4model.columns
    evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, 
                                    predictionCol = predictionCol, metricName = metricName)
    
    maxDepth = np.arange(5,16,2)
    numTrees = np.arange(5,51,5)
    
    counter = 0
    metricValues = []
    bestValue = 0.0
    optValues = {
        'maxDepth': 0.0,
        'numTrees': 0.0,
        metricName: 0.0
    }
    
    end_cycle = 'no'
    for depth in maxDepth:
        for ntrees in numTrees:
            
            counter += 1
            df4model = df4model.select(columns)

            rForestModel = RandomForestClassifier(featuresCol='features',
                        labelCol=labelCol,
                        predictionCol = predictionCol,          
                        maxDepth = depth,
                        numTrees = ntrees).fit(df4model)

            df4model = rForestModel.transform(df4model)

            metricValue = evaluator.evaluate(df4model)

            if metricValue > bestValue:
                bestValue = metricValue
                optValues['maxDepth'] = depth
                optValues['numTrees'] = ntrees
                optValues[metricName] = bestValue
                bestModel = rForestModel.copy()
            if metricValue >= threshold:
                end_cycle = 'yes'
                break
        if end_cycle == 'yes':
            break

    return bestModel, optValues

def zip_path(file2ZipName):
    subprocess.run(['zip', '-rq', f'{file2ZipName}.zip', f'{file2ZipName}'])

@F.udf(returnType=IntegerType())
def int_score(pred_score, option):
    
    original_scores = [    1,   2,   3,   4,   5,  6]
    if option == 1:
        grade = 5*pred_score+1
        if grade <= 0:
            return 1
        if grade >= 1:
            return 6
        return round(grade)
    elif option == 2:

        scaled_scores   = [  0.2, 0.4, 0.6, 0.8, 1.0]
        
        for j in range(len(scaled_scores)):
            if round(pred_score,1) < scaled_scores[j]:
                return original_scores[j]
        return 6
    else:
    
        scaled_scores   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for j in range(1,len(scaled_scores)):
            if round(pred_score,1) <= (scaled_scores[j]+scaled_scores[j-1])/2:
                return original_scores[j-1]
            else:
                return original_scores[j]

def logistic_regression_cross_validation(df_train, predictionCol, labelCol, df_test, metricName="accuracy", 
                                            metricLabel = 0.0, threshold=1.0):
    
    columns = df_train.columns
    evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, 
                                    predictionCol = predictionCol, metricName=metricName, metricLabel=metricLabel)
    
    regPars = [0.01, 0.05, 0.1,  0.5]
    iters = [10, 50, 100, 1000]
    elasticPars = [0.0,  0.8]
    counter = 0
    metricValues = []
    bestValue = 0.0
    optValues = {
        'regParam': 0.0,
        'maxIter': 0,
        'elasticNetParam': 0.0,
        'metricValue': 0.0
    }
    
    end_cycle = 'no'
    for regPar in regPars:
        for itr in iters:
            for elasticPar in elasticPars:
                counter += 1
                df_train = df_train.select(columns)

                linearRegModel = LogisticRegression(featuresCol='features',
                            labelCol=labelCol,
                            predictionCol = predictionCol,          
                            regParam=regPar,
                            family = 'multinomial',
                            elasticNetParam=elasticPar,
                            maxIter=itr).fit(df_train)

                df_test = df_test.select(columns)
                df_test = linearRegModel.transform(df_test)
                            

                metricValue = evaluator.evaluate(df_test)
 
                if metricValue > bestValue:
                    bestValue = metricValue
                    optValues['regParam'] = regPar
                    optValues['maxIter'] = itr
                    optValues['elasticNetParam'] = elasticPar
                    optValues['metricValue'] = bestValue
                    bestModel = linearRegModel.copy()
                if metricValue >= threshold:
                    end_cycle = 'yes'
                    break
            if end_cycle == 'yes':
                break
        if end_cycle == 'yes':
            break

    return bestModel, optValues

def oneVsRestPrediction(df, models):
    original_columns = df.columns
    original_columns_2 = df.columns
    predCols = []
    probCols = []
    for j in range(len(models)):
        model = models[j]
        probColName = 'max_prob_'+str(j)
        predColName = 'pred_'+str(j)
        #last postion of the array, which is also the label of the class
        
        df = model.transform(df)
        item_pos = df.limit(1).select(F.size(vector_to_array(model.getProbabilityCol())))\
                        .collect()[0][0]-1
        df = df.withColumn(probColName, 
                               vector_to_array(model.getProbabilityCol()).getItem(item_pos)
                                )\
                    .withColumnRenamed(model.getPredictionCol(), predColName)\
                    .withColumn(predColName, F.lit(item_pos))
                                           
        predCols += [predColName]
        probCols += [probColName]
        original_columns_2 += [predColName, probColName]
        df = df.select(original_columns_2)
    df = df.withColumn('predictions', F.array(predCols))\
           .withColumn('probabilities', F.array(probCols))\
           .withColumn('final_prediction', F.col('predictions')\
                       .getItem(
                               F.expr(f"array_position(probabilities, array_max(probabilities))")-1
                               #array_position is not zero based, but 1 based index. We need to substract 1. 
                               )
                      
                      ) 
    original_columns += ['predictions', 'probabilities', 'final_prediction']
    #original_columns += ['final_prediction']
    return df.select(original_columns)

def oneVsRestPredictionClf(df, models):
    original_columns = df.columns
    original_columns_2 = df.columns
    predCols = []
    probCols = []
    for j in range(len(models)):
        model = models[j]
        
        #last postion of the array, which is also the label of the class
        outputCol = model.getOutputCol()
        df = model.transform(df)
        keys = df.limit(1).select(F.map_keys(F.col(outputCol).metadata[0])).collect()[0][0]
        for key in keys:
            if key not in ['sentence', '0']:
                target_key = key
                break
        probColName = 'max_prob_'+target_key
        predColName = 'pred_'+target_key

        df = df.withColumn(probColName, 
                               F.col(outputCol).metadata[0][target_key].cast(DoubleType())
                                )\
                    .withColumn(predColName, F.lit(target_key).cast(DoubleType()))
                                           
        predCols += [predColName]
        probCols += [probColName]
        original_columns_2 += [predColName, probColName]
        df = df.select(original_columns_2)
    df = df.withColumn('predictions', F.array(predCols))\
           .withColumn('probabilities', F.array(probCols))\
           .withColumn('final_prediction', F.col('predictions')\
                       .getItem(
                               F.expr(f"array_position(probabilities, array_max(probabilities))")-1
                               #array_position is not zero based, but 1 based index. We need to substract 1. 
                               )
                      
                      ) 
    original_columns += ['predictions', 'probabilities', 'final_prediction']
    #original_columns += ['final_prediction']
    return df.select(original_columns)

def majorityVotingClf(df, models):
    '''
        It is assumed that all the models in "models" produce
        the same label outputs    
    '''
    original_columns = df.columns
    for j in range(len(models)):
        model = models[j]
        outputCol = model.getOutputCol()
        df = model.transform(df)
        if j == 0:
            keys = df.limit(1)\
                     .select(F.map_keys(F.col(outputCol).metadata[0])).collect()[0][0]
            keys = [key for key in keys if key != 'sentence']
            keys.sort()

            df = df.withColumn('score_values', F.array([F.lit(key) for key in keys]))\
                    .withColumn('score_probabilities', 
                      F.array([F.col(outputCol).metadata[0].getItem(key) for key in keys]))
            df = df.withColumn('score_probabilities', 
                            F.col('score_probabilities').cast(ArrayType(DoubleType())))

        else:
            df = df.withColumn('current_probabilities', 
                      F.array([F.col(outputCol).metadata[0].getItem(key) for key in keys]))
            df = df.withColumn('current_probabilities', 
                            F.col('current_probabilities').cast(ArrayType(DoubleType())))
            df = df.withColumn('score_probabilities',
                F.zip_with('score_probabilities', 'current_probabilities',lambda x,y: x+y))
            
    expression = f"array_position(score_probabilities, array_max(score_probabilities))"
    df = df.withColumn('final_prediction', F.col('score_values')\
                       .getItem(
                               F.expr(expression)-1
                               #array_position is not zero based, but 1 based index. 
                               #We have to substract 1. 
                               ).cast(IntegerType())
                      
                      ) 
    original_columns += ['score_values', 'score_probabilities', 'final_prediction']
    return df.select(original_columns)

def majorityVotingRfc(df, models):
    original_columns = df.columns
    for j in range(len(models)):
        model = models[j]
        outputCol = model.getPredictionCol()
        df = model.transform(df)
        if j == 0:
 
            df = df.withColumn('score_probabilities', 
                      vector_to_array(model.getProbabilityCol()))
            original_columns.append('score_probabilities')
            df = df.select(original_columns)
        else:
            df = df.withColumn('current_probabilities', 
                      vector_to_array(model.getProbabilityCol()))
            df = df.withColumn('score_probabilities',
                      F.zip_with('score_probabilities', 
                                 'current_probabilities',lambda x,y: x+y))
            df = df.select(original_columns)
            
    expression = f"array_position(score_probabilities, array_max(score_probabilities))"
    df = df.withColumn('final_prediction', 
                               (
                               F.expr(expression)-1
                               #array_position is not zero based, but 1 based index. 
                               #We have to substract 1. 
                               ).cast(DoubleType())
                      
                      ) 
    original_columns += ['final_prediction']
    return df.select(original_columns)

def write_to_file(logfile_name, expression, option='a'):
    with open(logfile_name,option) as my_file:
        my_file.write(expression)

def refined_classification_clf(df, models, pCol):
    '''
    Essay reclassification, considering the similarity
    among near scores (score A is very similar to both
    score A-1 and score A+1), which forces to apply 
    models capable of differenciating similar scores in
    order to improve final predictions.
    
    It is assumed that the dataframe "df" was previouly
    evaluated, with the results inclded in the colum
    "pCol".
    
    The "models" list includes a series of ClassifierDl 
    models trained to separate near scores 
    (one point difference).
    '''
    scores = [1, 2, 3, 4, 5, 6]
    classes_list = []
    model_dict = {}
    for model in models:
        mlist = model.getClasses()
        mlist.sort()
        mlist = tuple([float(p) for p in mlist ])
        model_dict[mlist] = model
        classes_list.append(mlist)
    classes_list .sort()  
    aux_df = df.select(df.columns)
    for counter in range(len(classes_list)):
        classes = classes_list[counter]
        model = model_dict[classes]
        outputCol = model.getOutputCol()
        aux_df = model.transform(aux_df)
        aux_df = aux_df.withColumn('reclassification', 
                               F.col(outputCol).result[0].cast(DoubleType()))\
                       .drop(outputCol)      
        if counter == 0:
            if counter == len(classes_list)-1:#Only one tuple in classes list
                final_predictions = aux_df
            else:
                final_predictions = aux_df.filter(
                   (F.col(pCol).isin(list(classes))) & 
                   ( F.col('reclassification')==classes[0])
                )
        elif counter == len(classes_list)-1:
            final_predictions = final_predictions.union(aux_df)
        else:
            final_predictions = final_predictions\
                    .union(
                        aux_df.filter(
                                           (F.col(pCol).isin(list(classes))) & 
                                           ( F.col('reclassification')==classes[0])
                        )
                     )
        aux_df = aux_df.filter(
                               ~(
                                 (F.col(pCol).isin(list(classes))) & 
                                 ( F.col('reclassification')==classes[0])
                                )
                              )\
                       .drop('reclassification')
    return final_predictions
        

def near_scores_df(*,train_df, near_scores):
    
    near_scores.sort()
    scores = [1,2,3,4,5,6]
                
    df_a = train_df.filter(F.col('score')==near_scores[0])
    df_b = train_df.filter(F.col('score')==near_scores[1])
    
    count_a = df_a.count()
    count_b = df_b.count()
 
    if count_a > count_b:
        numerator = count_b
        fraction = count_b/count_a
        df = df_b.union(df_a.sample(withReplacement=False, 
                                    fraction = fraction, seed = 123456))
    else:
        numerator = count_a
        fraction = count_a/count_b
        df = df_a.union(df_b.sample(withReplacement=False, 
                                    fraction = fraction, seed = 123456))

    return df

def targetted_sample(*,train_df, score, fraction_coefficient = 1):
    '''
    Returns a dataframe with an additional binary column, where one of
    the labels is the target score, and the zero label is compossed by
    all the scores not near to the target.

    Here the scores near to target S are [S-1, S, S+1]
    '''
    
    scores = [1,2,3,4,5,6]
    forbiden_scores = [score-1, score, score+1]
    far_scores = [sc for sc in scores if sc not in forbiden_scores]
            
    df_a = train_df.filter(F.col('score')==score)
    df_b = train_df.filter(F.col('score').isin(far_scores))
  
    count_a = df_a.count()
    count_b = df_b.count()
   
    if count_a > count_b:
        numerator = count_b
        fraction = fraction_coefficient*count_b/count_a
        fraction = (count_b/count_a, fraction)[fraction < 1]
        df = df_b.union(df_a.sample(withReplacement=False, 
                                    fraction = fraction, seed = 123456))
    else:
        numerator = count_a
        fraction = fraction_coefficient*count_a/count_b
        fraction = (count_a/count_b, fraction)[fraction < 1]
        df = df_a.union(df_b.sample(withReplacement=False, 
                                    fraction = fraction, seed = 123456))
    df = df.withColumn(f'score_{score}', 
                       F.when(F.col('score')==score,F.col('score'))\
                        .otherwise(0)
                      )
    return df

def samplingByPairs(df, sampling_column, score_pair):
    df_1 = df.filter(F.col(sampling_column)==score_pair[0])
    df_2 = df.filter(F.col(sampling_column)==score_pair[1])
    
    c1 = df_1.count()
    c2 = df_2.count()
    if c1 > 3*c2:
        return df_2.union(df_1.sample(fraction=2.2*c2/c1,seed=123456))
    elif c2 > 3*c1:
        return df_1.union(df_2.sample(fraction=2.2*c1/c2,seed=123456))
    else:
        return df_1.union(df_2)

@F.udf(IntegerType())
def calculate_mode(arr):
    try:
        return mode(arr)
    except:
        return 0

def rforest_regressor_cross_validation(df4model, predictionCol, labelCol):
    
    columns = df4model.columns
    evaluator = RegressionEvaluator(labelCol=labelCol, 
                                    predictionCol = predictionCol, metricName="rmse")
    
    maxDepth = [5,10,15]
    numTrees = [5, 10, 20, 50]
    
    counter = 0
    metricValues = []
    bestValue = 0.0
    optValues = {
        'maxDepth': 0.0,
        'numTrees': 0.0,
        'metricValue': 0.0
    }
    
    for depth in maxDepth:
        for ntrees in numTrees:
            
            counter += 1
            df4model = df4model.select(columns)

            rForestModel = RandomForestRegressor(featuresCol='features',
                        labelCol=labelCol,
                        predictionCol = predictionCol,          
                        maxDepth = depth,
                        numTrees = ntrees).fit(df4model)

            df4model = rForestModel.transform(df4model)
#                         .withColumn(predictionCol, int_score(predictionCol).cast(DoubleType()))

            metricValue = evaluator.evaluate(df4model)

            if metricValue > bestValue:
                bestValue = metricValue
                optValues['maxDepth'] = depth
                optValues['numTrees'] = ntrees
                optValues['metricValue'] = bestValue
                bestModel = rForestModel.copy()

    # rmse - root mean squared error (default)
    # mse - mean squared error
    # r2 - r^2 metric
    # mae - mean absolute error
    # var - explained variance.

    return bestModel, optValues

def gbt_regressor_cross_validation(df4model, predictionCol, labelCol):
    
    columns = df4model.columns
    evaluator = RegressionEvaluator(labelCol=labelCol, 
                                    predictionCol = predictionCol, metricName="rmse")
    
    maxDepth = [5,10,15]
    maxIter = [5, 10, 20, 50]
    
    counter = 0
    metricValues = []
    bestValue = 0.0
    optValues = {
        'maxDepth': 0.0,
        'maxIter': 0.0,
        'metricValue': 0.0
    }
    
    for depth in maxDepth:
        for iters in maxIter:
            
            counter += 1
            df4model = df4model.select(columns)

            gbtModel = GBTRegressor(featuresCol='features',
                        labelCol=labelCol,
                        predictionCol = predictionCol,          
                        maxDepth = depth,
                        maxIter = iters).fit(df4model)

            df4model =gbtModel.transform(df4model)
#                         .withColumn(predictionCol, int_score(predictionCol).cast(DoubleType()))

            metricValue = evaluator.evaluate(df4model)

            if metricValue > bestValue:
                bestValue = metricValue
                optValues['maxDepth'] = depth
                optValues['maxIter'] = iters
                optValues['metricValue'] = bestValue
                bestModel = gbtModel.copy()

    return bestModel, optValues

def gbt_classifier_cross_validation(df4model, predictionCol, labelCol, metricName):
    
    columns = df4model.columns
    evaluator = RegressionEvaluator(labelCol=labelCol, 
                                    predictionCol = predictionCol, metricName=metricName)
    
    maxDepth = [5,10,15]
    maxIter = [5, 10, 20, 50]
    
    counter = 0
    metricValues = []
    bestValue = 0.0
    optValues = {
        'maxDepth': 0.0,
        'maxIter': 0.0,
        'metricValue': 0.0
    }
    
    for depth in maxDepth:
        for iters in maxIter:
            
            counter += 1
            df4model = df4model.select(columns)

            gbtModel = GBTClassifier(featuresCol='features',
                        labelCol=labelCol,
                        predictionCol = predictionCol,          
                        maxDepth = depth,
                        maxIter = iters).fit(df4model)

            df4model =gbtModel.transform(df4model)
#                         .withColumn(predictionCol, int_score(predictionCol).cast(DoubleType()))

            metricValue = evaluator.evaluate(df4model)

            if metricValue > bestValue:
                bestValue = metricValue
                optValues['maxDepth'] = depth
                optValues['maxIter'] = iters
                optValues['metricValue'] = bestValue
                bestModel = gbtModel.copy()

    return bestModel, optValues

def count_paragraphs(text):
    paragraphs = text.split("\n\n")
    return len(paragraphs)

@F.udf(IntegerType())
def score_decision(arr):
    return (arr[-1],arr[0])[arr[0] in arr[1:]]

@F.udf(ArrayType(IntegerType()))
def preds_array(cols): #cols is an array column
    return [col for col in cols if col != 0]

@F.udf(ArrayType(ArrayType(IntegerType())))
def tuple_key_generation(ordered_scores):
    ordered_scores.sort()
    tuples = []
    for j in range(len(ordered_scores)-1):
        for k in range(j+1, len(ordered_scores)):
            tuples.append((ordered_scores[j], ordered_scores[k]))
    return tuples

@F.udf(BooleanType())
def compare_arrays(arr, given_arr):
    return arr == given_arr

@F.udf(IntegerType())
def combined_predictions(predictions):
    from collections import Counter
    
    counts = Counter(predictions)
    sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
#     [{"label": label, "count": count} for fruit, count in sorted_counts]
    if len(sorted_counts) == 0:
        return 0
    return sorted_counts[0][0]


def separator_accuracy(file_name, expression):
    sep_acc = 0.0
    counter = 0
    with open(file_name, 'r') as file:
        for line in file:
            aux = line.split(':')
            key = aux[0].replace('\n','').strip() 
            
            if expression in key:
                sep_acc += float(aux[1])
                counter += 1
    return sep_acc/counter

def weigthed_predictions(weigth=1.0):
    '''
    Returns the most frequent score, together
    with its absolute frequence. Case need the
    counting could be weigthed by providing a 
    weigth.
    '''
    def comb_predictions(predictions):
        from collections import Counter

        counts = Counter(predictions)
        for key in list(counts.keys()):
            #multiplying the score count by the score test accuracy
            counts[key] *= weigth
        sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

        return [float(sorted_counts[0][0]), float(sorted_counts[0][1])]
    return F.udf(comb_predictions, ArrayType(FloatType()))

def similar_models_decision(df, models, 
                            outputCol='comb_pred', weigth=1.0):
    '''
    A list of similar models (same output labels) is applied on the
    dataframe df, returning the most frequent label as well
    as its frequency    
    '''
    cols2keep = df.columns
    cols2keep_2 = df.columns
    cols = []
    counter = 0
    
    for model in models:
        counter += 1
        new_col = 'p_'+str(counter)
        df = model.transform(df)\
                .withColumn(new_col,
                        F.col(model.getPredictionCol()).cast(IntegerType()) )
        cols.append(new_col)
        cols2keep.append(new_col)
        df = df.select(cols2keep)
    df = df.withColumn(outputCol, 
                weigthed_predictions(weigth)(
                                    F.array(*[col for col in cols])))
    cols2keep_2.append(outputCol)
    return df.select(cols2keep_2)

@F.udf(FloatType())
def combined_combinations(preds):
    score_dict = {
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        
    }
    if len(preds) == 1:
        return float(preds[0][0])
    for col in preds:
        if col[0] != 0:
            score_dict[int(col[0])] += col[1]
    sorted_counts = sorted(score_dict.items(), key=lambda x: (-x[1], x[0]))
    return float(sorted_counts[0][0])

def load_models_in_folder(path_to_folder, model_class, model_word):
    models_folder = list_folders_in_folder(path_to_folder)
    models_folder = [fold for fold in models_folder if model_word in fold]
    models = [model_class.load(path_to_folder+modname)
              for modname in models_folder]
    return models

def load_separators_models(separators_path, separators_keys,
                           model_class, model_word):
    separators_models = {}
    separators_folders = list_folders_in_folder(separators_path)
    for key in separators_keys:
        label = ''.join(str(p) for p in key)
        key_folder = [mname for mname in separators_folders if label in mname][0]
        key_names = list_folders_in_folder(separators_path+key_folder)
        key_names = [mname for mname in key_names if model_word in mname]
        _path = separators_path+key_folder+'/'
        separators_models[key] = [model_class.load(_path+modname)
                                for modname in key_names]
    return separators_models

def apply_separators_and_predict(df, separators_keys, 
                                 separators_models, 
                                 predictionCol):
    pcols = []
    for key in separators_keys:
        models = separators_models[key]
        label = 'sep_'+''.join(str(p) for p in key)+'_pred'
        df = similar_models_decision(df, models, outputCol=label)
        df = df.checkpoint()
        pcols.append(label)
        
    for sep_pred in pcols:
        df = df.withColumn(sep_pred, F.col(sep_pred)[0].cast(DoubleType())) 
    df = df.withColumn(predictionCol, 
                combined_predictions(
                    F.array(*[F.col(col)\
                              .cast(IntegerType()) for col in pcols]))\
                       .cast(DoubleType()))
    return df

def keys_to_apply_generator(ref_key, separators_keys):
    if ref_key == 0 or ref_key == 123456:
        return separators_keys
    ref_key = str(ref_key)
    keys2apply = []
    for key in ref_key:
        aux_list = [tup for tup in separators_keys if int(key) in tup]
        keys2apply += aux_list
    keys2apply = list(set(keys2apply))
    keys2apply.sort()
    return keys2apply

def bisKmeansEvaluation(data, max_num_clusters, featuresCol, predictionCol):
    scoresData = []
    c_evaluator = ClusteringEvaluator(predictionCol=predictionCol, 
                                  featuresCol=featuresCol, metricName='silhouette')
    for i in range(2,max_num_clusters+1):
        tm = time.time()
        bkm_model = BisectingKMeans(featuresCol=featuresCol, k=i, predictionCol=predictionCol).fit(data)
        output = bkm_model.transform(data)
        score = c_evaluator.evaluate(output)
        cost = bkm_model.summary.trainingCost 
        silhouette_score.append(score) #Silhouette with squared euclidean distance
        cost_score.append(cost) #Sum of Squared Errors  
        scoresData.append((i,score,cost))
    scoresData = pd.DataFrame(scoresData, columns = ['Number of clusters','Silhouette score', 'Cost score'])
    return scoresData

def score_probability(target_score, models_applied_number=100):
    def prediction_for_entry(prediction): #array including the prediction and the number of models that predicted it
        ratio = prediction[1]/models_applied_number
        return (1-ratio, ratio)[prediction[0]==target_score]
    return F.udf(prediction_for_entry, DoubleType())

def uniform_samples_for_score(target_score, tdf, fraction):
    tdf.cache()
    columns_needed = tdf.columns
    scores = [1,2,3,4,5,6]
    score_sizes = {score: tdf.filter(F.col('score')==score).count() for score in scores}
    df_target = tdf.filter(F.col('score')==target_score).sample(fraction=fraction)
    tgtcount = df_target.count()
    bucket_cols = []
    complement_scores = []
    for score in scores:
        if score != target_score:
            
            aux = tdf.filter(F.col('score')==score)
            sccount = aux.count()
            num_groups = (1,int(sccount/tgtcount))[sccount>tgtcount]
            new_col = 'buckets_'+str(score)+f'_for_{target_score}'
            aux = aux.withColumn(new_col, 
                                 (num_groups*F.rand(seed=156)).cast(IntegerType()))
            tdf = tdf.join(aux.select('essay_id',new_col), 'essay_id', how='left')
            bucket_cols.append(new_col)
            complement_scores.append(score)
            
    buckets_per_score = {}
    max_buckets = [0,0]
    for j in range(len(bucket_cols)):
        bucketCol = bucket_cols[j]
        score = complement_scores[j]
        buckets_per_score[score] = tdf.filter(F.col('score')==score).select(bucketCol)\
                                             .distinct().toPandas()[bucketCol].tolist()
#         print(f'Score: {score}. Buckets: {len(buckets_per_score[score])}')
        if len(buckets_per_score[score]) > max_buckets[1]:
            max_buckets[0] = score
            max_buckets[1] = len(buckets_per_score[score])

    max_buckets = tuple(max_buckets)
    new_scores = [score for score in complement_scores if score != max_buckets[0]]

    max_list = buckets_per_score[max_buckets[0]]
    columns_needed += ['sample_id']
    min_score_sample = tdf.filter(F.col('score')==target_score)

    for j in range(len(max_list)):
        current_sample = tdf.filter(F.col('buckets_'+str(max_buckets[0])+f'_for_{target_score}')==max_list[j])
        for score in new_scores:
            if int(max_list[j]) < len(buckets_per_score[score]):
                current_sample = current_sample.union(
                   tdf.filter(F.col('buckets_'+str(score)+f'_for_{target_score}')==max_list[j]) 
                )
            else:
                num_buckets = len(buckets_per_score[score])
                selected_bucket = buckets_per_score[score][np.random.randint(num_buckets)]
                current_sample = current_sample.union(
                   tdf.filter(F.col('buckets_'+str(score)+f'_for_{target_score}')==selected_bucket) 
                )  
        current_sample = current_sample.union(min_score_sample)\
                                       .withColumn('sample_id', F.lit(j))\
                                       .select(columns_needed).cache()
        if j == 0:
            uniform_samples = current_sample
        else:
            uniform_samples = uniform_samples.union(current_sample)
    return uniform_samples

def binary_column_for_target_score(df, target_column, target_score, scores):
    dfs = {score: df.filter(F.col(target_column)==score) for score in scores}
    dfs_counts = {score: dfs[score].count() for score in scores}
    base_df = dfs[target_score]
    bdfcount = dfs_counts[target_score]
    quota = int(bdfcount/(len(scores)-1))
    
    dfs_quotas = {score: (dfs_counts[score],quota)[quota<dfs_counts[score]] for score in scores}
    dfs_quotas[target_score] = bdfcount
    
    dfs_fractions = {score: dfs_quotas[score]/dfs_counts[score] for score in scores}
    
    df = df.sampleBy(target_column, dfs_fractions)\
            .withColumn('binary_score', F.when(F.col(target_column)==target_score,target_score)\
                                         .otherwise(0))
    return df

def kmeansEvaluation(data, max_num_clusters, featuresCol, predictionCol):
    scoresData = []
    c_evaluator = ClusteringEvaluator(predictionCol=predictionCol, 
                                  featuresCol=featuresCol, metricName='silhouette')
    for i in range(2,max_num_clusters+1):
        tm = time.time()
        bkm_model = KMeans(featuresCol=featuresCol, k=i, predictionCol=predictionCol).fit(data)
        output = bkm_model.transform(data)
        score = c_evaluator.evaluate(output)
        cost = bkm_model.summary.trainingCost #Silhouette with squared euclidean distance
        scoresData.append((i,score,cost))
    scoresData = pd.DataFrame(scoresData, columns = ['Number of clusters','Silhouette score', 'Cost score'])
    return scoresData

# %%time

# def possible_scores(ref_list):
#     def inner_function(predictions):
#         scores = [1,2,3,4,5]
#         scores_to_exclude = [ref_list[j] for j in range(len(ref_list))
#                                 if int(predictions[j])==0]
#         posib_scores = [sc for sc in scores if sc not in scores_to_exclude]
#         return (scores, posib_scores)[len(posib_scores)>0]
#     return F.udf(inner_function,ArrayType(IntegerType()))
        
# @F.udf(DoubleType())
# def prediction_based_on_possible_scores(posib_scores, predictions):
#     allowed_predictions = [p for p in predictions if p in posib_scores]
#     if len(allowed_predictions)==0:
#         allowed_predictions = [p for p in posib_scores]
#     counts = Counter(allowed_predictions)
#     sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
#     return float(sorted_counts[0][0])
# binpreds = ['pred_1', 'pred_2','pred_4','pred_5',]
# # df_test = df_test.withColumn('possible_scores', possible_scores([1,2,4,5])(F.array(*binpreds)))
# # df_test = df_test.withColumn('final_prediction', 
# #                             prediction_based_on_possible_scores('possible_scores', F.array(*sepcols)))
# df_test = df_test.withColumn('final_prediction', 
#                             lab.weigthed_predictions()(F.array(*sepcols))[0].cast(DoubleType()))
# df_test.show(5)
