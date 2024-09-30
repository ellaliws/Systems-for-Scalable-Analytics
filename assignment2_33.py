#!/usr/bin/env python
# coding: utf-8

# # Assignment 2 DSC 102 FA23

# ## Introduction
# 
# In this assignment we will conduct data engineering for the Amazon dataset. It is divided into 2 parts. The extracted features in Part 1 will be used for the Part 2 of assignment, where you train a model (or models) to predict user ratings for a product.
# 
# We will be using Apache Spark for this assignment. The default Spark API will be DataFrame, as it is now the recommended choice over the RDD API. That being said, please feel free to switch back to the RDD API if you see it as a better fit for the task. We provide you an option to request RDD format to start with. Also you can switch between DataFrame and RDD in your solution. 
# 
# Another newer API is Koalas, which is also avaliable. However, it has constraints and is not applicable to most tasks. Refer to the PA statement for detail.

# ### Set the following parameters

# In[1]:


PID = 'A16632664' # your pid, for instance: 'a43223333'
INPUT_FORMAT = 'dataframe' # choose a format of your input data, valid options: 'dataframe', 'rdd', 'koalas'


# In[2]:


# Boiler plates, do NOT modify
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os
import getpass
from pyspark.sql import SparkSession
from utilities import SEED
from utilities import PA2Test
from utilities import PA2Data
from utilities import data_cat
from pa2_main import PA2Executor
import time
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix

os.environ['PYSPARK_SUBMIT_ARGS'] = '--py-files utilities.py,assignment2.py \
--deploy-mode client \
pyspark-shell'

class args:
    review_filename = data_cat.review_filename
    product_filename = data_cat.product_filename
    product_processed_filename = data_cat.product_processed_filename
    ml_features_train_filename = data_cat.ml_features_train_filename
    ml_features_test_filename = data_cat.ml_features_test_filename
    output_root = '/home/{}/{}-pa2/test_results'.format(getpass.getuser(), PID)
    test_results_root = data_cat.test_results_root
    pid = PID

pa2 = PA2Executor(args, input_format=INPUT_FORMAT)
data_io = pa2.data_io
data_dict = pa2.data_dict
begin = time.time()


# In[3]:


from pyspark.sql import SparkSession
import time


# In[4]:


# Import your own dependencies



#-----------------------------


# # Part 1: Feature Engineering

# In[5]:


# Bring the part_1 datasets to memory and de-cache part_2 datasets. 
# Execute this once before you start working on this Part
data_dict, _ = data_io.cache_switch(data_dict, 'part_1')


# # Task0: warm up 
# This task is provided for you to get familiar with Spark API. We will use the dataframe API to demonstrate. Solution is given to you and this task won't be graded.
# 
# Refer to https://spark.apache.org/docs/latest/api/python/pyspark.sql.html for API guide.
# 
# The task is to implement the function below. Given the ```product_data``` table:
# 1. Take and print five rows.
# 
# 1. Select only the ```asin``` column, then print five rows of it.
# 
# 1. Select the row where ```asin = B00I8KEOTM``` and print it.
# 
# 1. Count the total number of rows.
# 
# 1. Calculate the mean ```price```.
# 
# 1. You need to conduct the above operations, then extract some statistics out of the generated columns. You need to put the statistics in a python dictionary named ```res```. The description and schema of it are as follows:
#     ```
#     res
#      | -- count_total: int -- count of total rows of the entire table after your operations
#      | -- mean_price: float -- mean value of column price
#     ```

# In[6]:


def task_0(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    product_data.show(5)
    product_data[['asin']].show(5)
    product_data.where(F.col('asin') == 'B00I8KEOTM').show()
    count_rows = product_data.count()
    mean_price = product_data.select(F.avg(F.col('price'))).head()[0]
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmatically. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {'count_total': None, 'mean_price': None}
    
    # Modify res:
    
    res['count_total'] = count_rows
    res['mean_price'] = mean_price

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    return res
    # -------------------------------------------------------------------------


# In[7]:


if INPUT_FORMAT == 'dataframe':
    res = task_0(data_io, data_dict['product'])
    pa2.tests.test(res, 'task_0')


# In[ ]:





# # Task1

# In[8]:


# %load -s task_1 assignment2.py
def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    t1 = review_data[[asin_column, overall_column]].groupby(asin_column).agg(F.avg(overall_column).alias(mean_rating_column),
    F.count(overall_column).alias(count_rating_column))
    t1_data = product_data[[asin_column]].join(t1, on=asin_column, how='left')



    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:
    

    res['count_total'] = t1_data.count()
    res['mean_meanRating'] = t1_data.select(F.avg(F.col(mean_rating_column))).head()[0]
    res['variance_meanRating'] = t1_data.select(F.variance(F.col(mean_rating_column))).head()[0]
    res['numNulls_meanRating'] = t1_data.where(F.col(mean_rating_column).isNull()).count()
    res['mean_countRating'] = t1_data.select(F.avg(F.col(count_rating_column))).head()[0]
    res['variance_countRating'] = t1_data.select(F.variance(F.col(count_rating_column))).head()[0]
    res['numNulls_countRating'] = t1_data.where(F.col(count_rating_column).isNull()).count()


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


# In[9]:


start_time = time.time()


# In[ ]:


res = task_1(data_io, data_dict['review'], data_dict['product'])
pa2.tests.test(res, 'task_1')


# In[ ]:


end_time = time.time()

# Calculate the running time
running_time = end_time - start_time
print(running_time)


# 
# # Task 2

# In[ ]:


# %load -s task_2 assignment2.py
def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    def get_category(arr):
        if arr is None:
            return None
        elif len(arr) == 0:
            return None
        elif len(arr[0]) == 0:
            return None
        elif len(arr[0][0]) == 0:
            return None
        else:
            return arr[0][0]
        
    def get_salescat_rank(dic):
        if dic is None:
            return None
        if len(dic) == 0:
            return None
        return tuple(dic.keys())[0], tuple(dic.values())[0]
    
    
    t2_category = F.udf(lambda i: get_category(i), T.StringType())
    t2_salescat = F.udf(lambda i: None if get_salescat_rank(i) is None else get_salescat_rank(i)[0] , T.StringType())
    t2_rank = F.udf(lambda i: None if get_salescat_rank(i) is None else get_salescat_rank(i)[1], T.IntegerType())
    t2_data = product_data[[categories_column, salesRank_column]].select(
        t2_category(F.col(categories_column)).alias(category_column),
        t2_salescat(F.col(salesRank_column)).alias(bestSalesCategory_column),
        t2_rank(F.col(salesRank_column)).alias(bestSalesRank_column)
    )


    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:

    res['count_total'] = t2_data.count()
    res['mean_bestSalesRank'] = t2_data.select(F.avg(F.col(bestSalesRank_column))).head()[0]
    res['variance_bestSalesRank'] = t2_data.select(F.variance(F.col(bestSalesRank_column))).head()[0]
    res['numNulls_category'] = t2_data.where(F.col(category_column).isNull()).count()
    res['countDistinct_category'] = t2_data.where(F.col(category_column).isNotNull()).select(category_column).distinct().count()
    res['numNulls_bestSalesCategory'] = t2_data.where(F.col(bestSalesCategory_column).isNull()).count()
    res['countDistinct_bestSalesCategory'] = t2_data.where(F.col(bestSalesCategory_column).isNotNull()).select(bestSalesCategory_column).distinct().count()
    
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


# In[ ]:


start_time = time.time()


# In[14]:


res = task_2(data_io, data_dict['product'])
pa2.tests.test(res, 'task_2')


# In[15]:


end_time = time.time()

# Calculate the running time
running_time = end_time - start_time
print(running_time)


# # Task 3
# 
# 
# 
# 

# In[17]:


# %load -s task_3 assignment2.py
def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    def get_length(arr, attr=attribute):
        if arr is None:
            return None
        elif attr not in arr:
            return None
        elif len(arr[attr]) == 0:
            return None
        else:
            return len(arr[attr])
        
    def get_asin(arr, attr=attribute):
        if get_length(arr, attr) is None:
            return None
        else:
            return arr[attr]
            
    t3_count = F.udf(lambda i: get_length(i), T.IntegerType())
    t3_attr = F.udf(lambda i: get_asin(i), T.ArrayType(T.StringType(), False))
    
    t3_data = (product_data[[asin_column, price_column, related_column]]
        .withColumn(countAlsoViewed_column, t3_count(F.col(related_column)))
        .withColumn(attribute, t3_attr(F.col(related_column))))
    

    t3_prodprice = t3_data.select(F.col(asin_column).alias('key'), t3_data[price_column])
    t3_av = t3_data.select(t3_data[asin_column], F.explode_outer(t3_data[attribute]).alias('key'))
    
    t3_meanprice = (t3_av.join(t3_prodprice, how='left', on='key')
        .where(F.col(price_column).isNotNull())
        .groupBy(asin_column)
        .agg(F.avg(price_column).alias(meanPriceAlsoViewed_column)))
    

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:
    total_counter = t3_data.count()
    res['count_total'] = total_counter

    res['mean_meanPriceAlsoViewed'] = t3_meanprice.select(F.avg(F.col(meanPriceAlsoViewed_column))).head()[0]
    res['variance_meanPriceAlsoViewed'] = t3_meanprice.select(F.variance(F.col(meanPriceAlsoViewed_column))).head()[0]
    res['numNulls_meanPriceAlsoViewed'] = total_counter - t3_meanprice.count()
    res['mean_countAlsoViewed'] = t3_data.select(F.avg(F.col(countAlsoViewed_column))).head()[0]
    res['variance_countAlsoViewed'] = t3_data.select(F.variance(F.col(countAlsoViewed_column))).head()[0]
    res['numNulls_countAlsoViewed'] = t3_data.where(F.col(countAlsoViewed_column).isNull()).count()



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


# In[19]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("YourAppName").getOrCreate()


# In[20]:


start_time = time.time()


# In[21]:


res = task_3(data_io, data_dict['product'])
pa2.tests.test(res, 'task_3')


# In[22]:


end_time = time.time()

# Calculate the running time
running_time = end_time - start_time


# In[23]:


running_time


# # Task 4

# In[24]:


# %load -s task_4 assignment2.py
def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    t4_data = (
        product_data
        .withColumn(price_column, product_data[price_column].cast(T.FloatType()))
        .withColumn(title_column, product_data[title_column].cast(T.StringType()))
    )
    t4_data = product_data
    mean = t4_data.select(F.avg(F.col(price_column))).head()[0]
    median = t4_data.stat.approxQuantile(price_column, [0.5], 0.01)[0]


    def imputeMean(x):
        if x is None:
            return mean
        else:
            return x
    
    def imputeMedian(x):
        if x is None:
            return median
        else:
            return x
    
    def imputeUnknown(x):
        if x is None or len(x) == 0:
            return 'unknown'
        else:
            return x
    
    t4_imputeMean = F.udf(lambda x: imputeMean(x), T.FloatType())
    t4_imputeMedian = F.udf(lambda x: imputeMedian(x), T.FloatType())
    t4_imputeUnknown = F.udf(lambda x: imputeUnknown(x), T.StringType())
    
    t4_data = (
        t4_data
        .withColumn(meanImputedPrice_column, t4_imputeMean(F.col(price_column)))
        .withColumn(medianImputedPrice_column, t4_imputeMedian(F.col(price_column)))
        .withColumn(unknownImputedTitle_column, t4_imputeUnknown(F.col(title_column)))
    )
    t4_data = t4_data[[meanImputedPrice_column, medianImputedPrice_column, unknownImputedTitle_column]]






    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:

    res['count_total'] = t4_data.count()
    res['mean_meanImputedPrice'] = t4_data.select(F.avg(F.col(meanImputedPrice_column))).head()[0]
    res['variance_meanImputedPrice'] = t4_data.select(F.variance(F.col(meanImputedPrice_column))).head()[0]
    res['numNulls_meanImputedPrice'] = t4_data.where(F.col(meanImputedPrice_column).isNull()).count()
    
    res['mean_medianImputedPrice'] = t4_data.select(F.avg(F.col(medianImputedPrice_column))).head()[0]
    res['variance_medianImputedPrice'] = t4_data.select(F.variance(F.col(medianImputedPrice_column))).head()[0]
    res['numNulls_medianImputedPrice'] = t4_data.where(F.col(medianImputedPrice_column).isNull()).count()
    
    res['numUnknowns_unknownImputedTitle'] = t4_data.where(F.col(unknownImputedTitle_column) == 'unknown').count()


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


# In[25]:


start_time = time.time()


# In[26]:


res = task_4(data_io, data_dict['product'])
pa2.tests.test(res, 'task_4')


# In[27]:


end_time = time.time()

# Calculate the running time
running_time = end_time - start_time
print(running_time)


# # Task 5

# In[28]:


# %load -s task_5 assignment2.py
def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    def getTitleArray(str):
        output = str.lower().split(' ')
        return output
    
    t5_getTitleArray = F.udf(lambda x: getTitleArray(x), T.ArrayType(T.StringType(), False))
    
    product_processed_data_output = (
        product_processed_data
        .withColumn(titleArray_column, t5_getTitleArray(F.col(title_column)))
    )
    word2Vec = M.feature.Word2Vec(
        minCount=100, 
        seed=SEED, 
        numPartitions=4,
        vectorSize=16, 
        inputCol=titleArray_column, 
        outputCol="model")
    
    model = word2Vec.fit(product_processed_data_output)
    

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:
    res['count_total'] = product_processed_data_output.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


# In[29]:


start_time = time.time()


# In[30]:


res = task_5(data_io, data_dict['product_processed'], 'piano', 'rice', 'laptop')
pa2.tests.test(res, 'task_5')


# In[31]:


end_time = time.time()

# Calculate the running time
running_time = end_time - start_time
print(running_time)


# # Task 6

# In[32]:


# %load -s task_6 assignment2.py
def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------

    t6_indexer = M.feature.StringIndexer(inputCol=category_column, 
                                              outputCol=categoryIndex_column)
    t6_data = t6_indexer.fit(product_processed_data).transform(product_processed_data)
    
    t6_oneHot = M.feature.OneHotEncoder(inputCol=categoryIndex_column, 
                                  outputCol=categoryOneHot_column,
                                 dropLast=False)
    
    t6_data = t6_oneHot.fit(t6_data).transform(t6_data)
    
    t6_pca = M.feature.PCA(k=15, 
                        inputCol=categoryOneHot_column, 
                        outputCol=categoryPCA_column) 
    
    t6_data = t6_pca.fit(t6_data).transform(t6_data)



    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:
    res['count_total'] = t6_data.count()
    res['meanVector_categoryOneHot'] = t6_data.select(M.stat.Summarizer.mean(t6_data[categoryOneHot_column])).head()[0]
    res['meanVector_categoryPCA'] = t6_data.select(M.stat.Summarizer.mean(t6_data[categoryPCA_column])).head()[0]



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------


# In[33]:


start_time = time.time()


# In[34]:


res = task_6(data_io, data_dict['product_processed'])
pa2.tests.test(res, 'task_6')


# In[35]:


end_time = time.time()

# Calculate the running time
running_time = end_time - start_time
print(running_time)


# In[36]:


print ("End to end time: {}".format(time.time()-begin))


# # Part 2: Model Selection

# In[37]:


# Bring the part_2 datasets to memory and de-cache part_1 datasets.
# Execute this once before you start working on this Part
data_dict, _ = data_io.cache_switch(data_dict, 'part_2')


# # Task 7

# In[38]:


def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    def transform(data):
        return data.select(data['features'],data['overall'])
    
    t7_train_data = transform(train_data)
    t7_test_data = transform(test_data)
    
    t7_regressor = M.regression.DecisionTreeRegressor(maxDepth=5, labelCol = 'overall')
    t7_evaluator = M.evaluation.RegressionEvaluator(labelCol = 'overall')
    
    t7_model = t7_regressor.fit(t7_train_data)
    t7_result= t7_model.transform(t7_test_data)
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:
    res['test_rmse'] = t7_evaluator.evaluate(t7_result, {t7_evaluator.metricName: "rmse"})

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------


# In[39]:


start_time = time.time()


# In[40]:


res = task_7(data_io, data_dict['ml_features_train'], data_dict['ml_features_test'])
pa2.tests.test(res, 'task_7')


# In[41]:


end_time = time.time()

# Calculate the running time
running_time = end_time - start_time
print(running_time)


# # Task 8

# In[42]:


def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    
    def transform(data):
        return data.select(data['features'],data['overall'])
    
    t8_train_data = transform(train_data)
    t8_test_data = transform(test_data)
    t8_train, t8_valid = t8_train_data.randomSplit([0.75, 0.25])
    
    max_depth = [5, 7, 9, 12]
    t8_evaluator = M.evaluation.RegressionEvaluator(labelCol = 'overall')

    
    best_param = None
    min_rmse = None
    t8_rmse = {}
    
    for d in max_depth:
        t8_regressor = M.regression.DecisionTreeRegressor(maxDepth=d,labelCol = 'overall')
        t8_model = t8_regressor.fit(t8_train)
        valid_res = t8_model.transform(t8_valid)
        rmse = t8_evaluator.evaluate(valid_res, {t8_evaluator.metricName: "rmse"})
        t8_rmse[d] = rmse
        if min_rmse is None or rmse < min_rmse:
            best_param = d
            min_rmse = rmse
    
    
    t8_regressor = M.regression.DecisionTreeRegressor(maxDepth=best_param,labelCol = 'overall')
    t8_model = t8_regressor.fit(t8_train_data)
    t8_test_result = t8_model.transform(t8_test_data)
    
    

    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:

    for d in max_depth:
        res[f'valid_rmse_depth_{d}'] = t8_rmse[d]
    res['test_rmse'] = t8_evaluator.evaluate(t8_test_result, {t8_evaluator.metricName: "rmse"})
    
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------


# In[43]:


start_time = time.time()


# In[44]:


res = task_8(data_io, data_dict['ml_features_train'], data_dict['ml_features_test'])
pa2.tests.test(res, 'task_8')


# In[45]:


end_time = time.time()

# Calculate the running time
running_time = end_time - start_time
print(running_time)


# In[46]:


print ("End to end time: {}".format(time.time()-begin))


# In[ ]:





# In[ ]:




