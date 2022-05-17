import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys
from pyproj import Transformer
import json
from shapely.geometry import Point

if __name__ == '__main__':
    sc = pyspark.SparkContext.getOrCreate()
    spark = SparkSession(sc)

  
    supermarkets = spark.read.load('./nyc_supermarkets.csv', format='csv', header=True, inferSchema=True).select('safegraph_placekey')
    weekly = spark.read.load('/tmp/bdm/weekly-patterns-nyc-2019-2020/*', format='csv', header=True, inferSchema=True, escape='"')\
        .select('placekey', 'poi_cbg', 'visitor_home_cbgs', 'date_range_start', 'date_range_end')\
        .withColumn('date_range_start', F.col('date_range_start').cast('timestamp'))\
        .withColumn('date_range_end', F.col('date_range_end').cast('timestamp'))
    # filter stores in the list
    store_filtered = weekly.join(supermarkets, weekly['placekey'] == supermarkets['safegraph_placekey'],how='inner' )

    
    centroids = pd.read_csv('./nyc_cbg_centroids.csv')
    # filter location in NY
    NY_CBG = centroids.loc[centroids['cbg_fips'].astype(str).str.startswith(('36061', '36005', '36047', '36081', '36085'))]
    NY_CBG['cbg_fips'] = NY_CBG['cbg_fips'].astype(int)
    nyc_centroid_set = set(NY_CBG['cbg_fips'])
    centroids_dict = {}
    geo_trans = Transformer.from_crs(4326, 2263)
    for k, v in zip(NY_CBG['cbg_fips'], zip(NY_CBG['latitude'], NY_CBG['longitude'])):
        centroids_dict[k] = geo_trans.transform(v[0], v[1])
    
    # Filter date
    def filter_date(start_date, end_date):
        def check_date(d):
            Mar_01_2019 = pd.datetime(2019, 3, 1, 0, 0, 0)
            Mar_31_2019 = pd.datetime(2019, 3, 31, 23, 59, 59)
            Oct_01_2019 = pd.datetime(2019, 10, 1, 0, 0, 0)
            Oct_31_2019 = pd.datetime(2019, 10, 31, 23, 59, 59)
            Mar_01_2020 = pd.datetime(2020, 3, 1, 0, 0, 0)
            Mar_31_2020 = pd.datetime(2020, 3, 31, 23, 59, 59)
            Oct_01_2020 = pd.datetime(2020, 10, 1, 0, 0, 0)
            Oct_31_2020 = pd.datetime(2020, 10, 31, 23, 59, 59)
            if Mar_01_2019 <= d <= Mar_31_2019:
                return 1
            if Oct_01_2019 <= d <= Oct_31_2019:
                return 2
            if Mar_01_2020 <= d <= Mar_31_2020:
                return 3
            if Oct_01_2020 <= d <= Oct_31_2020:
                return 4
            return 0
        start, end = check_date(start_date), check_date(end_date)
        #  either the start or the end date falls within the period
        if start or end:
            return max(start, end)
        else:
            return 0

    fun1 = F.udf(filter_date)
    filtered_date = store_filtered.withColumn("within_date", fun1(F.col('date_range_start'), F.col('date_range_end'))).filter(F.col('within_date') >= 1)

    def distance_compute(poi_cbg, home_cbgs):
        poi = centroids_dict.get(int(poi_cbg), None)
        if poi:
            vis_count, total_dist = 0, 0
            for key,value in json.loads(home_cbgs).items():
                count = int(value)
                home_cbg = int(key)
                home_pt = centroids_dict.get(home_cbg, None)
                if home_pt:
                    vis_count += count
                    tmp_dist = Point(poi[0], poi[1]).distance(Point(home_pt[0], home_pt[1]))/5280
                    total_dist += tmp_dist 
            if vis_count >0:
                return str(round(total_dist / vis_count, 2))

    fun2 = F.udf(distance_compute)
    weeklyDistance = filtered_date.withColumn("dist", fun2(F.col('poi_cbg'),F.col('visitor_home_cbgs'))).select('placekey', 'poi_cbg', 'within_date', F.col('dist'))
    
    output = weeklyDistance.groupBy('poi_cbg').pivot('within_date').agg(F.first('dist')).na.fill('').sort('poi_cbg', ascending=True)\
    .select(F.col('poi_cbg').alias('cbg_fips'), F.col('1').alias('2019-03'), F.col('2').alias('2019-10'), F.col('3').alias('2020-03'), F.col('4').alias('2020-10'))
    # output = final_pivot.na.fill('').sort('poi_cbg', ascending=True).select(F.col('poi_cbg').alias('cbg_fips'), F.col('1').alias('2019-03'), F.col('2').alias('2019-10'), F.col('3').alias('2020-03'), F.col('4').alias('2020-10'))
    output.rdd.map(tuple).saveAsTextFile(sys.argv[1])