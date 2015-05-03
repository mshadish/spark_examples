#!usr/bin/env python
__author__ = 'mshadish'
"""
This script scans through the Open Paymets medical data
and computes the total amount contributed to each hospital

If there are any contributors whose donations comprise >= 70%
of all donations received by a particular hospital,
then that information will be print to the console as follows:

(as a single line):
[hospital],[total received],[top donor],
[top donor's donations to that hospital],
[top donor's donations as pct of total received]

Note that we have definied several global constants here
specifying open payments file path and number of cores in the cluster
for ease of editing
"""
# CONSTANTS HERE
# number of cores in the cluster
num_cores = 4
# path name of the open payments csv file
open_payments_path = '/users/mshadish/spark-1.3.1-bin-hadoop2.4/data/OPPR_ALL_DTL_GNRL_12192014.csv'
# column positions of the hospital, contributor, and amount features
hospital_pos = 7
contributor_pos = 4
amount_pos = 49

# IMPORTS
# standard imports
import csv
from StringIO import StringIO
# spark imports
from pyspark import SparkContext
from pyspark.sql import SQLContext




def readCSVLine(input_line):
    """
    Reads in a line using a CSV parser.  We'll use this to read
    from the open payments CSV file
    
    :param input_line = a line presumably from a CSV file
    
    Returns:
        A list representation of that line
    """
    # read in as a string IO object
    line = StringIO(input_line)
    
    # initialize the csv reader
    reader = csv.reader(line)
    
    # and return the given line as a list
    return reader.next()



def splitSQLRowIntoKeyValue(line):
    """
    We'll need this function to convert records in our SchemaRDD
    into (key, value) tuples for joining purposes later
    
    Takes the first element of the line as the key
    and the remaining two elements as values (as a tuple).
    The key is converted to lowercase and stripped,
    as is the first element in the value tuple.
    The second element in the value tuple, presumably a number,
    is converted to a float
    
    Returns:
        A tuple representation of (key, value)
        where value is itself a tuple
    """
    # pull out the key/hospital
    key = str(line[0]).strip().lower()
    # pull out the contributor
    value_1 = str(line[1]).strip().lower()
    # pull out the donations contributed to that hospital
    # by that contributor
    try:
        value_2 = float(line[2])
    except ValueError:
        value_2 = None
    
    return (key, (value_1, value_2))



def catchFloatConvertError(in_string):
    """
    Tries to convert the input string to a float,
    with the main intention of filtering out the header
    
    Note the use of try/catch here -- we expect most of our records
    to pass this (with the exception of the header), so we opt for
    faster performance with the try/except instead of if/else, knowing
    that we should (in theory) only throw the exception when the
    header is read in
    
    Returns True if it can be converted to a float, False otherwise
    """
    try:
        float(in_string)
        return True
    except ValueError:
        return False




if __name__ == '__main__':
    # initialize the spark and sql contexts
    sc = SparkContext(appName = 'OpenPayments')
    sqlcont = SQLContext(sc)
    
    # read in the open payments data
    open_payments = sc.textFile(open_payments_path).map(readCSVLine)
    # filter for a header, if any
    open_payments = open_payments.filter(lambda x: \
                        catchFloatConvertError(x[amount_pos-1]))
    
    # convert to SQL
    # note that this is done simply as an exercise in using spark SQL
    # and is not necessary -- we could have simply used standard RDD's
    open_payments_sql = sqlcont.inferSchema(open_payments)
    open_payments_sql.registerAsTable('open_payments')


    # define the query to pull out the hospital, contributor, and sum(donated)
    query_string = "select _7 hospital, _4 contributor, sum(_49) donated \
                    from open_payments where _7 != '' group by _7, _4"
    # and convert to an RDD
    # with key = hospital
    # and value = (contributor, total donated)
    all_contribs = sqlcont.sql(query_string).map(splitSQLRowIntoKeyValue)
    all_contribs = all_contribs.partitionBy(num_cores).cache()
    # note the use of a partitioner,
    # since we'll be joining with all of our summed data


    # compute the total sums, by hospital
    # first, we want our data to look like: (hospital, (donor, donation_amt))
    totals = all_contribs.map(lambda x: (x[0], x[1][1]))
    # reduce
    totals = totals.reduceByKey(lambda x,y: x + y)

    # join the totals with all of our contributions
    results = all_contribs.join(totals)
    # and filter for records for which the contributor
    # paid > 70% of all contributions to that hospital
    results = results.filter(lambda x: x[1][0][1] >= 0.7 * x[1][1]).collect()

    # report
    for line in results:
        # pull out the hospital
        hospital = line[0]
        # pull out the total contributed
        total_contrib = line[1][1]
        # pull out the top contributor
        top_contributor = line[1][0][0]
        # and the associated donation
        donation = line[1][0][1]
        
        # compute their donation as a percentage of the total contribution
        donation_pct = donation * 100 / total_contrib
        
        # and print the formatted results
        print '{0},${1},{2},${3},{4}%'.format(hospital,
                                              format(total_contrib, '.2f'),
                                              top_contributor,
                                              format(donation, '.2f'),
                                              format(donation_pct, '.2f'))
    # repeat for every result returned
    
    sc.stop()
