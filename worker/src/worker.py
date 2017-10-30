
print(__doc__)

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause

from datetime import datetime

import os
import sys
import time
import boto3
import string, random
import numpy as np
from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import urlencode
from sklearn import cluster, covariance, manifold

# Use matplotlib renderer friendly with headless machine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# AWS connection params provided by bind credential environment vars
aws_vars = {
    'sqs': {
        'access_key': os.environ.get('SQS_AWS_ACCESS_KEY'),
        'secret_key': os.environ.get('SQS_AWS_SECRET_KEY'),
        'queue_arn': os.environ.get('SQS_QUEUE_ARN'),
        'queue_name': os.environ.get('SQS_QUEUE_NAME'),
        'queue_url': os.environ.get('SQS_QUEUE_URL'),
        'region': os.environ.get('SQS_REGION')
    },
    'sns': {
        'access_key': os.environ.get('SNS_AWS_ACCESS_KEY'),
        'secret_key': os.environ.get('SNS_AWS_SECRET_KEY'),
        'topic_arn': os.environ.get('SNS_TOPIC_ARN'),
        'region': os.environ.get('SNS_AWS_REGION')
    },
    's3': {
        'access_key': os.environ.get('S3_BUCKET_AWS_ACCESS_KEY_ID'),
        'secret_key': os.environ.get('S3_BUCKET_AWS_SECRET_ACCESS_KEY'),
        'bucket_arn': os.environ.get('S3_BUCKET_ARN'),
        'bucket_name': os.environ.get('S3_BUCKET_NAME'),
        'region': os.environ.get('S3_BUCKET_REGION')
    }
}

# #############################################################################
# Retrieve the data from Internet

def retry(f, n_attempts=3):
    "Wrapper function to retry function calls in case of exceptions"
    def wrapper(*args, **kwargs):
        for i in range(n_attempts):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if i == n_attempts - 1:
                    raise
    return wrapper


def quotes_historical_google(symbol, date1, date2):
    """Get the historical data from Google finance.

    Parameters
    ----------
    symbol : str
        Ticker symbol to query for, for example ``"DELL"``.
    date1 : datetime.datetime
        Start date.
    date2 : datetime.datetime
        End date.

    Returns
    -------
    X : array
        The columns are ``date`` -- datetime, ``open``, ``high``,
        ``low``, ``close`` and ``volume`` of type float.
    """
    params = urlencode({
        'q': symbol,
        'startdate': date1.strftime('%b %d, %Y'),
        'enddate': date2.strftime('%b %d, %Y'),
        'output': 'csv'
    })

    print("Pulling historical data: " + symbol)

    url = 'http://finance.google.com/finance/historical?' + params
    response = urlopen(url)
    dtype = {
        'names': ['date', 'open', 'high', 'low', 'close', 'volume'],
        'formats': ['object', 'f4', 'f4', 'f4', 'f4', 'f4']
    }
    converters = {0: lambda s: datetime.strptime(s.decode(), '%d-%b-%y')}
    return np.genfromtxt(response, delimiter=',', skip_header=1,
                         dtype=dtype, converters=converters,
                         missing_values='-', filling_values=-1)

def run_clustering_job(symbol_dict, start_date=datetime(2003, 1, 1), end_date=datetime(2008, 1, 1), output_filename='results'):
    # Choose a time period reasonably calm (not too long ago so that we get
    # high-tech firms, and before the 2008 crash)
    # d1 = datetime(2003, 1, 1)
    # d2 = datetime(2008, 1, 1)
    # d1 = datetime(2017, 1, 1)
    # d2 = datetime(2017, 7, 1)

    d1 = start_date
    d2 = end_date

    symbols, names = np.array(list(symbol_dict.items())).T

    # retry is used because quotes_historical_google can temporarily fail
    # for various reasons (e.g. empty result from Google API).
    try:
        quotes = [
                retry(quotes_historical_google)(symbol, d1, d2) for symbol in symbols
        ]
    except:
        print("Failed to retrieve historical quotes for " + symbol)

    try:
        close_prices = np.vstack([q['close'] for q in quotes])
        open_prices = np.vstack([q['open'] for q in quotes])
    except:
        print("Please double-check that historical dates were provided.")
        print("Unexpected error:", sys.exc_info()[0])

    print("\n====")
    print("Calculating daily price variance")
    print("====")
    # The daily variations of the quotes are what carry most information
    variation = close_prices - open_prices

    print("\n====")
    print("Find graphical structure with unsupervised graphical lasso ML")
    print("====")
    # #############################################################################
    # Learn a graphical structure from the correlations
    edge_model = covariance.GraphLassoCV()

    # standardize the time series: using correlations rather than covariance
    # is more efficient for structure recovery
    X = variation.copy().T
    X /= X.std(axis=0)
    edge_model.fit(X)

    print("\n====")
    print("Finding clusters using affinity propogation")
    print("====")
    # #############################################################################
    # Cluster using affinity propagation

    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    n_labels = labels.max()

    complete_text_output = ""

    for i in range(n_labels + 1):
        output_i = 'Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i]))
        complete_text_output += output_i + "\n"
        print(output_i)

    print("\n====")
    print("Plot clustered nodes (stocks) onto a 2D plane for visualization")
    print("====")
    # #############################################################################
    # Find a low-dimension embedding for visualization: find the best position of
    # the nodes (the stocks) on a 2D plane

    # We use a dense eigen_solver to achieve reproducibility (arpack is
    # initiated with random vectors that we don't control). In addition, we
    # use a large number of neighbors to capture the large-scale structure.
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver='dense', n_neighbors=6)

    embedding = node_position_model.fit_transform(X.T).T

    print("\n====")
    print("Rendering graph")
    print("====")
    # #############################################################################
    # Visualization
    plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    # Display a graph of the partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                cmap=plt.cm.spectral)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r,
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002

        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.spectral(label / float(n_labels)),
                           alpha=.6))

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())


    # Save ML algorithm output
    pdf_filename = output_filename + '.pdf'
    text_filename = output_filename + '.txt'

    plt.savefig(pdf_filename)
    f = open(text_filename, "w")
    f.write(complete_text_output)
    f.close()

    return { 'text_output': complete_text_output, 'graph_filename': pdf_filename }

temporary_symbol_sets = {
    'example-set' : {
        'TOT': 'Total',
        'XOM': 'Exxon',
        'CVX': 'Chevron',
        'COP': 'ConocoPhillips',
        'VLO': 'Valero Energy',
        'MSFT': 'Microsoft',
        'IBM': 'IBM',
        'TWX': 'Time Warner',
        'CMCSA': 'Comcast',
        'CVC': 'Cablevision',
        'YHOO': 'Yahoo',
        'DELL': 'Dell',
        'HPQ': 'HP',
        'AMZN': 'Amazon',
        'TM': 'Toyota',
        'CAJ': 'Canon',
        'SNE': 'Sony',
        'F': 'Ford',
        'HMC': 'Honda',
        'NAV': 'Navistar',
        'NOC': 'Northrop Grumman',
        'BA': 'Boeing',
        'KO': 'Coca Cola',
        'MMM': '3M',
        'MCD': 'McDonald\'s',
        'PEP': 'Pepsi',
        'K': 'Kellogg',
        'UN': 'Unilever',
        'MAR': 'Marriott',
        'PG': 'Procter Gamble',
        'CL': 'Colgate-Palmolive',
        'GE': 'General Electrics',
        'WFC': 'Wells Fargo',
        'JPM': 'JPMorgan Chase',
        'AIG': 'AIG',
        'AXP': 'American express',
        'BAC': 'Bank of America',
        'GS': 'Goldman Sachs',
        'AAPL': 'Apple',
        'SAP': 'SAP',
        'CSCO': 'Cisco',
        'TXN': 'Texas Instruments',
        'XRX': 'Xerox',
        'WMT': 'Wal-Mart',
        'HD': 'Home Depot',
        'GSK': 'GlaxoSmithKline',
        'PFE': 'Pfizer',
        'SNY': 'Sanofi-Aventis',
        'NVS': 'Novartis',
        'KMB': 'Kimberly-Clark',
        'R': 'Ryder',
        'GD': 'General Dynamics',
        'RTN': 'Raytheon',
        'CVS': 'CVS',
        'CAT': 'Caterpillar',
        'DD': 'DuPont de Nemours'
    },
    'contrafund-set': {
        'NVDA': 'NVIDIA Corp',
        'AAPL': 'Apple',
        'AMZN': 'Amazon.com',
        'GOOGL': 'Alphabet',
        'FB': 'Facebook',
        'CRM': 'Salesforcecom',
        'MSFT': 'Microsoft Corp',
        'TSLA': 'Tesla',
        'ATVI': 'Activision Blizzard',
        'LULU': 'lululemon athletica',
        'V': 'Visa',
        'RHT': 'Red Hat',
        'ALNY': 'Alnylam Pharmaceuticals',
        'IONS': 'Ionis Pharmaceuticals',
        'CMCSA': 'Comcast Corp',
        'NFLX': 'Netflix',
        'MA': 'MasterCard',
        'PYPL': 'PayPal Holdings',
        'BLUE': 'bluebird bio',
        'ADBE': 'Adobe Systems',
        'KO': 'The Coca-Cola Co',
        'SLAB': 'Silicon Laboratories',
        'EA': 'Electronic Arts',
        'SKX': 'Skechers USA',
        'MCD': 'McDonald\'s',
        'SBUX': 'Starbucks Corp',
        'ALXN': 'Alexion Pharmaceuticals',
        'MMM': '3M Co',
        'DIS': 'The Walt Disney Co',
        'SAGE': 'Sage Therapeutics',
        'JBLU': 'JetBlue Airways Corp',
        'ACAD': 'ACADIA Pharmaceuticals',
        'HON': 'Honeywell International',
        'CELG': 'Celgene Corp',
        'TXN': 'Texas Instruments',
        'NKE': 'NIKE',
        'COST': 'Costco Wholesale Corp',
        'BA': 'The Boeing Co',
        'LUV': 'Southwest Airlines Co',
        'AGIO': 'Agios Pharmaceuticals',
        'AMD': 'Advanced Micro Devices',
        'PEP': 'PepsiCo',
        'UPS': 'United Parcel Service',
        'SGEN': 'Seattle Genetics',
        'BIIB': 'Biogen',
        'LXRX': 'Lexicon Pharmaceuticals',
        'ADSK': 'Autodesk',
        'CRUS': 'Cirrus Logic',
        'MDCO': 'The Medicines Company',
        'MNTA': 'Momenta Pharmaceuticals',
        'IRWD': 'Ironwood Pharmaceuticals',
        'SQ': 'Square'
    }
}

def id_generator(size=12, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def validate_aws_vars(aws_vars):
    vars_ok = True

    for service_name, service_var_set in aws_vars.items():
        for service_var in service_var_set:
            if service_var.isspace():
                print(service_name + " env var not found or empty: " + service_var)
                vars_ok = False

    return vars_ok

def publish_sns(aws_vars, message):
    sns_client = boto3.client(
        'sns',
        region_name=aws_vars['sns']['region'],
        aws_access_key_id=aws_vars['sns']['access_key'],
        aws_secret_access_key=aws_vars['sns']['secret_key']
    )
    response = sns_client.publish(TopicArn=aws_vars['sns']['topic_arn'], Message=message)
    return response

def upload_s3(aws_vars, filepath_local, filename_s3):
    # Upload file
    s3 = boto3.resource(
        's3',
        region_name=aws_vars['s3']['region'],
        aws_access_key_id=aws_vars['s3']['access_key'],
        aws_secret_access_key=aws_vars['s3']['secret_key']
    )
    s3.meta.client.upload_file(filepath_local, aws_vars['s3']['bucket_name'], filename_s3)

    # Generate unique link expiring in 10 hours
    s3_client = boto3.client(
        's3',
        region_name=aws_vars['s3']['region'],
        aws_access_key_id=aws_vars['s3']['access_key'],
        aws_secret_access_key=aws_vars['s3']['secret_key']
    )
    url = s3_client.generate_presigned_url('get_object', Params = {'Bucket': aws_vars['s3']['bucket_name'], 'Key': filename_s3}, ExpiresIn = 36000)
    return url

def read_sqs_queue(aws_vars):
    sqs = boto3.resource(
        'sqs',
        region_name=aws_vars['sqs']['region'],
        aws_access_key_id=aws_vars['sqs']['access_key'],
        aws_secret_access_key=aws_vars['sqs']['secret_key']
    )

    # Read messages from queue
    queue = sqs.Queue(url=aws_vars['sqs']['queue_url'])

    message_body_list = []

    for message in queue.receive_messages():
        print("Received message: " + str(message.body))
        message_body_list.append(message.body)
        message.delete()

    return message_body_list

def main():
    if not validate_aws_vars(aws_vars):
        print("WARNING: One or more expected environment variables is missing. Ensure that binding with SQS, SNS, and S3 was successful.")

    while(True):
        print("Polling SQS queue...")
        # check if there are any new jobs on the queue
        message_body_list = read_sqs_queue(aws_vars)

        if len(message_body_list) < 1:
            time.sleep(1)

        else:
            # execute job from top of queue if found
            for message in message_body_list:
                # Temporarily only sending dataset name in message.
                # Plan to add dates later
                dataset_name = message

                print("=============================================================")
                print("================== STARTING CLUSTERING JOB ==================")
                print("=============================================================")

                output_filename = id_generator()
                start_time = time.time()
                results = run_clustering_job(temporary_symbol_sets[dataset_name], output_filename=output_filename)
                time_elapsed = str(round(time.time() - start_time, 3))

                print ("\nFinished job in " + str(time_elapsed) + " seconds.")
                print("=============================================================")
                print("======================= JOB COMPLETED =======================")
                print("=============================================================\n\n")

                text_output = results['text_output']
                graph_filename = results['graph_filename']

                # upload resulting file to s3 bucket
                s3_url = upload_s3(aws_vars, graph_filename, graph_filename)

                # clean up files
                os.remove(output_filename + ".txt")
                os.remove(output_filename + ".pdf")

                # email results including s3 link
                email_body = "Stock clustering job completed in " + time_elapsed + " seconds.\n\n" + text_output + "\n\nGraphical output download link (valid for 10 hours): " + s3_url
                email_success = publish_sns(aws_vars, email_body)

if __name__ == "__main__":
    main()
