from bottle import route, run, get, static_file, template, view, debug
import os, boto3, string, random

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

@route('/')
@view('index')
def index():
    return None

@route('/token_login')
@view('token_login')
def token_login():
    return None

@route('/queue_clustering_job')
def queue_clustering_job():

    sqs = boto3.resource(
        'sqs',
        region_name=aws_vars['sqs']['region'],
        aws_access_key_id=aws_vars['sqs']['access_key'],
        aws_secret_access_key=aws_vars['sqs']['secret_key']
    )

    queue = sqs.Queue(aws_vars['sqs']['queue_url'])
    response = queue.send_message(MessageBody='example-set', MessageGroupId=id_generator(), MessageDeduplicationId=id_generator())

    print(response)
    return(response)

@route('/admin_panel')
@view('admin_panel')
def admin_panel():
    return None

# For Static files
@get("/static/css/<filename:re:.*\.css>")
def css(filename):
    return static_file(filename, root="static/css")


@get("/static/font/<filename:re:.*\.(eot|otf|svg|ttf|woff|woff2?)>")
def font(filename):
    return static_file(filename, root="static/font")


@get("/static/img/<filename:re:.*\.(jpg|png|gif|ico|svg)>")
def img(filename):
    return static_file(filename, root="static/img")


@get("/static/js/<filename:re:.*\.js>")
def js(filename):
    return static_file(filename, root="static/js")


def id_generator(size=128, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def validate_aws_vars(aws_vars):
    vars_ok = True

    for service_name, service_var_set in aws_vars.items():
        for service_var_name, service_var_val in service_var_set.items():
            if not service_var_val:
                print(service_name + " env var not found or empty: " + service_var_name)
                vars_ok = False

    return vars_ok

if __name__ == '__main__':
    if not validate_aws_vars(aws_vars):
        print("WARNING: One or more expected environment variables is missing. Ensure that binding with SQS, SNS, and S3 was successful.")

    run(host='0.0.0.0', port=8080, debug=True, reloader=True)
