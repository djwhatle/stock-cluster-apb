from bottle import route, run, get, static_file, template, view, debug
import os, boto3, string, random

@route('/')
@view('index')
def index():
    return None

@route('/token_login')
@view('token_login')
def token_login():
    return None

@route('/token_generate')
def token_generate():
    sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
    region_name = os.environ.get('AWS_REGION_NAME')
    aws_access_key = os.environ.get('AWS_ACCESS_KEY')
    aws_secret_key = os.environ.get('AWS_SECRET_KEY')

    generated_token = id_generator()

    client = boto3.client("sns", region_name=region_name, aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    response = client.publish(TopicArn=sns_topic_arn, Message="Single use code: " + generated_token)

    return dict(generated_token=generated_token)

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


def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

if __name__ == '__main__':
    run(host='0.0.0.0', port=8080, debug=True, reloader=True)
