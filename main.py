from flask import Flask, request, render_template
import run
app = Flask(__name__)


@app.route('/')
def start():
    page = '<form method="post" enctype="multipart/form-data" action="/facial-landmark-detection"><input type="file" name="image"><button type="submit">Send</button></form>'
    return page


@app.route('/facial-landmark-detection', methods=['POST'])
def main():
    file = request.files['image']
    file.save('static/input.jpeg')
    run.run('static/input.jpeg')
    return render_template('out.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0')
