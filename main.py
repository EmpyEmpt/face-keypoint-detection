from flask import Flask, request, render_template
import run
app = Flask(__name__)


@app.route('/')
def start():
    page = '<form method="post" enctype="multipart/form-data" action="/face-segmentation"><input type="file" name="image"><button type="submit">Send</button></form>'
    return page


@app.route('/face-segmentation', methods=['POST'])
def main():
    file = request.files['image']
    file.save(file.filename)
    run.run(file.filename)
    return render_template('out.html')


if __name__ == "__main__":
    app.run()