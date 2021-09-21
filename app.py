from flask import Flask, render_template, request, abort, url_for
from prepocessor import Preprocessor
from engine import SearchEngine

app = Flask(__name__)

search_engine = SearchEngine("data/articles/processed")
search_engine.generate_vectors()
preprocessor = Preprocessor()


@app.route("/", methods=["GET"])
def search_page():
    query = request.args.get("q") or ""
    limit = int(request.args.get("limit") or "10")
    search_results = search_engine.find_matches(preprocessor.process(query), top=limit)
    return render_template("search.html", article_titles=search_results, query_text=query)


@app.route("/details/<title>", methods=["GET"])
def details_page(title: str):
    docs_path = "./data/articles/raw/"
    try:
        with open(docs_path + title, "r") as f:
            return render_template("details.html", title=title, content=f.read())
    except FileNotFoundError:
        abort(404)


if __name__ == "__main__":
    app.run()
