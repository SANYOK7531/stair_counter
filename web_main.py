from app.web_app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True, use_reloader=False)
