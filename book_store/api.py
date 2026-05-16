from flask import Flask, jsonify
from playwright.sync_api import sync_playwright

app = Flask(__name__)

@app.route("/books", methods=["GET"])
def get_books():

    books_data = []

    with sync_playwright() as p:

        # Launch browser
        browser = p.chromium.launch(headless=True)

        # Open page
        page = browser.new_page()

        # Open website
        page.goto("https://books.toscrape.com")

        # Get all books
        books = page.locator("article.product_pod")

        # Loop through books
        for i in range(books.count()):

            # Book name
            name = books.nth(i).locator("h3 a").get_attribute("title")

            # Price
            price = books.nth(i).locator(".price_color").inner_text()

            books_data.append({
                "name": name,
                "price": price
            })

        browser.close()

    return jsonify(books_data)

if __name__ == "__main__":
    app.run(debug=True)