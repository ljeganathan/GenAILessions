from playwright.sync_api import sync_playwright

with sync_playwright() as p:

    # Launch browser
    browser = p.chromium.launch(headless=True)

    # Open new page
    page = browser.new_page()

    # Open website
    page.goto("https://books.toscrape.com")

    # Get all books
    books = page.locator("article.product_pod")

    # Loop through books
    for i in range(books.count()):

        # Get book name
        name = books.nth(i).locator("h3 a").get_attribute("title")

        # Get price
        price = books.nth(i).locator(".price_color").inner_text()

        print(f"Book Name: {name} | Price: {price}")

    browser.close()