sample_text_data = [
    "     Alabama have a lot of people  ",
    "This is a text data",
    "                        This is a text data and it is very long     ",
]

# Remove leading and trailing whitespaces
stripped_text_data = [text.strip() for text in sample_text_data]

print(stripped_text_data)

# Replace text data
replaced_text_data = [text.replace("text", "string") for text in stripped_text_data]

print(replaced_text_data)

fun = lambda string: string.upper()

# Apply function to text data

upper_text_data = [fun(text) for text in replaced_text_data]

# Cleaning and processing html

from bs4 import BeautifulSoup

html = """
<html>
    <head>
        <title>My Title</title>
    </head>
    <body>
        <h1>Heading</h1>
        <p>My paragraph</p>
    </body>
</html>
"""

soup = BeautifulSoup(html, "lxml")

print(soup.find("h1").text)
