# Heritage Translate

This project contains a Python script for translating text content stored in a SQLite database. It uses machine translation models to translate text between English, French, and Spanish.

## About

The `translate.py` script connects to a SQLite database, reads text from a `source` table, and then uses pre-trained models from the Hugging Face `transformers` library to perform the translation. The script is designed to:

-   Detect the source language of the text (English or French).
-   Translate the text into the other two languages (English, French, and Spanish).
-   Update the database with the translated text.

## Getting Started

### Prerequisites

-   Python 3
-   A SQLite database with a `source` table. The script expects the table to have columns for `id`, `language`, `text`, `english`, `french`, and `spanish`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/scottsyms/Heritage.translate.git
    cd Heritage.translate
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Before running the script, you need to make sure that the path to your SQLite database is correctly configured in `translate.py`. The script currently uses a hardcoded path:

```python
dbcon = create_engine(
    'sqlite:////Users/scottsyms/code/HeritageCanada/data/sitecontent2.db')
```

Modify this line to point to your database file.

To run the translation script, execute the following command:

```bash
python translate.py
```

The script will then connect to the database, fetch the text to be translated, and update the records with the translated content.

## Dependencies

This project uses the following major Python libraries:

-   [transformers](https://huggingface.co/transformers/)
-   [torch](https://pytorch.org/)
-   [SQLAlchemy](https://www.sqlalchemy.org/)

For a full list of dependencies, see the `requirements.txt` file.
