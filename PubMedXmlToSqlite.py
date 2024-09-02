import gzip
import logging
import re
import xml.etree.ElementTree as ET
import datetime

# import duckdb
import os
import sqlite3

XML_FOLDER = 'E:/Medline/Unprocessed'

logger = logging.getLogger(__name__)
logging.basicConfig(filename="log.txt",
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Function to parse a single PubMed XML file and extract titles and abstracts
def parse_pubmed_xml(file_path):
    pmids = []
    titles = []
    abstracts = []
    pub_dates = []
    mesh_terms = []
    delete_pmids = []

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        tree = ET.parse(f)
        root = tree.getroot()

        for article in root.findall('.//PubmedArticle'):
            pmid_elem = article.find('.//PMID')
            version = pmid_elem.get('Version')
            if version != "1":
                continue

            title_elem = article.find('.//ArticleTitle')

            pmid = int(pmid_elem.text) if pmid_elem is not None else None

            title = title_elem.text if title_elem is not None else None

            # Merge all abstract sections into a single text:
            abstract_sections = []
            for abstract_elem in article.findall('.//AbstractText'):
                label = abstract_elem.get('Label')
                text = abstract_elem.text

                if text:
                    if label:
                        section_text = f"{label}\n{text}"
                    else:
                        section_text = text

                    abstract_sections.append(section_text)

            abstract = '\n\n'.join(abstract_sections) if abstract_sections else None
            pub_date = extract_publication_date(article).toordinal()

            # Extract MeSH terms
            mesh_terms_list = []
            for mesh_heading in article.findall('.//MeshHeading'):
                descriptor_name = mesh_heading.find('DescriptorName')
                if descriptor_name is not None:
                    mesh_terms_list.append(descriptor_name.text)

            mesh_terms_combined = '\n'.join(mesh_terms_list) if mesh_terms_list else None

            pmids.append(pmid)
            titles.append(title)
            abstracts.append(abstract)
            pub_dates.append(pub_date)
            mesh_terms.append(mesh_terms_combined)

        for delete_citation in root.findall('.//DeleteCitation'):
            pmid_elem = delete_citation.find('.//PMID')
            pmid = int(pmid_elem.text) if pmid_elem is not None else None
            delete_pmids.append(pmid)

    return pmids, titles, abstracts, pub_dates, mesh_terms, delete_pmids


def extract_publication_date(article):
    """
    Extracts and formats the publication date from a PubMedArticle element.
    Attempts to format the date as yyyy-mm-dd, or yyyy-mm, or yyyy depending on available information.
    """
    pub_date_elem = article.find('.//PubDate')
    if pub_date_elem is None:
        return None

    year_elem = pub_date_elem.find('Year')
    medline_date_elem = pub_date_elem.find('MedlineDate')
    month_elem = pub_date_elem.find('Month')
    day_elem = pub_date_elem.find('Day')

    if year_elem is not None:
        year = year_elem.text.strip()
        month = month_elem.text.strip() if month_elem is not None else '01'
        day = day_elem.text.strip() if day_elem is not None else '01'

        try:
            return datetime.datetime.strptime(f'{year} {month} {day}', '%Y %b %d').date()
        except ValueError:
            try:
                return datetime.date(int(year), int(month), int(day))
            except ValueError:
                raise Exception(f"Cannot parse date: {year}, {month}, {day}")

    if medline_date_elem is not None:
        medline_date_text = medline_date_elem.text
        # Parse MedlineDate, which can have a format like "2002 Jan-Feb" or "2002 Spring"
        return parse_medline_date(medline_date_text)

    return None

def parse_medline_date(medline_date_text):
    """
    Parses a MedlineDate field which can be complex and returns a standardized date format.
    """
    # Example cases of MedlineDate formats:
    # "2002 Jan-Feb", "2002 Spring", "1998 Jul 23-30"
    try:
        if len(medline_date_text) == 4 and medline_date_text.isdigit():  # Simple year
            return datetime.date(int(medline_date_text), 1, 1)
        if " " in medline_date_text:
            # This can handle formats like "2002 Jan-Feb" or "1998 Jul 23-30"
            return datetime.datetime.strptime(medline_date_text.split()[0], '%Y').date()
        if medline_date_text[:4].isdigit():
            return datetime.date(int(medline_date_text[:4]), 1, 1)
    except ValueError:
        if medline_date_text[:4].isdigit():
            return datetime.date(int(medline_date_text[:4]), 1, 1)
        elif medline_date_text[-4:].isdigit():
            return datetime.date(int(medline_date_text[-4:]), 1, 1)
        else:
            raise Exception(f"Cannot parse Medline date: '{medline_date_text}'")

def extract_sequence_number(file_name):
    match = re.search(r'n(\d+)\.xml\.gz', file_name)
    if match:
        return int(match.group(1))
    return None

def process_xml_files():
    # Initialize DuckDB and create a table
    # con = duckdb.connect('PubMed.db')
    con = sqlite3.connect('PubMed.sqlite')
    con.execute('''
        CREATE TABLE IF NOT EXISTS pubmed_articles (
            pmid INTEGER PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            publication_date INTEGER,
            mesh_terms TEXT,
            file_number INTEGER
        )
    ''')

    file_list = sorted([f for f in os.listdir(XML_FOLDER) if f.endswith('.xml.gz')])

    for file_name in file_list:
        logging.info(f"Processing {file_name}")
        file_path = os.path.join(XML_FOLDER, file_name)
        file_number = extract_sequence_number(file_name)

        logging.info("- Parsing XML")
        pmids, titles, abstracts, pub_dates, mesh_terms, delete_pmids = parse_pubmed_xml(file_path)


        logging.info(f"- Inserting {len(pmids)} records into database")
        # total_elements = len(pmids) * 6  # 6 elements per record
        # flattened_data = [None] * total_elements
        # index = 0
        # for pmid, title, abstract, pub_date, mesh_term in zip(pmids, titles, abstracts, pub_dates, mesh_terms):
        #     flattened_data[index:index + 6] = [pmid, title, abstract, pub_date, mesh_term, file_number]
        #     index += 6
        #
        # placeholders = ', '.join(['(?, ?, ?, ?, ?, ?)'] * len(pmids))
        # query = f'''
        #         INSERT OR REPLACE INTO pubmed_articles (pmid, title, abstract, publication_date, mesh_terms, file_number)
        #         VALUES {placeholders}
        #     '''
        # con.execute(query, flattened_data)

        file_numbers = [file_number] * len(pmids)
        con.execute('BEGIN TRANSACTION;')
        con.executemany(f'''
            INSERT OR REPLACE INTO pubmed_articles (pmid, title, abstract, publication_date, mesh_terms, file_number)
            VALUES (?, ?, ?, ?, ?, ?)
        ''',
           list(zip(pmids, titles, abstracts, pub_dates, mesh_terms, file_numbers))
        )

        if len(delete_pmids) > 0:
            logging.info(f"- Deleting {len(delete_pmids)} records")
            placeholders = ', '.join(['?'] * len(delete_pmids))
            query = f'''
                DELETE FROM pubmed_articles
                WHERE pmid IN ({placeholders})
            '''
            con.execute(query, delete_pmids)
        con.commit()

    results = con.execute('SELECT COUNT(*) FROM pubmed_articles').fetchall()
    logging.info(f'Total articles inserted: {results[0][0]}')

    # Close the DuckDB connection
    con.close()

if __name__ == "__main__":
    process_xml_files()