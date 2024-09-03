import gzip
import logging
import datetime
import re
import sys
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import os
import sqlite3
import yaml
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from PubMedXmlToSqliteSettings import PubMedXmlToSqliteSettings
from Logging import open_log


@dataclass
class Records:
    pmids: list = field(default_factory=list)
    titles: list = field(default_factory=list)
    abstracts: list = field(default_factory=list)
    publication_dates: list = field(default_factory=list)
    mesh_terms: list = field(default_factory=list)
    keywords: list = field(default_factory=list)
    chemicals: list = field(default_factory=list)
    authors: list = field(default_factory=list)
    journal_names: list = field(default_factory=list)
    years: list = field(default_factory=list)
    volumes: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    paginations: list = field(default_factory=list)
    publication_types: list = field(default_factory=list)
    delete_pmids: list = field(default_factory=list)


def parse_pubmed_xml(file_path: str) -> Records:
    """
    Extract information about all citations described in the XML file. This is a subset of the information in PubMed,
    and collapses everything to a single value. For example, all author names are combined into a single (EOL-delimited)
    string.

    :param file_path: The path to the xml.gz file.
    :return: An object of type Records.
    """
    records = Records()

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        tree = ElementTree.parse(f)
        root = tree.getroot()

        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            version = pmid_elem.get("Version")
            if version != "1":
                continue

            # PMID
            pmid = int(pmid_elem.text) if pmid_elem is not None else None
            records.pmids.append(pmid)

            # Title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else None
            records.titles.append(title)

            # Abstract
            abstract_sections = []
            for abstract_elem in article.findall(".//AbstractText"):
                label = abstract_elem.get("Label")
                text = abstract_elem.text

                if text:
                    # If the section has a header, include it in the text:
                    if label:
                        section_text = f"{label}\n{text}"
                    else:
                        section_text = text

                    abstract_sections.append(section_text)

            abstract = "\n\n".join(abstract_sections) if abstract_sections else None
            records.abstracts.append(abstract)

            # Publication date
            pub_date = extract_publication_date(article)
            records.publication_dates.append(pub_date.toordinal())

            # MeSH terms
            mesh_terms_list = []
            for mesh_heading in article.findall(".//MeshHeading"):
                descriptor_name = mesh_heading.find("DescriptorName")
                if descriptor_name is not None:
                    mesh_terms_list.append(descriptor_name.text)
            mesh_terms_combined = "\n".join(mesh_terms_list) if mesh_terms_list else None
            records.mesh_terms.append(mesh_terms_combined)

            # Keywords
            keyword_list = []
            for keyword in article.findall(".//Keyword"):
                keyword_text = keyword.text
                # Keywords don't use a controlled vocabulary, so could contain non-parseable garbage:
                if keyword_text is not None:
                    keyword_list.append(keyword_text)
            keywords_combined = "\n".join(keyword_list) if keyword_list else None
            records.keywords.append(keywords_combined)

            # Chemicals
            chemical_list = []
            for chemical in article.findall(".//NameOfSubstance"):
                chemical_list.append(chemical.text)
            chemicals_combined = "\n".join(chemical_list) if chemical_list else None
            records.chemicals.append(chemicals_combined)

            # Authors
            author_list = []
            for author in article.findall(".//Author"):
                author_list.append(parse_author(author))
            authors_combined = "\n".join(author_list) if author_list else None
            records.authors.append(authors_combined)

            # Journal name
            journal = article.find(".//Journal")
            journal_name_elem = journal.find(".//Title")
            journal_name = journal_name_elem.text if journal_name_elem is not None else None
            records.journal_names.append(journal_name)

            # Year
            year = pub_date.year
            records.years.append(year)

            # Volume
            volume_elem = journal.find(".//Volume")
            volume = volume_elem.text if volume_elem is not None else None
            records.volumes.append(volume)

            # Issue
            issue_elem = journal.find(".//Issue")
            issue = issue_elem.text if issue_elem is not None else None
            records.issues.append(issue)

            # Pagination
            pagination_elem = article.find(".//MedlinePgn")
            pagination = pagination_elem.text if pagination_elem is not None else None
            records.paginations.append(pagination)

            # Publication types
            publication_type_list = []
            for publication_type in article.findall(".//PublicationType"):
                publication_type_list.append(publication_type.text)
            publication_types_combined = "\n".join(publication_type_list) if publication_type_list else None
            records.publication_types.append(publication_types_combined)

        for delete_citation in root.findall(".//DeleteCitation"):
            pmid_elem = delete_citation.find(".//PMID")
            pmid = int(pmid_elem.text) if pmid_elem is not None else None
            records.delete_pmids.append(pmid)

    return records


def parse_author(author: Element) -> str:
    collective_name = author.find(".//CollectiveName")
    if collective_name is not None:
        return collective_name.text

    last_name = author.find(".//LastName")
    initials = author.find(".//Initials")
    if last_name is None:
        raise ValueError("Cannot parse author: LastName and CollectiveName are both missing")
    if initials is None:
        return last_name.text
    return f"{last_name.text}, {initials.text}"


def extract_publication_date(article: Element) -> datetime.date:
    """
    Extracts and formats the publication date from a PubMedArticle element.
    Attempts to format the date as yyyy-mm-dd, or yyyy-mm, or yyyy depending on available information.
    """
    pub_date_elem = article.find(".//PubDate")
    if pub_date_elem is None:
        pmid = article.find(".//PMID").text
        raise ValueError(f"Article with PMID {pmid} has no PubDate element")

    year, month, day = extract_date_components(pub_date_elem)

    if year:
        return parse_date(year, month, day)

    medline_date_elem = pub_date_elem.find("MedlineDate")
    if medline_date_elem is not None:
        return parse_medline_date(medline_date_elem.text)

    pmid = article.find(".//PMID").text
    raise ValueError(f"Unable to parse Pubdate for article with PMID {pmid}.")


def extract_date_components(pub_date_elem: Element) -> Tuple[Optional[str], str, str]:
    """
    Extracts the year, month, and day elements from a PubDate element.

    Returns:
        A tuple containing (year, month, day) where year is optional.
    """
    year_elem = pub_date_elem.find("Year")
    month_elem = pub_date_elem.find("Month")
    day_elem = pub_date_elem.find("Day")

    year = year_elem.text.strip() if year_elem is not None else None
    month = month_elem.text.strip() if month_elem is not None else "01"
    day = day_elem.text.strip() if day_elem is not None else "01"

    return year, month, day


def parse_date(year: str, month: str, day: str) -> datetime.date:
    """
    Attempts to parse the date given the year, month, and day.

    Returns:
        A datetime.date object representing the parsed date.

    Raises:
        ValueError: If the date cannot be parsed.
    """
    try:
        return datetime.datetime.strptime(f"{year} {month} {day}", "%Y %b %d").date()
    except ValueError:
        try:
            return datetime.date(int(year), int(month), int(day))
        except ValueError:
            raise ValueError(f"Cannot parse date: {year}, {month}, {day}")


def parse_medline_date(medline_date_text: str) -> datetime.date:
    """
    Parses a MedlineDate field which can be complex and returns a standardized date format.

    Returns:
        A datetime.date object representing the parsed Medline date.

    Raises:
        ValueError: If the Medline date cannot be parsed.
    """
    if len(medline_date_text) == 4 and medline_date_text.isdigit():
        return datetime.date(int(medline_date_text), 1, 1)

    if " " in medline_date_text:
        year = medline_date_text.split()[0]
        if year.isdigit():
            return datetime.date(int(year), 1, 1)

    try:
        if medline_date_text[:4].isdigit():
            return datetime.date(int(medline_date_text[:4]), 1, 1)
    except ValueError:
        pass

    if medline_date_text[:4].isdigit():
        return datetime.date(int(medline_date_text[:4]), 1, 1)

    if medline_date_text[-4:].isdigit():
        return datetime.date(int(medline_date_text[-4:]), 1, 1)

    raise ValueError(f"Cannot parse Medline date: {medline_date_text}")


def extract_sequence_number(file_name):
    match = re.search(r"n(\d+)\.xml\.gz", file_name)
    if match:
        return int(match.group(1))
    return None


def main(args: List[str]):
    with open(args[0]) as file:
        config = yaml.safe_load(file)
    settings = PubMedXmlToSqliteSettings(config)
    open_log(settings.log_path)

    con = sqlite3.connect(settings.sqlite_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS pubmed_articles (
            pmid INTEGER PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            publication_date INTEGER,
            mesh_terms TEXT,
            keywords TEXT,
            chemicals TEXT,
            authors TEXT,
            journal_name TEXT,
            year INTEGER,
            volume TEXT,
            issue TEXT,
            pagination TEXT,
            publication_types TEXT,
            file_number INTEGER
        )
    """)

    file_list = sorted([f for f in os.listdir(settings.xml_folder) if f.endswith(".xml.gz")])

    for file_name in file_list:
        logging.info(f"Processing {file_name}")
        file_path = os.path.join(settings.xml_folder, file_name)
        file_number = extract_sequence_number(file_name)

        logging.info("- Parsing XML")
        records = parse_pubmed_xml(file_path)

        logging.info(f"- Inserting {len(records.pmids)} records into database")
        file_numbers = [file_number] * len(records.pmids)
        con.execute("BEGIN TRANSACTION;")
        con.executemany(f"""
            INSERT OR REPLACE INTO pubmed_articles (
                pmid, 
                title, 
                abstract, 
                publication_date, 
                mesh_terms, 
                keywords,
                chemicals,
                authors,
                journal_name,
                year,
                volume,
                issue,
                pagination,
                publication_types,
                file_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
                        list(zip(records.pmids,
                                 records.titles,
                                 records.abstracts,
                                 records.publication_dates,
                                 records.mesh_terms,
                                 records.keywords,
                                 records.chemicals,
                                 records.authors,
                                 records.journal_names,
                                 records.years,
                                 records.volumes,
                                 records.issues,
                                 records.paginations,
                                 records.publication_types,
                                 file_numbers)))

        if len(records.delete_pmids) > 0:
            logging.info(f"- Deleting {len(records.delete_pmids)} records")
            placeholders = ", ".join(["?"] * len(records.delete_pmids))
            query = f"""
                DELETE FROM pubmed_articles
                WHERE pmid IN ({placeholders})
            """
            con.execute(query, records.delete_pmids)
        con.commit()

    results = con.execute("SELECT COUNT(*) FROM pubmed_articles").fetchall()
    logging.info(f"Total articles inserted: {results[0][0]}")

    con.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to yaml file as argument")
    else:
        main(sys.argv[1:])
