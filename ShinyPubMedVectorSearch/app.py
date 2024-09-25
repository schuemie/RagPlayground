import os

from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from sentence_transformers import SentenceTransformer
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5", trust_remote_code=True)
conn = psycopg.connect(host=os.getenv("ECP_RDS_HOST"),
                       user=os.getenv("ECP_RDS_USER"),
                       password=os.getenv("ECP_RDS_PASSWORD"),
                       dbname=os.getenv("ECP_RDS_DBNAME"))
register_vector(conn)
conn.execute("SET hnsw.ef_search = 25")

app_ui = ui.page_fluid(
    ui.panel_title("PubMed Vector Search"),
    ui.row(
        ui.input_text_area("query",
                           "Query:",
                           placeholder="Enter your search text here. Full natural language sentences are best",
                           width="100%",
                           height="100px"),
        ui.column(2, ui.input_action_button("search", "Search"))
    ),
    ui.output_ui("search_results_output")
 )


def server(input: Inputs, output: Outputs, session: Session):
    search_results = reactive.value([])

    @reactive.effect
    @reactive.event(input.search)
    def _():
        query = input.query()
        query_embedding = embedding_model.encode(query, prompt_name="query").tolist()
        sql = """
                SELECT embedding <=> %s AS similarity,
                    pubmed_articles.pmid,
                    title,
                    authors,
                    journal_name,
                    year,
                    volume,
                    issue,
                    pagination
                FROM public.vectors_snowflake_arctic_m_partitioned vectors
                INNER JOIN public.pubmed_articles
                    ON pubmed_articles.pmid = vectors.pmid
                ORDER BY embedding <=> %s
                LIMIT 25;
                """
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        result = conn.execute(sql, (embedding_str, embedding_str))
        similar_rows = result.fetchall()
        search_results.set([{"similarity" : row[0],
                             "pmid": row[1],
                             "title":row[2],
                             "authors": row[3],
                             "journal": row[4],
                             "year": row[5],
                             "volume": row[6],
                             "issue": row[7],
                             "pagination": row[8]} for row in similar_rows])

    @output
    @render.ui
    def search_results_output():
        if not search_results:
            return ui.p("No results found.")

        return ui.tags.ul(
            *[
                ui.tags.div(
                    ui.h5(
                        ui.a(
                            f"{article['title']}",
                            href=f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/",
                            target="_blank"
                        )
                    ),
                    ui.p(f"Authors: {article['authors']}"),
                    ui.p(
                        f"{article['journal']}, Vol. {article['volume']} (Issue {article['issue']}), "
                        f"pp. {article['pagination']}, {article['year']}, PMID: {article['pmid']}"
                    ),
                    ui.p(f"Semantic distance: {article['similarity']:.3f}"),
                    ui.hr()
                )
                for article in search_results.get()
            ]
        )



app = App(app_ui, server, debug=False)
