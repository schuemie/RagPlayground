import random
from datetime import date, datetime
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shiny.types import ImgData

import app_functions

template = """
What is the primary treatment option for {indication}?
"""

version = "0.2.0"

app_ui = ui.page_fluid(
    ui.row(
        ui.column(11, ui.panel_title("RAG test app")),
        ui.column(1, ui.output_image("image", width="100px", height="97px"), align="right")
    ),
    ui.layout_column_wrap(
        ui.input_text("indication", "Indication:", "Multiple myeloma"),
        ui.input_text("product", "Product:", "nipocalimab"),
    ),
    ui.input_text_area("search_template",
                       "Search template:",
                       template,
                       width="100%",
                       height="200px"),
    ui.input_numeric("k", "Number of text chunks to retrieve:", 3),
    ui.input_action_button("search", "Search"),
    ui.p("Search results:"),
    ui.pre(ui.output_text("search_results_output"), style="white-space: pre-wrap; word-break: keep-all;"),
    ui.input_text_area("prompt_template",
                       "Prompt template:",
                       "",
                       width="100%",
                       height="200px"),
    ui.input_action_button("copy", "Copy search template to prompt template", inline=True),
    ui.input_action_button("submit", "Submit", inline=True),
    ui.p("Response:"),
    ui.pre(ui.output_text("llm_result_output"), style="white-space: pre-wrap; word-break: keep-all;"),
    ui.download_button("download_data", "Download")
)


def server(input: Inputs, output: Outputs, session: Session):

    llm_result = reactive.value("")

    search_results = reactive.value([])

    @reactive.effect
    @reactive.event(input.copy)
    def _():
        ui.update_text_area("prompt_template", value=input.search_template())

    def search():
        prompt = input.search_template().format(indication=input.indication(), product=input.product())
        search_results.set(app_functions.get_search_results(prompt, input.k()))

    @reactive.effect
    @reactive.event(input.search)
    def _():
        search()

    def convert_search_results_to_text() -> str:
        texts = []
        for doc, score in search_results.get():
            texts.append(f"Score: {score}, Metadata: {doc.metadata}\n--------------\n{doc.page_content}\n--------------\n")
        return "\n".join(texts)

    @render.text
    def search_results_output():
        return convert_search_results_to_text()

    @reactive.effect
    @reactive.event(input.submit)
    async def _():
        id = ui.notification_show("Processing...", duration=999, type="message", close_button=False)
        prompt = input.prompt_template().format(indication=input.indication(), product=input.product())
        llm_result.set(await app_functions.get_llm_response(prompt, search_results.get()))
        ui.notification_remove(id)

    @render.text
    def llm_result_output():
        return llm_result.get()

    @render.download(
        filename=lambda: f"brPrompts-{date.today().isoformat()}-{random.randint(100, 999)}.txt"
    )
    async def download_data():
        date_time = datetime.now() .strftime("%Y-%m-%d, %H:%M:%S")
        yield "Timestamp: " + date_time + "\n"
        yield "App version: " + version + "\n\n"
        yield "Indication: " + input.indication() + "\n"
        yield "Product: " + input.product() + "\n\n"
        yield "Search template:\n"
        yield input.search_template().strip() + "\n\n"
        yield "\nNumber of text chunks to retrieve:" + str(input.k()) + "\n\n"
        yield "Search results:\n"
        yield convert_search_results_to_text() + "\n\n"
        yield "Prompt template:\n"
        yield input.prompt_template().strip() + "\n\n"
        yield "LLM response:\n"
        yield llm_result.get() + "\n"

    @render.image
    def image():
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "logo.png"), "width": "100px"}
        return img


app = App(app_ui, server, debug=False)
