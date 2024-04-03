from datetime import date
import random
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shiny.types import ImgData

import app_functions


template = """
You are a pharmaceutical research & development professional performing the structured benefit-risk analysis for a new investigational drug.
Use only the provided pieces of retrieved context to perform the requested tasks. 

Tasks: 
1. Write the 'Analysis of Condition' section on the evidence and uncertainties for the nature of {indication}, focusing on the most important aspects of the condition that {product} may target, symptom burden, and natural course of the disease. What aspects of the condition have the most impact on the population with the condition?  Provide the most critical information on incidence, duration, disease progression, morbidity, symptoms, impact on patient functioning, mortality, health-related quality of life, and important differences in outcome or severity in subpopulations. For each point of information you provide, reference what document and page number this information was taken from.
2. Summarize in no more than 3 bullets the key conclusions of the information you just provided.
"""

app_ui = ui.page_fluid(
    ui.row(
        ui.column(11, ui.panel_title("RAG test app")),
        ui.column(1, ui.output_image("image", width="100px", height="97px"), align="right")
    ),
    ui.layout_column_wrap(
        ui.input_text("indication", "Indication:", "Early Onset Severe Hemolytic Disease of the Fetus and Newborn (EOS-HDFN)"),
        ui.input_text("product", "Product:", "nipocalimab")
    ),
    ui.input_text_area("template",
                       "Prompt template:",
                       template,
                       width="100%",
                       height="200px"),
    ui.input_numeric("k", "Number of text chunks to retrieve:", 3),
    ui.input_action_button("submit", "Submit"),
    ui.p("Instantiated prompt with context:"),
    ui.pre(ui.output_text("prompt_with_context"), style="white-space: pre-wrap; word-break: keep-all;"),
    ui.p("Response:"),
    ui.pre(ui.output_text("response"), style="white-space: pre-wrap; word-break: keep-all;"),
    ui.download_button("download_data", "Download")
)


def server(input: Inputs, output: Outputs, session: Session):

    value = reactive.value(app_functions.LlmResponse("", ""))

    @reactive.effect
    @reactive.event(input.submit)
    async def _():
        id = ui.notification_show("Processing...", duration=999, type="message", close_button=False)
        prompt = input.template().format(indication=input.indication(), product=input.product())
        value.set(await app_functions.get_llm_response(prompt, input.k()))
        ui.notification_remove(id)

    @render.text
    def response():
        return value.get().response

    @render.text
    def prompt_with_context():
        return value.get().prompt

    @render.download(
        filename=lambda: f"brPrompts-{date.today().isoformat()}-{random.randint(100, 999)}.txt"
    )
    async def download_data():
        yield "\nIndication:\n"
        yield input.indication() + "\n"
        yield "\nProduct:\n"
        yield input.product() + "\n"
        yield "Template:\n"
        yield input.template() + "\n"
        yield "\nNumber of text chunks to retrieve:\n"
        yield str(input.k()) + "\n"
        yield "\nPrompt with context:\n"
        yield value.get().prompt + "\n"
        yield "\nResponse:\n"
        yield value.get().response + "\n"

    @render.image
    def image():
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "logo.png"), "width": "100px"}
        return img


app = App(app_ui, server, debug=False)
