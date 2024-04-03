from shiny import App, Inputs, Outputs, Session, render, ui, reactive
import app_functions

app_ui = ui.page_fluid(
    ui.panel_title("RAG test app"),
    ui.input_text_area("prompt", "Prompt:", "What is the primary treatment option for multiple myeloma?"),
    ui.input_action_button("submit", "Submit"),
    ui.p("Response:"),
    ui.pre(ui.output_text("response"), style="white-space: pre-wrap; word-break: keep-all;"),
    ui.p("Prompt with context:"),
    ui.pre(ui.output_text("prompt_with_context"), style="white-space: pre-wrap; word-break: keep-all;"),
)


def server(input: Inputs, output: Outputs, session: Session):

    value = reactive.value(app_functions.LlmResponse("", ""))

    @reactive.effect
    @reactive.event(input.submit)
    async def _():
        id = ui.notification_show("Processing...", duration=999, type="message", close_button=False)
        value.set(await app_functions.get_llm_response(input.prompt()))
        ui.notification_remove(id)

    @render.text
    def response():
        return value.get().response

    @render.text
    def prompt_with_context():
        return value.get().prompt


app = App(app_ui, server, debug=False)
