import json
import inference as user_src

from potassium import (
    Potassium,
    Request,
    Response,
)

app = Potassium("server")

@app.init
def init():
    (model, model_runner) = user_src.init()

    return {
        "model": model,
        "model_runner": model_runner,
    }

@app.handler()
def handler(context: dict, request: Request) -> Response:
    model_inputs = request.json

    output = user_src.inference(model_inputs, context["model_runner"])

    return Response(
        # silly to re-load the JSON, but that's how it goes
        json=json.loads(output),
        status=200
    )

if __name__ == "__main__":
    app.serve()
