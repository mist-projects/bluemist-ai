# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.3
# Email: dew@bluemist-ai.one
# Created: Aug 27, 2023
# Last modified: Oct 25, 2023

import nest_asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Union, Optional

from pyngrok import ngrok

from bluemist.llm.wrapper import perform_task

app = FastAPI()


# Define a Pydantic model for the input data
class InputDataModel(BaseModel):
    input_data: Union[str, List[str]]


# Endpoint for performing a task
@app.post("/perform_task/")
async def perform_nlp_task(
        request: Request,
        task_name: str,
        input_data: InputDataModel,
        question: Optional[str] = None,
        override_models: Optional[List[str]] = None,
        limit: int = 5,
        evaluate_models: bool = True
):
    try:
        results_df = perform_task(task_name, input_data.input_data, question=question, override_models=override_models,
                                  limit=limit, evaluate_models=evaluate_models)
        return JSONResponse(content=results_df.to_dict(orient="records"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def start_api_server(host='localhost', port=8000):
    """
    Starts the FastAPI server.

    Args:
        host : str, default='localhost'
            The host IP address. Defaults to 'localhost'.
        port : number, default=8000
            The port number. Defaults to 8000.
    """
    ngrok_tunnel = ngrok.connect(port)
    ngrok_tunnel
    nest_asyncio.apply()
    uvicorn.run(app, host=host, port=port)
