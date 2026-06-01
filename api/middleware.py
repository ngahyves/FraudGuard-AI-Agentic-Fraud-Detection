# src/api/middleware.py

from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

async def error_handler(request: Request, call_next):
    try:
        response = await call_next(request)
        return response

    except Exception as e:
        error_trace = traceback.format_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(e),
                "trace": error_trace
            }
        )
