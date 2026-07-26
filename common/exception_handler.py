import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api_response import ErrorCode, error

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning("[HTTP ERROR %d] %s", exc.status_code, exc.detail)
        code = ErrorCode.INVALID_INPUT if exc.status_code < 500 else ErrorCode.INTERNAL_ERROR
        return JSONResponse(
            status_code=exc.status_code,
            content=error(code, str(exc.detail)),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        first_msg = (
            exc.errors()[0].get("msg", error_message_default(ErrorCode.INVALID_INPUT))
            if exc.errors()
            else error_message_default(ErrorCode.INVALID_INPUT)
        )
        logger.warning("[VALIDATION FAILED] %s", first_msg)
        return JSONResponse(
            status_code=400,
            content=error(ErrorCode.INVALID_INPUT, first_msg),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("[BAD REQUEST] %s", exc)
        return JSONResponse(
            status_code=400,
            content=error(ErrorCode.INVALID_INPUT, str(exc)),
        )

    @app.exception_handler(TimeoutError)
    async def timeout_handler(request: Request, exc: TimeoutError):
        logger.error("[TIMEOUT] %s", exc)
        return JSONResponse(
            status_code=504,
            content=error(ErrorCode.PIPELINE_TIMEOUT),
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        logger.error("[UNHANDLED ERROR] %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content=error(ErrorCode.INTERNAL_ERROR),
        )


def error_message_default(code: ErrorCode) -> str:
    from .api_response import error_message
    return error_message(code)
