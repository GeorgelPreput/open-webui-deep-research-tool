from pydantic import BaseModel


class QueryCollectionResponse(BaseModel):
    distances: list[list[float]]
    documents: list[list[str]]
    metadatas: list[list[dict]]


class ProcessFileResponse(BaseModel):
    status: bool
    collection_name: str | None = None
    filename: str | None = None
    content: str


class ProcessWebResponse(BaseModel):
    status: bool
    content: str


class FileUploadResponse(BaseModel):
    id: str
    filename: str
    meta: dict
    data: dict


class KBResponse(BaseModel):
    id: str
    name: str
    description: str


class ModelInfo(BaseModel):
    id: str
    name: str
    context_window: int | None = None
    meta: dict | None = None
