class DeepDiveError(Exception):
    status_code: int = 500


class EmbeddingServiceError(DeepDiveError):
    status_code = 502


class DatabaseError(DeepDiveError):
    status_code = 500


class AnalysisError(DeepDiveError):
    status_code = 500
