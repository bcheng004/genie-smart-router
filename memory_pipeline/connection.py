"""Pure-Python Postgres transport for the serverless pipeline sandbox."""

from contextlib import contextmanager
import ssl


class PipelineConnection:
    def __init__(self, connection):
        self.connection = connection

    def execute(self, query: str, params=()):
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return cursor

    @contextmanager
    def transaction(self):
        self.execute("BEGIN")
        try:
            yield
        except BaseException:
            self.execute("ROLLBACK")
            raise
        else:
            self.execute("COMMIT")


@contextmanager
def connect(host: str, database: str, user: str, password: str):
    from importlib.metadata import version
    import pg8000.dbapi

    try:
        connection = pg8000.dbapi.connect(host=host, database=database, user=user,
                                         password=password, ssl_context=ssl.create_default_context(),
                                         timeout=15)
    except AttributeError as error:
        raise RuntimeError(f"Postgres dependency mismatch: pg8000={version('pg8000')}, "
                           f"scramp={version('scramp')}") from error
    connection.autocommit = True
    try:
        yield PipelineConnection(connection)
    finally:
        connection.close()
