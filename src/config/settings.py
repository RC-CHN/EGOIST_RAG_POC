from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import PostgresDsn, RedisDsn

class Neo4jSettings(BaseSettings):
    """Neo4j connection settings."""
    model_config = SettingsConfigDict(env_prefix='NEO4J_')

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"

class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings."""
    model_config = SettingsConfigDict(env_prefix='POSTGRES_')

    user: str = "user"
    password: str = "password"
    host: str = "localhost"
    port: int = 5432
    db: str = "graphrag_meta"

    @property
    def dsn(self) -> str:
        return str(PostgresDsn.build(
            scheme="postgresql",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            path=self.db,
        ))

class RedisSettings(BaseSettings):
    """Redis connection settings."""
    model_config = SettingsConfigDict(env_prefix='REDIS_')

    host: str = "localhost"
    port: int = 6379
    db: int = 0

    @property
    def dsn(self) -> str:
        return str(RedisDsn.build(
            scheme="redis",
            host=self.host,
            port=self.port,
            path=f"/{self.db}"
        ))

class MinioSettings(BaseSettings):
    """MinIO connection settings."""
    model_config = SettingsConfigDict(env_prefix='MINIO_')

    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    secure: bool = False

class Settings(BaseSettings):
    """Main application settings."""
    neo4j: Neo4jSettings = Neo4jSettings()
    postgres: PostgresSettings = PostgresSettings()
    redis: RedisSettings = RedisSettings()
    minio: MinioSettings = MinioSettings()

# The single instance of settings to be used throughout the application
settings = Settings()