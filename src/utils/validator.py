import os
import yaml
import logging
from jsonschema import validate, ValidationError
from jsonschema.validators import validator_for
from jsonschema import RefResolver
from src.utils.logger import get_logger
from typing import Any, Dict, List, Tuple

# TODO удалить, решили использовать pydantic
class APISchemaValidator:
    """
    Класс-обертка для валидации данных против OpenAPI (Swagger) файла.
    Использует jsonschema RefResolver для разрешения ссылок ($ref) внутри yaml.
    """
    def __init__(self, spec_path: str, logger: logging.Logger) -> None:
        self.logger = logger

        if not os.path.exists(spec_path):
            self.logger.error(f"OpenAPI spec file not found at: {spec_path}")
            raise FileNotFoundError(f"OpenAPI spec file not found at: {spec_path}")

        with open(spec_path, 'r', encoding='utf-8') as f:
            self.spec = yaml.safe_load(f)

        self.resolver = RefResolver.from_schema(self.spec)

    def validate_data(self, data: Dict[str, Any], schema_name: str) -> None:
        """
        Валидирует словарь data против схемы с именем schema_name из components/schemas.
        """
        schema = self.spec.get('components', {}).get('schemas', {}).get(schema_name)

        if not schema:
            self.logger.error(f"Schema definition '{schema_name}' not found.")
            raise ValueError(f"Schema definition '{schema_name}' not found in OpenAPI spec.")

        try:
            validate(instance=data, schema=schema, resolver=self.resolver)
            self.logger.debug(f"Validation successful against schema '{schema_name}'")
        except ValidationError as e:
            error_path = " -> ".join([str(p) for p in e.path])
            self.logger.warning(f"Validation failed for '{schema_name}': {e.message} (path: {error_path})")
            raise ValueError(f"Validation Failed for '{schema_name}': {e.message} (at {error_path})")
