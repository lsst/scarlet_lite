from __future__ import annotations

from typing import ClassVar, Callable

__all__ = ["MigrationError", "MigrationRegistry", "migration"]

Migrator = Callable[[dict], dict]

PRE_SCHEMA = "0.0.0"


class MigrationError(Exception):
    """Custom error for migration issues."""
    pass


class MigrationRegistry:
    """Manages migration of data between different schema versions."""
    registry: ClassVar[dict[tuple[str, str], Migrator]] = {}
    current: ClassVar[dict[str, str]] = {}

    @staticmethod
    def register(type_name: str, from_version: str, migrator: Migrator) -> None:
        """Register a migration function from one version to another."""
        MigrationRegistry.registry[(type_name, from_version)] = migrator

    @staticmethod
    def set_current(data_type: str, version: str) -> None:
        """Set the current version for a given data type."""
        MigrationRegistry.current[data_type] = version

    @staticmethod
    def migrate(data_type: str, data: dict) -> dict:
        """Migrate data to the current schema version.

        Parameters
        ----------
        data :
            The data to migrate. Must contain 'type' and 'version' keys.

        Returns
        -------
        result :
            The migrated data.
        """
        if "version" not in data:
            # Unversioned data is pre-schema and is considered to be
            # version "0.0.0" for backwards compatibility.
            data["version"] = PRE_SCHEMA

        from_version = data["version"]

        if data_type not in MigrationRegistry.current:
            raise ValueError(f"No current version set for data type '{data_type}'.")

        to_version = MigrationRegistry.current[data_type]

        # Keep track of seen versions to avoid infinite loops
        seen = set()

        while from_version != to_version:
            key = (data_type, from_version)
            if key not in MigrationRegistry.registry or key in seen:
                raise MigrationError(
                    f"No migration path from version '{from_version}' for type '{data_type}'."
                )

            migrator = MigrationRegistry.registry[key]
            data = migrator(data)
            from_version = data["version"]

        return data


def migration(type_name: str, from_version: str) -> Migrator:
    """Decorator to register a migration step.

    Parameters
    ----------
    type_name :
        The type of data being migrated.
    from_version :
        The version the migrator converts from.

    Returns
    -------
    result :
        The decorator that registers the migration function.
    """
    def decorator(func: Migrator) -> Migrator:
        MigrationRegistry.register(type_name, from_version, func)
        return func
    return decorator
