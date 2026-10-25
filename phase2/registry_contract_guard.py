"""Cross-feature contracts checked before either physical generator runs."""

from copy import deepcopy

from phase2.contract_quality import database_checks
from phase2.fk_validation import foreign_key_issues, sql_type
from phase2.operation_contract import operation_schema_issues, session_policy_issues
from phase2.workflow_contract import (
    workflow_check_constraints,
    workflow_contract_issues,
)


def contract_artifacts(features):
    """Project declared fields and operations without inferring physical resources."""
    tables, endpoints = {}, []
    for feature in features:
        contract = feature.get("dbContract") or {}
        for name in contract.get("tables") or []:
            tables.setdefault(name, {"name": name, "columns": [], "indexes": [], "foreignKeys": []})
        for field in contract.get("fields") or []:
            name = field.get("table") or contract.get("primaryTable")
            if name not in tables:
                continue
            tokens = ["PRIMARY_KEY" if field.get("primaryKey") else "",
                      "NOT_NULL" if field.get("required") or field.get("primaryKey") else "NULL",
                      "UNIQUE" if field.get("unique") else ""]
            value = field.get("default")
            if value is not None:
                if isinstance(value, str):
                    literal = value if value.upper() in {"NOW()", "CURRENT_DATE", "CURRENT_TIMESTAMP", "GEN_RANDOM_UUID()", "UUID_GENERATE_V4()"} else "'" + value.replace("'", "''") + "'"
                else:
                    literal = str(value).upper() if isinstance(value, bool) else str(value)
                tokens.append("DEFAULT " + literal)
            column = {**deepcopy(field), "type": str(field.get("type") or "").upper(), "constraints": " ".join(filter(None, tokens))}
            columns = tables[name]["columns"]
            previous = next((index for index, item in enumerate(columns) if item["name"] == field["name"]), None)
            if previous is None:
                columns.append(column)
            else:
                columns[previous] = column
        for fk in contract.get("foreignKeys") or []:
            source = fk.get("table") or contract.get("primaryTable")
            if source not in tables or fk in tables[source]["foreignKeys"]:
                continue
            tables[source]["foreignKeys"].append(deepcopy(fk))
        for unique in contract.get("uniqueConstraints") or []:
            name = unique.get("table") or contract.get("primaryTable")
            columns = unique.get("columns") or []
            if name in tables and columns:
                definition = f"CREATE UNIQUE INDEX uq_{name}_{'_'.join(columns)} ON {name}({','.join(columns)})"
                if definition not in tables[name]["indexes"]:
                    tables[name]["indexes"].append(definition)
        endpoints.extend({"primaryTable": contract.get("primaryTable"), **deepcopy(operation), "featureId": feature["featureId"]}
                         for operation in feature.get("apiContract") or [])
    # Shared fields are merged before references are rendered; later owners of
    # the same field must not erase an earlier REFERENCES constraint.
    for table in tables.values():
        for fk in table["foreignKeys"]:
            reference = fk.get("references") or {}
            if fk.get("column") and reference.get("column"):
                column = next((item for item in table["columns"] if item["name"] == fk["column"]), None)
                if column is not None:
                    column["constraints"] += f" FOREIGN_KEY REFERENCES {reference['table']}({reference['column']})"
                    if fk.get("onDelete"):
                        column["constraints"] += " ON DELETE " + fk["onDelete"]
    for name, constraints in workflow_check_constraints(features).items():
        if name in tables:
            tables[name]["constraints"] = constraints
    for name, constraints in database_checks(features).items():
        if name in tables:
            tables[name]["constraints"] = list(dict.fromkeys([*tables[name].get("constraints", []), *constraints]))
    return {"tables": list(tables.values())}, {"endpoints": endpoints}


def ownership_contract_issues(features):
    issues = []
    for index, feature in enumerate(features):
        ownership = feature.get("ownership") or {}
        if ownership.get("scope") not in {"USER", "SHARED"}:
            continue
        contract = feature.get("dbContract") or {}
        if not contract.get("tables"):
            continue
        entity, key = ownership.get("ownerEntity"), ownership.get("ownerKey")
        primary = contract.get("primaryTable")
        if entity not in contract["tables"]:
            issues.append(f"FEATURE_OWNER_TABLE_UNKNOWN: /features/{index}/ownership: ownerEntity must be an exact declared table name")
        elif primary != entity and not any(
            fk.get("table", primary) == primary and (fk.get("column") == key or key in (fk.get("columns") or []))
            and (fk.get("references") or {}).get("table") == entity
            for fk in contract.get("foreignKeys") or []
        ):
            issues.append(f"FEATURE_OWNER_REFERENCE_MISSING: /features/{index}/ownership: declared primary table FK must link ownerKey to ownerEntity")
    return issues


def shared_db_issues(features):
    tables, owners, declarations, versions = {}, {}, {}, {}
    issues = []
    for index, feature in enumerate(features):
        contract = feature.get("dbContract") or {}
        for name in contract.get("tables") or []:
            if isinstance(name, str):
                tables.setdefault(name, {"name": name, "columns": [], "foreignKeys": [], "indexes": []})
        for field in contract.get("fields") or []:
            if not isinstance(field, dict):
                continue
            key = (field.get("table") or contract.get("primaryTable"), field.get("name"))
            if not all(isinstance(value, str) and value for value in key):
                issues.append(f"FEATURE_DB_FIELD_SHAPE: /features/{index}/dbContract/fields: table/name strings required")
                continue
            owners.setdefault(key, set()).add(index)
            versions.setdefault(key, []).append(field)
            previous = declarations.get(key)
            declarations[key] = {**(previous or {}), **field}
    # Report every participant, not just the first adjacent conflicting pair.
    # Otherwise one scoped repair leaves later copies of the same conflict behind.
    for key, fields in versions.items():
        conflict = False
        for part in ("type", "required", "default", "primaryKey"):
            values = [sql_type(field[part]) if part == "type" else field[part] for field in fields if part in field]
            conflict = conflict or any(value != values[0] for value in values[1:])
        if conflict:
            for owner in sorted(owners[key]):
                issues.append(f"FEATURE_DB_FIELD_CONFLICT: /features/{owner}/dbContract/fields: {key[0]}.{key[1]}")
    for (table_name, name), field in declarations.items():
        if table_name not in tables or not name:
            continue
        primary = field.get("primaryKey") or (name == "id" and field.get("generated"))
        tokens = ["PRIMARY_KEY" if primary else "", "UNIQUE" if field.get("unique") else "",
                  "NOT_NULL" if field.get("required") else ""]
        if field.get("default") is not None:
            tokens.append("DEFAULT")
        column = {**field, "name": name, "type": field.get("type", ""), "constraints": " ".join(tokens).strip()}
        tables[table_name]["columns"].append(column)
    for index, feature in enumerate(features):
        contract = feature.get("dbContract") or {}
        for unique in contract.get("uniqueConstraints") or []:
            if isinstance(unique, dict) and isinstance(unique.get("table"), str) and unique["table"] in tables:
                tables[unique["table"]]["indexes"].append({"type": "UNIQUE", "columns": unique.get("columns") or []})
    for index, feature in enumerate(features):
        contract = feature.get("dbContract") or {}
        for operation_index, operation in enumerate(feature.get("apiContract") or []):
            scoped_operation = {"primaryTable": contract.get("primaryTable"), **operation}
            for error in operation_schema_issues(scoped_operation, tables):
                issues.append(f"{error}: /features/{index}/apiContract/{operation_index}")
        for fk_index, raw in enumerate(contract.get("foreignKeys") or []):
            if not isinstance(raw, dict):
                issues.append(f"FEATURE_DB_REFERENCE_SHAPE: /features/{index}/dbContract/foreignKeys/{fk_index}: object required")
                continue
            source = raw.get("table") or contract.get("primaryTable")
            if not isinstance(source, str) or source not in tables:
                issues.append(f"FEATURE_DB_REFERENCE_SOURCE: /features/{index}/dbContract/foreignKeys/{fk_index}: source table missing")
                continue
            # Check each reference separately to retain its feature/patch scope.
            schema = deepcopy(list(tables.values()))
            next(table for table in schema if table["name"] == source)["foreignKeys"] = [raw]
            for error in foreign_key_issues({"tables": schema}):
                issues.append(f"{error['code']}: /features/{index}/dbContract/foreignKeys/{fk_index}: {error['message']}")
    return list(dict.fromkeys(issues))


def registry_contract_issues(features):
    return [*shared_db_issues(features), *session_policy_issues(features), *workflow_contract_issues(features)]
