"""PostgreSQL reference checks without guessing relationships from column names."""

import re

from phase2.validation_contract import ContractIssue


def sql_type(value):
    value = re.sub(r"\([^)]*\)", "", str(value).upper()).strip()
    return {"INT8": "BIGINT", "BIGSERIAL": "BIGINT", "INT4": "INTEGER", "INT": "INTEGER",
            "SERIAL": "INTEGER", "INT2": "SMALLINT", "SMALLSERIAL": "SMALLINT", "CHARACTER VARYING": "VARCHAR",
            "BOOL": "BOOLEAN", "TIMESTAMP WITH TIME ZONE": "TIMESTAMPTZ"}.get(value, value)


def foreign_keys(table):
    seen = set()
    for column in table.get("columns") or []:
        text = str(column.get("constraints") or "")
        match = re.search(r'REFERENCES\s+"?([a-z_][a-z0-9_]*)"?\s*\(([^)]+)\)', text, re.IGNORECASE)
        if match:
            target_columns = [name.strip(' "') for name in match.group(2).split(",")]
            delete = re.search(r"ON\s+DELETE\s+(SET\s+NULL|SET\s+DEFAULT|NO\s+ACTION|CASCADE|RESTRICT)", text, re.IGNORECASE)
            item = {"columns": [column["name"]], "references": {"table": match.group(1), "columns": target_columns},
                    "onDelete": delete.group(1).upper() if delete else "NO ACTION"}
            seen.add((tuple(item["columns"]), match.group(1), tuple(target_columns)))
            yield item
    for raw in table.get("foreignKeys") or []:
        if not isinstance(raw, dict):
            continue
        ref = raw.get("references") or {}
        columns = raw.get("columns") or [raw.get("column")]
        targets = ref.get("columns") or [ref.get("column") or "id"]
        key = (tuple(columns), ref.get("table"), tuple(targets))
        if key not in seen:
            seen.add(key)
            yield {**raw, "columns": columns, "references": {**ref, "columns": targets}}


def unique_keys(table):
    columns = table.get("columns") or []
    primary = [column["name"] for column in columns if re.search(r"PRIMARY[_\s]+KEY", str(column.get("constraints") or ""), re.IGNORECASE)]
    result = {tuple(primary)} if primary else set()
    result.update((column["name"],) for column in columns if re.search(r"\bUNIQUE\b", str(column.get("constraints") or ""), re.IGNORECASE))
    for definition in [*(table.get("indexes") or []), *(table.get("constraints") or [])]:
        if isinstance(definition, dict):
            if str(definition.get("type") or "").upper() in {"UNIQUE", "PRIMARY_KEY", "PRIMARY KEY"}:
                result.add(tuple(definition.get("columns") or []))
        elif re.search(r"\bUNIQUE\b|PRIMARY[_\s]+KEY", str(definition), re.IGNORECASE) and not re.search(r"\bWHERE\b", str(definition), re.IGNORECASE):
            match = re.search(r"\(([^)]+)\)", str(definition))
            if match:
                result.add(tuple(name.strip(' "') for name in match.group(1).split(",")))
    return result


def foreign_key_issues(db):
    tables = {table["name"]: table for table in db.get("tables") or []}
    issues = []
    for name, table in tables.items():
        columns = {column["name"]: column for column in table.get("columns") or []}
        for column in columns.values():
            constraints = str(column.get("constraints") or "")
            declared = any(column["name"] in (item.get("columns") or [item.get("column")]) for item in table.get("foreignKeys") or [])
            if "REFERENCES" in constraints.upper() and not re.search(r'REFERENCES\s+"?[a-z_][a-z0-9_]*"?\s*\([^)]+\)', constraints, re.IGNORECASE):
                issues.append(ContractIssue("DB_FOREIGN_KEY_REFERENCE_INVALID", "db", name, f"/columns/{column['name']}",
                                            f"{name}.{column['name']}: malformed reference").to_dict())
            if "FOREIGN_KEY" in constraints.upper() and "REFERENCES" not in constraints.upper() and not declared:
                issues.append(ContractIssue("DB_FOREIGN_KEY_REFERENCE_REQUIRED", "db", name, f"/columns/{column['name']}",
                                            f"{name}.{column['name']}: explicit reference required").to_dict())
        for fk in foreign_keys(table):
            source = fk["columns"]
            target_name = fk["references"].get("table")
            target_columns = fk["references"]["columns"]
            target = tables.get(target_name)
            pointer = "/columns/" + ",".join(str(value) for value in source)
            def add(code, detail, name=name, pointer=pointer, source=source):
                issues.append(ContractIssue(code, "db", name, pointer, f"{name}.{','.join(map(str, source))}: {detail}").to_dict())
            if target is None:
                add("DB_FOREIGN_KEY_TARGET_MISSING", f"target table {target_name} not found")
                continue
            targets = {column["name"]: column for column in target.get("columns") or []}
            if len(source) != len(target_columns) or any(value not in columns for value in source):
                add("DB_FOREIGN_KEY_SOURCE_INVALID", "source columns/arity invalid")
                continue
            if any(value not in targets for value in target_columns):
                add("DB_FOREIGN_KEY_COLUMN_MISSING", f"referenced columns {target_name}.{target_columns} not found")
                continue
            if not any(len(key) == len(target_columns) and set(key) == set(target_columns) for key in unique_keys(target)):
                add("DB_FOREIGN_KEY_UNIQUE_REQUIRED", f"referenced columns {target_name}.{target_columns} must be PK/UNIQUE")
            for left, right in zip(source, target_columns, strict=True):
                if sql_type(columns[left].get("type")) != sql_type(targets[right].get("type")):
                    add("DB_FOREIGN_KEY_TYPE_MISMATCH", f"{left} and {target_name}.{right} have incompatible types")
                action = str(fk.get("onDelete") or "NO ACTION").replace("_", " ").upper()
                if action not in {"NO ACTION", "RESTRICT", "CASCADE", "SET NULL", "SET DEFAULT"}:
                    add("DB_FOREIGN_KEY_DELETE_INVALID", action)
                if action == "SET NULL" and (columns[left].get("nullable") is False or re.search(r"NOT[_\s]+NULL|PRIMARY[_\s]+KEY", str(columns[left].get("constraints")), re.IGNORECASE)):
                    add("DB_FOREIGN_KEY_NULLABILITY_CONFLICT", f"SET NULL conflicts with {left}")
                constraint_text = re.sub(r"ON\s+DELETE\s+SET\s+DEFAULT", "", str(columns[left].get("constraints")), flags=re.IGNORECASE)
                not_null = columns[left].get("nullable") is False or bool(re.search(r"NOT[_\s]+NULL|PRIMARY[_\s]+KEY", constraint_text, re.IGNORECASE))
                if action == "SET DEFAULT" and not_null and columns[left].get("default") is None and not re.search(r"\bDEFAULT\b", constraint_text, re.IGNORECASE):
                    add("DB_FOREIGN_KEY_DEFAULT_REQUIRED", left)
    return issues
