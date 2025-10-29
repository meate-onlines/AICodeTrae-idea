#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import inspect
from sqlalchemy.engine import Engine, make_url, create_engine
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    default: Optional[str]
    autoincrement: Optional[bool] = None


@dataclass(frozen=True)
class ForeignKeyInfo:
    constrained_columns: Tuple[str, ...]
    referred_schema: Optional[str]
    referred_table: str
    referred_columns: Tuple[str, ...]
    name: Optional[str] = None


@dataclass(frozen=True)
class IndexInfo:
    name: Optional[str]
    columns: Tuple[str, ...]
    unique: bool
    dialect: Optional[str] = None
    is_expression: bool = False


@dataclass
class TableInfo:
    exists: bool
    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    primary_key: Tuple[str, ...] = field(default_factory=tuple)
    foreign_keys: Tuple[ForeignKeyInfo, ...] = field(default_factory=tuple)
    indexes: Tuple[IndexInfo, ...] = field(default_factory=tuple)


def canonicalize_type(tp: Any) -> str:
    try:
        return str(tp).strip().lower()
    except Exception:
        return repr(tp).strip().lower()


def canonicalize_default(def_val: Any) -> Optional[str]:
    if def_val is None:
        return None
    try:
        text = str(def_val).strip()
        # normalize common noise
        return " ".join(text.split()).lower()
    except Exception:
        return None


def derive_label_from_uri(db_uri: str) -> str:
    url = make_url(db_uri)
    backend = url.get_backend_name() or "db"
    host = url.host or "localhost"
    database = url.database or ""
    return f"{backend}://{host}/{database}"


def build_engine(db_uri: str) -> Engine:
    return create_engine(db_uri, future=True, pool_pre_ping=True)


def fetch_table_info(engine: Engine, table_name: str, schema: Optional[str]) -> TableInfo:
    inspector = inspect(engine)
    try:
        # columns
        raw_columns = inspector.get_columns(table_name, schema=schema)
    except NoSuchTableError:
        return TableInfo(exists=False)
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Failed to inspect table '{table_name}': {exc}") from exc

    columns: Dict[str, ColumnInfo] = {}
    for col in raw_columns:
        name = col.get("name")
        col_type = canonicalize_type(col.get("type"))
        nullable = bool(col.get("nullable", True))
        default = canonicalize_default(col.get("default") or col.get("server_default"))
        autoincrement = col.get("autoincrement")
        columns[name] = ColumnInfo(
            name=name,
            type=col_type,
            nullable=nullable,
            default=default,
            autoincrement=autoincrement if autoincrement in (True, False) else None,
        )

    # primary key
    pk = inspector.get_pk_constraint(table_name, schema=schema) or {}
    pk_cols = tuple(pk.get("constrained_columns") or [])

    # foreign keys
    fk_list: List[ForeignKeyInfo] = []
    for fk in inspector.get_foreign_keys(table_name, schema=schema) or []:
        fk_list.append(
            ForeignKeyInfo(
                constrained_columns=tuple(fk.get("constrained_columns") or ()),
                referred_schema=fk.get("referred_schema"),
                referred_table=fk.get("referred_table"),
                referred_columns=tuple(fk.get("referred_columns") or ()),
                name=fk.get("name"),
            )
        )

    # indexes
    indexes: List[IndexInfo] = []
    backend = engine.url.get_backend_name() if hasattr(engine, "url") else None
    for idx in inspector.get_indexes(table_name, schema=schema) or []:
        col_names = idx.get("column_names")
        if col_names is None:
            # expression-based or unknown; use name as hint
            expr_cols = tuple((idx.get("expression") or idx.get("name") or "<expr>",))
            indexes.append(
                IndexInfo(
                    name=idx.get("name"),
                    columns=expr_cols,
                    unique=bool(idx.get("unique", False)),
                    dialect=backend,
                    is_expression=True,
                )
            )
        else:
            indexes.append(
                IndexInfo(
                    name=idx.get("name"),
                    columns=tuple(col_names),
                    unique=bool(idx.get("unique", False)),
                    dialect=backend,
                )
            )

    # unique constraints as indexes (to normalize across dialects)
    try:
        for uc in inspector.get_unique_constraints(table_name, schema=schema) or []:
            cols = tuple(uc.get("column_names") or ())
            if not cols:
                continue
            candidate = IndexInfo(
                name=uc.get("name"),
                columns=cols,
                unique=True,
                dialect=backend,
            )
            # avoid duplicates: compare (unique, columns)
            sigs = {(i.unique, i.columns) for i in indexes}
            if (True, cols) not in sigs:
                indexes.append(candidate)
    except NotImplementedError:
        # some dialects may not implement this API; ignore
        pass

    return TableInfo(
        exists=True,
        columns=columns,
        primary_key=pk_cols,
        foreign_keys=tuple(fk_list),
        indexes=tuple(indexes),
    )


def diff_columns(all_infos: Dict[str, TableInfo]) -> Dict[str, Dict[str, Optional[ColumnInfo]]]:
    # union of column names
    names: set[str] = set()
    for info in all_infos.values():
        if not info.exists:
            continue
        names.update(info.columns.keys())

    per_column: Dict[str, Dict[str, Optional[ColumnInfo]]] = {}
    for col in sorted(names):
        per_db: Dict[str, Optional[ColumnInfo]] = {}
        for db, info in all_infos.items():
            per_db[db] = info.columns.get(col) if info.exists else None
        per_column[col] = per_db
    return per_column


def normalize_index_signature(idx: IndexInfo) -> Tuple[bool, Tuple[str, ...]]:
    # ignore index name; compare by (unique flag, columns or expression placeholder)
    return (idx.unique, idx.columns)


def diff_indexes(all_infos: Dict[str, TableInfo]) -> Dict[str, Dict[str, List[Tuple[bool, Tuple[str, ...]]]]]:
    # build per-db sets
    per_db_sets: Dict[str, set[Tuple[bool, Tuple[str, ...]]]] = {}
    union: set[Tuple[bool, Tuple[str, ...]]] = set()

    for db, info in all_infos.items():
        if not info.exists:
            per_db_sets[db] = set()
            continue
        s = {normalize_index_signature(i) for i in info.indexes}
        per_db_sets[db] = s
        union |= s

    result: Dict[str, Dict[str, List[Tuple[bool, Tuple[str, ...]]]]] = {}
    for db, s in per_db_sets.items():
        missing = sorted(union - s)
        extra = sorted(s - union)  # should be empty by definition; kept for clarity
        result[db] = {"missing": missing, "present": sorted(s), "all": sorted(union), "extra": extra}
    return result


def all_equal(values: Sequence[Any]) -> bool:
    it = iter(values)
    try:
        first = next(it)
    except StopIteration:
        return True
    return all(v == first for v in it)


def render_text_report(table: str, per_db_info: Dict[str, TableInfo]) -> str:
    lines: List[str] = []
    # existence
    for db, info in per_db_info.items():
        lines.append(f"[{db}] table '{table}': {'FOUND' if info.exists else 'MISSING'}")
    lines.append("")

    # primary key differences
    pk_map = {db: info.primary_key if info.exists else tuple() for db, info in per_db_info.items()}
    if not all_equal(pk_map.values()):
        lines.append("Primary key differences:")
        for db, pk in pk_map.items():
            lines.append(f"  - {db}: {list(pk) if pk else 'None'}")
        lines.append("")

    # columns
    lines.append("Columns:")
    per_col = diff_columns(per_db_info)
    for col in sorted(per_col.keys()):
        per_db = per_col[col]
        # compute signatures for comparison
        sigs = []
        for db in per_db_info.keys():
            c = per_db[db]
            if c is None:
                sigs.append(None)
            else:
                sigs.append((c.type, c.nullable, c.default, c.autoincrement))
        if not all_equal(sigs):
            lines.append(f"  - {col}:")
            for db, c in per_db.items():
                if c is None:
                    lines.append(f"    {db}: MISSING")
                else:
                    lines.append(
                        f"    {db}: type={c.type}, nullable={c.nullable}, default={c.default}, autoinc={c.autoincrement}"
                    )
    lines.append("")

    # indexes
    idx_diff = diff_indexes(per_db_info)
    lines.append("Indexes (by [unique, columns]):")
    # show union
    union: set[Tuple[bool, Tuple[str, ...]]] = set()
    for v in idx_diff.values():
        union |= set(v.get("all", []))
    for spec in sorted(union):
        uflag, cols = spec
        lines.append(f"  - [{'U' if uflag else 'N'}] {list(cols)}")
        for db in per_db_info.keys():
            present = spec in set(idx_diff[db].get("present", []))
            lines.append(f"      {db}: {'present' if present else 'missing'}")
    return "\n".join(lines)


def render_markdown_report(table: str, per_db_info: Dict[str, TableInfo]) -> str:
    lines: List[str] = []
    lines.append(f"### 表 `{table}` 对比结果\n")
    # existence
    lines.append("**存在性**:")
    for db, info in per_db_info.items():
        lines.append(f"- **{db}**: {'存在' if info.exists else '不存在'}")
    lines.append("")

    # PK
    pk_map = {db: info.primary_key if info.exists else tuple() for db, info in per_db_info.items()}
    if not all_equal(pk_map.values()):
        lines.append("**主键差异**:")
        for db, pk in pk_map.items():
            lines.append(f"- **{db}**: {list(pk) if pk else '无'}")
        lines.append("")

    # columns table
    per_col = diff_columns(per_db_info)
    headers = ["列名"] + list(per_db_info.keys())
    lines.append("**列差异**:")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for col in sorted(per_col.keys()):
        per_db = per_col[col]
        # compute sigs
        sigs = []
        for db in per_db_info.keys():
            c = per_db[db]
            if c is None:
                sigs.append("MISSING")
            else:
                sigs.append(f"type={c.type}, null={c.nullable}, def={c.default}, ai={c.autoincrement}")
        if not all_equal(sigs):
            row = [col] + sigs
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # indexes
    lines.append("**索引差异** (按唯一性与列集合对比):")
    idx_diff = diff_indexes(per_db_info)
    union: set[Tuple[bool, Tuple[str, ...]]] = set()
    for v in idx_diff.values():
        union |= set(v.get("all", []))
    for spec in sorted(union):
        uflag, cols = spec
        lines.append(f"- [{'唯一' if uflag else '普通'}] {list(cols)}")
        for db in per_db_info.keys():
            present = spec in set(idx_diff[db].get("present", []))
            lines.append(f"  - **{db}**: {'存在' if present else '缺失'}")
    return "\n".join(lines)


def build_json_result(table: str, per_db_info: Dict[str, TableInfo]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"table": table, "databases": {}, "differences": {}}

    # per-db snapshot
    for db, info in per_db_info.items():
        result["databases"][db] = {
            "exists": info.exists,
            "primary_key": list(info.primary_key) if info.exists else [],
            "columns": {name: c.__dict__ for name, c in info.columns.items()} if info.exists else {},
            "indexes": [
                {"unique": i.unique, "columns": list(i.columns), "name": i.name, "is_expression": i.is_expression}
                for i in info.indexes
            ]
            if info.exists
            else [],
        }

    # diffs
    # pk diff
    pk_map = {db: info.primary_key if info.exists else tuple() for db, info in per_db_info.items()}
    result["differences"]["primary_key_equal"] = all_equal(pk_map.values())

    # column diffs
    per_col = diff_columns(per_db_info)
    col_diffs: Dict[str, Dict[str, Any]] = {}
    for col, per_db in per_col.items():
        sigs = []
        for db in per_db_info.keys():
            c = per_db[db]
            sigs.append(None if c is None else (c.type, c.nullable, c.default, c.autoincrement))
        if not all_equal(sigs):
            col_diffs[col] = {
                db: (None if per_db[db] is None else per_db[db].__dict__) for db in per_db_info.keys()
            }
    result["differences"]["columns"] = col_diffs

    # index diffs
    idx_diff = diff_indexes(per_db_info)
    result["differences"]["indexes"] = idx_diff

    return result


def compare_for_table(table: str, engines: List[Tuple[str, Engine]], schema: Optional[str], output: str) -> Tuple[str, bool]:
    per_db_info: Dict[str, TableInfo] = {}
    for label, engine in engines:
        try:
            info = fetch_table_info(engine, table, schema)
        except Exception as exc:
            per_db_info[label] = TableInfo(exists=False)
            print(f"[WARN] {label}: inspect error for table '{table}': {exc}", file=sys.stderr)
            continue
        per_db_info[label] = info

    has_diff = False
    # simple diff presence detection
    # existence
    exist_values = [info.exists for info in per_db_info.values()]
    if not all_equal(exist_values):
        has_diff = True

    # pk
    pk_map = {db: info.primary_key if info.exists else tuple() for db, info in per_db_info.items()}
    if not all_equal(pk_map.values()):
        has_diff = True

    # columns
    per_col = diff_columns(per_db_info)
    for col, per_db in per_col.items():
        sigs = []
        for db in per_db_info.keys():
            c = per_db[db]
            sigs.append(None if c is None else (c.type, c.nullable, c.default, c.autoincrement))
        if not all_equal(sigs):
            has_diff = True
            break

    # indexes
    idx_diff = diff_indexes(per_db_info)
    # if union differs across dbs, it's a diff
    union_sets = [set(v.get("present", [])) for v in idx_diff.values()]
    for s in union_sets[1:]:
        if s != union_sets[0]:
            has_diff = True
            break

    if output == "json":
        payload = build_json_result(table, per_db_info)
        return json.dumps(payload, ensure_ascii=False, indent=2), has_diff
    elif output == "markdown":
        return render_markdown_report(table, per_db_info), has_diff
    else:
        return render_text_report(table, per_db_info), has_diff


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比多个数据库中同名表的结构差异（包含索引）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--db", action="append", required=True,
        help="数据库连接URL（SQLAlchemy格式），可重复，至少两个",
    )
    parser.add_argument(
        "-n", "--name", action="append",
        help="对应 --db 的显示名称，数量需与 --db 相同；不提供则自动推导",
    )
    parser.add_argument(
        "-t", "--table", action="append", required=True,
        help="要对比的表名，可重复",
    )
    parser.add_argument(
        "-s", "--schema", default=None,
        help="可选的 schema（PostgreSQL 默认为 public；MySQL 可忽略）",
    )
    parser.add_argument(
        "-o", "--output", choices=["text", "markdown", "json"], default="text",
        help="输出格式",
    )
    parser.add_argument(
        "--fail-on-diff", action="store_true",
        help="若存在差异则以非零状态码退出（CI场景）",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if len(args.db) < 2:
        print("至少提供两个 --db 进行对比", file=sys.stderr)
        return 2

    names: List[str]
    if args.name:
        if len(args.name) != len(args.db):
            print("--name 的数量必须与 --db 一致", file=sys.stderr)
            return 2
        names = args.name
    else:
        names = [derive_label_from_uri(u) for u in args.db]

    # create engines once
    engines: List[Tuple[str, Engine]] = []
    try:
        for label, uri in zip(names, args.db):
            engines.append((label, build_engine(uri)))
    except SQLAlchemyError as exc:
        print(f"创建数据库连接失败: {exc}", file=sys.stderr)
        return 2

    overall_diff = False

    for table in args.table:
        report, has_diff = compare_for_table(table, engines, args.schema, args.output)
        print(report)
        print("\n" + ("=" * 80) + "\n")
        overall_diff = overall_diff or has_diff

    # dispose engines
    for _, eng in engines:
        try:
            eng.dispose()
        except Exception:
            pass

    if args.fail_on_diff and overall_diff:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
