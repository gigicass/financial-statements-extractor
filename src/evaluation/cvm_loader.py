"""Load and normalize the official CVM DFP/ITR data (the ground truth).

The CVM publishes one zip per year with one CSV per statement and scope, e.g.
`dfp_cia_aberta_BPA_con_2025.csv`. CSVs are `;`-separated and latin-1 encoded.
We read them straight from the zip, without extracting it to disk.
"""

import logging
import zipfile
from decimal import Decimal
from pathlib import Path

import pandas as pd

from src.extraction.schemas import Scope, StatementType

logger = logging.getLogger(__name__)

# The balance sheet is split in two CVM files: assets (BPA) and liabilities + equity (BPP).
CVM_FILES: dict[StatementType, list[str]] = {
    StatementType.BALANCE_SHEET: ["BPA", "BPP"],
    StatementType.INCOME_STATEMENT: ["DRE"],
    StatementType.CASH_FLOW: ["DFC_MI"],  # indirect method; most companies use it
}

SCOPE_SUFFIX = {Scope.CONSOLIDATED: "con", Scope.INDIVIDUAL: "ind"}

SCALE_MULTIPLIER = {"UNIDADE": 1, "MIL": 1000}

CURRENT_PERIOD = "ÚLTIMO"  # ORDEM_EXERC: ÚLTIMO = current year, PENÚLTIMO = prior year

# CVM code of the companies in the study (CD_CVM column).
COMPANIES: dict[str, str] = {
    "WEG": "005410",
    "VALE": "004170",
    "AMBEV": "023264",
    "MGLU": "022470",
}


def _read_csv_from_zip(zip_path: Path, name: str) -> pd.DataFrame:
    """Read one CSV from the CVM zip, keeping every column as text."""
    with zipfile.ZipFile(zip_path) as z, z.open(name) as f:
        return pd.read_csv(f, sep=";", encoding="latin-1", dtype=str)


def _csv_name(zip_path: Path, prefix: str, scope: Scope) -> str:
    """Build the CSV name inside the zip, e.g. dfp_cia_aberta_BPA_con_2025.csv."""
    # zip name: dfp_cia_aberta_2025.zip -> base 'dfp_cia_aberta', year '2025'
    base, year = zip_path.stem.rsplit("_", 1)
    return f"{base}_{prefix}_{SCOPE_SUFFIX[scope]}_{year}.csv"


def _to_units(raw_value: str, multiplier: int) -> Decimal:
    """'42645030.0000000000' in MIL -> Decimal('42645030000') (drops trailing zeros)."""
    value = Decimal(raw_value) * multiplier
    return value.quantize(Decimal(1)) if value == value.to_integral_value() else value


def normalize_cvm_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest version and current period, and convert values to BRL units.

    Returns columns: code, name, value (Decimal, BRL units), is_fixed, level.
    """
    df = df[df["ORDEM_EXERC"] == CURRENT_PERIOD]
    # A filing may be restated (VERSAO 2, 3...); the latest version is the valid one.
    df = df[df["VERSAO"].astype(int) == df["VERSAO"].astype(int).max()]

    multipliers = df["ESCALA_MOEDA"].map(SCALE_MULTIPLIER)
    if multipliers.isna().any():
        raise ValueError(f"Unknown ESCALA_MOEDA: {set(df['ESCALA_MOEDA'])}")

    values = [_to_units(v, m) for v, m in zip(df["VL_CONTA"], multipliers)]
    return pd.DataFrame({
        "code": df["CD_CONTA"].to_numpy(),
        "name": df["DS_CONTA"].to_numpy(),
        "value": values,
        "is_fixed": (df["ST_CONTA_FIXA"] == "S").to_numpy(),
        "level": df["CD_CONTA"].str.count(r"\.").to_numpy() + 1,
    })


def load_cvm_statement(
    zip_path: str | Path,
    cd_cvm: str,
    statement_type: StatementType,
    scope: Scope,
) -> pd.DataFrame:
    """Load one company's statement from the CVM zip, normalized (see normalize_cvm_rows)."""
    zip_path = Path(zip_path)
    frames = []
    for prefix in CVM_FILES[statement_type]:
        df = _read_csv_from_zip(zip_path, _csv_name(zip_path, prefix, scope))
        frames.append(normalize_cvm_rows(df[df["CD_CVM"] == cd_cvm]))
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise ValueError(f"No CVM data for CD_CVM={cd_cvm}, {statement_type.value}, {scope.value}")
    logger.info("Loaded %d CVM accounts for CD_CVM=%s (%s, %s)",
                len(result), cd_cvm, statement_type.value, scope.value)
    return result
