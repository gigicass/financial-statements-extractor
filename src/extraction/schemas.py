"""Pydantic models for extracted financial statements."""

from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class StatementType(str, Enum):
    """Statement types, matching the CVM dataset names in comments."""

    BALANCE_SHEET = "balance_sheet"  # CVM: BPA (assets) + BPP (liabilities and equity)
    INCOME_STATEMENT = "income_statement"  # CVM: DRE
    CASH_FLOW = "cash_flow"  # CVM: DFC_MI / DFC_MD


class Scope(str, Enum):
    """Consolidated or parent-company-only figures (CVM files `con` / `ind`)."""

    CONSOLIDATED = "consolidated"
    INDIVIDUAL = "individual"


class CurrencyScale(str, Enum):
    """Scale of the reported values, mirroring the CVM column ESCALA_MOEDA."""

    UNIT = "UNIDADE"
    THOUSAND = "MIL"


class StatementLine(BaseModel):
    """One line (account) of a financial statement for a single period."""

    account_name: str = Field(description="Account name exactly as printed in the PDF.")
    value: Decimal = Field(
        description="Value as printed, in the statement's scale. Negative if shown "
        "in parentheses or with a minus sign. Dashes ('-') mean zero."
    )
    is_total: bool = Field(
        default=False,
        description="True for subtotal/total lines (e.g. 'Ativo Circulante', 'TOTAL DO ATIVO').",
    )


class FinancialStatement(BaseModel):
    """A full statement for one company, scope and reference date."""

    company: str
    statement_type: StatementType
    scope: Scope
    reference_date: date = Field(description="Period end date, e.g. 2025-12-31.")
    currency_scale: CurrencyScale
    lines: list[StatementLine] = Field(min_length=1)

    def value_multiplier(self) -> int:
        """Factor that converts this statement's values to units of BRL."""
        return 1000 if self.currency_scale == CurrencyScale.THOUSAND else 1
