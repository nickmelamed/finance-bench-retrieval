import pandas as pd


class TableGenerator:
    def summary_table(
        self,
        results,
    ):
        return pd.DataFrame(results)