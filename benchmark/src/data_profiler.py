from pathlib import Path
import json

import pandas as pd
from ydata_profiling import ProfileReport


class DataProfiler:
    """DataFrame의 특성을 분석하고 메타데이터를 생성한다."""

    HIGH_MISSING_RATE_THRESHOLD = 0.3

    def profile_data(self, df):
        """ydata-profiling을 이용해 데이터 특성을 분석한다."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df는 pandas DataFrame이어야 합니다."
            )

        report = ProfileReport(
            df,
            title="Tourism Data Profile",
            minimal=True,
            progress_bar=False,
        )

        raw_profile = json.loads(
            report.to_json()
        )

        variable_profiles = raw_profile[
            "variables"
        ]

        numeric_columns = [
            column
            for column, stats in variable_profiles.items()
            if stats.get("type") == "Numeric"
        ]

        datetime_columns = [
            column
            for column, stats in variable_profiles.items()
            if stats.get("type") == "DateTime"
        ]

        categorical_columns = [
            column
            for column, stats in variable_profiles.items()
            if stats.get("type")
            in {
                "Text",
                "Categorical",
                "Boolean",
            }
        ]

        profile_stats = {
            "row_count": int(
                raw_profile["table"]["n"]
            ),
            "column_count": int(
                raw_profile["table"]["n_var"]
            ),
            "column_names": list(
                variable_profiles.keys()
            ),
            "data_types": {
                column: stats.get("type")
                for column, stats
                in variable_profiles.items()
            },
            "missing_count": {
                column: int(
                    stats.get("n_missing", 0)
                )
                for column, stats
                in variable_profiles.items()
            },
            "missing_rate": {
                column: float(
                    stats.get("p_missing", 0.0)
                )
                for column, stats
                in variable_profiles.items()
            },
            "overall_missing_rate": float(
                raw_profile[
                    "table"
                ]["p_cells_missing"]
            ),
            "duplicate_count": int(
                df.duplicated().sum()
            ),
            "numeric_columns": numeric_columns,
            "numeric_column_count": len(
                numeric_columns
            ),
            "datetime_columns": datetime_columns,
            "has_datetime": bool(
                datetime_columns
            ),
            "categorical_columns": (
                categorical_columns
            ),
            "categorical_column_count": len(
                categorical_columns
            ),
        }

        return profile_stats

    def generate_tags(self, profile_stats):
        """프로파일 결과로 이상탐지 매칭용 태그를 생성한다."""

        tags = []

        if profile_stats["has_datetime"]:
            tags.append("time-series")

        if (
            profile_stats[
                "numeric_column_count"
            ]
            >= 2
        ):
            tags.append("multivariate")

        if any(
            missing_rate
            >= self.HIGH_MISSING_RATE_THRESHOLD
            for missing_rate
            in profile_stats[
                "missing_rate"
            ].values()
        ):
            tags.append("high-missing-rate")

        return tags

    def export_metadata(
        self,
        dataset_info,
        profile_stats,
        tags,
        out_meta_path,
    ):
        """데이터셋 정보, 프로파일 결과와 태그를 JSON으로 저장한다."""

        metadata = {
            "dataset_info": dataset_info,
            "profile_stats": profile_stats,
            "tags": tags,
        }

        output_path = Path(out_meta_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return output_path