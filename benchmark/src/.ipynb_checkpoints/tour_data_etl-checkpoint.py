from pathlib import Path
import json

import pandas as pd


class TourDataETL:
    """관광 데이터를 불러오고 변환하여 Parquet으로 저장한다."""



    # 파일 로드
    def load_data(self, file_path):
        """파일 확장자를 확인하여 DataFrame으로 불러온다."""

        input_path = Path(file_path)
        extension = input_path.suffix.lower() # 파일 경로에서 마지막 확장자를 가져와 소문자로 변환한 뒤 extension 변수에 저장

        if extension == ".csv":
            return pd.read_csv(input_path)

        if extension == ".json":
            with input_path.open(
                "r", # 읽기 모드
                encoding="utf-8",
            ) as file:
                data = json.load(file) # python 객체로 변환

            if isinstance(data, dict):
                return pd.DataFrame([data]) # 딕셔너리를 한 행으로 만들기 위해

            if isinstance(data, list):
                return pd.DataFrame(data)

            raise ValueError(
                "JSON의 최상위 구조는 "
                "dict 또는 list여야 합니다."
            )

        if extension == ".xml":
            return pd.read_xml(input_path)

        if extension == ".parquet":
            return pd.read_parquet(input_path)

        raise ValueError(
            f"지원하지 않는 파일 형식입니다: {extension}"
        )



    # 데이터 변환
    def transform(self, df):
        """빈 행·열을 제거하고 중첩된 값을 평탄화한다."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df는 pandas DataFrame이어야 합니다."
            )

        transformed_df = df.copy() # 복사본

        # 완전히 빈 행 제거
        transformed_df = transformed_df.dropna(
            axis=0,
            how="all",
        )
        # 완전히 빈 열 제거
        transformed_df = transformed_df.dropna(
            axis=1,
            how="all",
        )

        # 중첩 구조 반복 탐색
        while True:
            nested_value_found = False

            for column in transformed_df.columns.tolist(): # DataFrame의 컬럼 이름을 리스트로 바꾸고 하나씩 확인
                # 검사 중 하나라도 딕셔너리가 있는지(True) 확인
                contains_dict = transformed_df[
                    column
                ].map(
                    lambda value: isinstance(
                        value,
                        dict,
                    )
                ).any()
                
                # 딕셔너리 평탄화
                if contains_dict:
                    normalized = pd.json_normalize(
                        transformed_df[column]
                    )
                    # 새로 만든 컬럼명 앞에 기존 컬럼명 붙임
                    normalized = normalized.add_prefix(
                        f"{column}."
                    )
                    # 평탄화한 DataFrame의 인덱스를 기존 DataFrame의 인덱스로 설정
                    normalized.index = (
                        transformed_df.index
                    )
                    # 기존 칼럼 제거 후 새 컬럼 결합
                    transformed_df = pd.concat(
                        [
                            transformed_df.drop(
                                columns=[column]
                            ),
                            normalized,
                        ],
                        axis=1,
                    )
                    # 중첩 구조가 발견되었음을 표시하고 반복문 종료
                    nested_value_found = True
                    break

                # 검사 중 하나라도 리스트가 있는지(True) 확인
                contains_list = transformed_df[
                    column
                ].map(
                    lambda value: isinstance(
                        value,
                        list,
                    )
                ).any()

                if contains_list:
                    # 리스트 평탄화 
                    transformed_df = (
                        transformed_df.explode( # explode()는 리스트를 행으로 분리하는 함수
                            column,
                            ignore_index=True,
                        )
                    )

                    nested_value_found = True
                    break

            if not nested_value_found:
                break
        # 평탄화 완료된 DataFrame의 인덱스를 초기화하고 기존 인덱스는 제거
        return transformed_df.reset_index(
            drop=True
        )



    # 결과 저장
    def save_to_parquet(
        self,
        df,
        out_data_path,
    ):
        """DataFrame을 Parquet 파일로 저장한다."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df는 pandas DataFrame이어야 합니다."
            )

        output_path = Path(out_data_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_parquet(
            output_path,
            index=False,
        )

        return output_path