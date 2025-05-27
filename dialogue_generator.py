#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NDGS v4.9α (Neo-Dialogue Generation System) - 統合・最適化版スクリプト

NDGS v4.9αをベースに、これまでの分析・検証結果と改善提案を踏まえ、
システムの安定性、堅牢性、保守性、そして表現力をさらに高めるための
修正パッチを適用したバージョン。
PsychologicalPhaseV49 Enum (11コアメンバー) への完全移行と、
Pydantic V2 スタイルへの準拠、YAML設定との整合性向上を目指す。
"""

# =============================================================================
# -- Part 0: Import & 初期設定 (v4.9α - Gemini最適化・修正版)
# =============================================================================

# -----------------------------------------------------------------------------
# -- 標準ライブラリ Import
# -----------------------------------------------------------------------------
import argparse
import copy
import enum
import hashlib
import inspect
import json
import logging
import logging.handlers
import math
import os
import pathlib # pathlib.Path を直接使用するため
import pickle
import random
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
import traceback
import builtins # For dummy numpy `any`
from collections import Counter, defaultdict, deque
from contextlib import contextmanager, nullcontext # nullcontext は Python 3.7+
from dataclasses import dataclass, field, fields as dc_fields, is_dataclass # 'fields' は dataclasses.fields を dc_fields として区別
from datetime import datetime, timedelta, timezone
from functools import lru_cache, partial, total_ordering, wraps
from operator import attrgetter, itemgetter
from pathlib import Path, PurePath # Path, PurePath を明示的にインポート
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator, RootModel
from types import MappingProxyType, SimpleNamespace # SimpleNamespace は Python 3.3+
from typing import (
    Any, Callable, ClassVar, Dict, Final, FrozenSet, Generator, Generic,
    Iterable, Iterator, List, Literal, Mapping, MutableMapping, NamedTuple,
    NewType, Optional, overload, ParamSpec, Pattern, Protocol, Sequence, Set,
    TextIO, Tuple, Type, TypeAlias, TypedDict, TypeGuard, TypeVar, Union,
    TYPE_CHECKING, Annotated
)
from uuid import UUID, uuid4
import weakref

# -----------------------------------------------------------------------------
# -- サードパーティライブラリ Import とエラーハンドリング
# -----------------------------------------------------------------------------

_part0_logger = logging.getLogger(f"{__name__}.Part0_Init") # このモジュール専用のロガー

# --- Pydantic ---
PYDANTIC_AVAILABLE = False
BaseModel: Type[Any] = object
Field: Callable[..., Any] = lambda **kwargs: None # type: ignore[misc]
field_validator: Callable[..., Any] = lambda *args, **kwargs: (lambda fn_decorated: fn_decorated)
model_validator: Callable[..., Any] = lambda *args, **kwargs: (lambda fn_decorated: fn_decorated)
ConfigDict: Type[Dict[str, Any]] = dict # Pydantic V2 ConfigDict is a TypedDict
ValidationError: Type[Exception] = type('PydanticValidationErrorFallback', (ValueError,), {})
ValidationInfo: Type[Any] = type('PydanticValidationInfoFallback', (), {})
RootModel: Type[Any] = type('PydanticRootModelFallback', (dict,), {}) # RootModel is a generic class
# Pydantic V2 の型
FilePath: Type[Any] = str
DirectoryPath: Type[Any] = str
EmailStr: Type[Any] = str
HttpUrl: Type[Any] = str
PositiveInt = int
NegativeInt = int
PositiveFloat = float
NegativeFloat = float
NonNegativeInt = int
NonNegativeFloat = float
StrictBool = bool
StrictStr = str
PydanticUndefinedAnnotation: Type[Exception] = type('PydanticUndefinedAnnotationFallback', (Exception,), {})
create_model: Callable[..., Type[BaseModel]] = lambda *args, **kwargs: type('DummyModel', (object,), {}) # type: ignore


try:
    from pydantic import (
        BaseModel as PydanticBaseModel, Field as PydanticField,
        field_validator as pydantic_field_validator, model_validator as pydantic_model_validator,
        ConfigDict as PydanticConfigDict, ValidationError as PydanticValidationError,
        ValidationInfo as PydanticValidationInfo, RootModel as PydanticRootModelActual,
        PositiveInt as PydanticPositiveInt, NegativeInt as PydanticNegativeInt,
        PositiveFloat as PydanticPositiveFloat, NegativeFloat as PydanticNegativeFloat,
        NonNegativeInt as PydanticNonNegativeInt, NonNegativeFloat as PydanticNonNegativeFloat,
        StrictBool as PydanticStrictBool, StrictStr as PydanticStrictStr,
        create_model as pydantic_create_model # create_modelもインポート
    )
    from pydantic.types import FilePath as PydanticFilePath, DirectoryPath as PydanticDirectoryPath
    from pydantic.networks import EmailStr as PydanticEmailStr, HttpUrl as PydanticHttpUrl
    from pydantic.errors import PydanticUndefinedAnnotation as ActualPydanticUndefinedAnnotation


    BaseModel = PydanticBaseModel
    Field = PydanticField # type: ignore[assignment]
    field_validator = pydantic_field_validator
    model_validator = pydantic_model_validator
    ConfigDict = PydanticConfigDict # type: ignore[misc]
    ValidationError = PydanticValidationError
    ValidationInfo = PydanticValidationInfo # type: ignore
    RootModel = PydanticRootModelActual # type: ignore
    FilePath = PydanticFilePath # type: ignore
    DirectoryPath = PydanticDirectoryPath # type: ignore
    EmailStr = PydanticEmailStr # type: ignore
    HttpUrl = PydanticHttpUrl # type: ignore
    PositiveInt = PydanticPositiveInt # type: ignore
    NegativeInt = PydanticNegativeInt # type: ignore
    PositiveFloat = PydanticPositiveFloat # type: ignore
    NegativeFloat = PydanticNegativeFloat # type: ignore
    NonNegativeInt = PydanticNonNegativeInt # type: ignore
    NonNegativeFloat = PydanticNonNegativeFloat # type: ignore
    StrictBool = PydanticStrictBool # type: ignore
    StrictStr = PydanticStrictStr # type: ignore
    create_model = pydantic_create_model
    PydanticUndefinedAnnotation = ActualPydanticUndefinedAnnotation # type: ignore
    PYDANTIC_AVAILABLE = True
    _part0_logger.info("Pydantic V2 ライブラリのロードに成功しました。")
except ImportError as e_pydantic:
    _part0_logger.critical(
        f"致命的エラー: Pydantic V2 ライブラリが見つかりません ({e_pydantic})。NDGS v4.9α は Pydantic V2 に強く依存しています。"
        "             'pip install \"pydantic>=2.0\"' でインストールしてください。処理を続行できません。"
    )
    sys.exit(1) # Pydanticなしでは動作不可なので終了

# --- annotated-types (Pydantic V2 との連携用) ---
ANNOTATED_TYPES_AVAILABLE = False
# フォールバック定義 (annotated-types がない場合でも基本的な型チェックが通るように)
# SyntaxError修正: if文とclass定義を分離
if 'Ge' not in globals():
    class Ge: # type: ignore
        def __init__(self, v: Any): self.ge = v
if 'Gt' not in globals():
    class Gt: # type: ignore
        def __init__(self, v: Any): self.gt = v
if 'Le' not in globals():
    class Le: # type: ignore
        def __init__(self, v: Any): self.le = v
if 'Lt' not in globals():
    class Lt: # type: ignore
        def __init__(self, v: Any): self.lt = v
if 'MinLen' not in globals():
    class MinLen: # type: ignore
        def __init__(self, v: Any): self.min_length = v
if 'MaxLen' not in globals():
    class MaxLen: # type: ignore
        def __init__(self, v: Any): self.max_length = v

try:
    from annotated_types import Ge as ActualGe, Gt as ActualGt, Le as ActualLe, Lt as ActualLt, MaxLen as ActualMaxLen, MinLen as ActualMinLen
    # グローバルスコープのフォールバック定義を上書き
    Ge = ActualGe # type: ignore
    Gt = ActualGt # type: ignore
    Le = ActualLe # type: ignore
    Lt = ActualLt # type: ignore
    MinLen = ActualMinLen # type: ignore
    MaxLen = ActualMaxLen # type: ignore
    ANNOTATED_TYPES_AVAILABLE = True
    _part0_logger.info("annotated-types ライブラリのロードに成功しました。")
except ImportError:
    _part0_logger.warning(
        "annotated-types ライブラリが見つかりません。一部のPydantic V2スタイル型アノテーションが制限される可能性があります。"
        "      'pip install annotated-types' でインストールを推奨します。"
    )

# --- Google Generative AI ---
GOOGLE_API_AVAILABLE = False
genai: Any = None # モックまたは実際のモジュール
google_exceptions: Any = None # モックまたは実際のモジュール
try:
    import google.generativeai as genai_actual
    from google.api_core import exceptions as google_exceptions_actual
    genai = genai_actual
    google_exceptions = google_exceptions_actual
    GOOGLE_API_AVAILABLE = True
    _part0_logger.info("google-generativeai ライブラリのロードに成功しました。")
except ImportError:
    _part0_logger.warning(
        "google-generativeai ライブラリが見つかりません。API関連機能は無効になります。"
        "     'pip install google-generativeai' でインストールを推奨します。"
    )
    class google_exceptions_mock: # type: ignore
        PermissionDenied = type('PermissionDenied', (Exception,), {})
        InvalidArgument = type('InvalidArgument', (Exception,), {})
        ResourceExhausted = type('ResourceExhausted', (Exception,), {})
        InternalServerError = type('InternalServerError', (Exception,), {})
        ServiceUnavailable = type('ServiceUnavailable', (Exception,), {})
        DeadlineExceeded = type('DeadlineExceeded', (Exception,), {})
        Unauthenticated = type('Unauthenticated', (Exception,), {})
        NotFound = type('NotFound', (Exception,), {})
        BadRequest = type('BadRequest', (Exception,), {})
        Aborted = type('Aborted', (Exception,), {})
        Cancelled = type('Cancelled', (Exception,), {})
        FailedPrecondition = type('FailedPrecondition', (Exception,), {})
        GoogleAPIError = type('GoogleAPIError', (Exception,), {})
        # Gemini API特有のエラーのモック (必要に応じて追加)
        BlockedByPolicy = type('BlockedByPolicy', (GoogleAPIError,), {}) # GoogleAPIErrorを継承
        GenerativeAIException = type('GenerativeAIException', (GoogleAPIError,), {})
        StopCandidateException = type('StopCandidateException', (GenerativeAIException,), {})
    google_exceptions = google_exceptions_mock

    class genai_mock_class: # クラス名を変更して genai 変数との衝突を回避
        class types: # type: ignore
            class GenerationConfig(dict): # dictを継承して**展開できるように
                pass
            class GenerateContentResponse: # type: ignore
                def __init__(self) -> None:
                    self.candidates: List[Any] = []
                    self.prompt_feedback: Any = None
                    self.text: Optional[str] = None
            class SafetyRating: # type: ignore
                def __init__(self, category: Any, probability: Any):
                    self.category = category
                    self.probability = probability
            class Candidate: # type: ignore
                class FinishReason(enum.Enum):
                    FINISH_REASON_UNSPECIFIED = 0
                    STOP = 1
                    MAX_TOKENS = 2
                    SAFETY = 3
                    RECITATION = 4
                    OTHER = 5
                    UNKNOWN = 6 # 独自追加
                def __init__(self) -> None:
                    self.content: Optional['genai_mock_class.types.Content'] = None
                    self.finish_reason: Optional['genai_mock_class.types.Candidate.FinishReason'] = None
                    self.safety_ratings: List['genai_mock_class.types.SafetyRating'] = []
            class Content: # type: ignore
                def __init__(self) -> None:
                    self.parts: List['genai_mock_class.types.Part'] = []
                    self.role: Optional[str] = None
            class Part: # type: ignore
                def __init__(self) -> None:
                    self.text: Optional[str] = None
            class PromptFeedback: # type: ignore
                def __init__(self) -> None:
                    self.block_reason: Optional[Any] = None # BlockReason enum or similar
                    self.safety_ratings: List['genai_mock_class.types.SafetyRating'] = []
            class HarmCategory(enum.Enum): # 実際のEnumに基づいて
                HARM_CATEGORY_UNSPECIFIED = 0; HARM_CATEGORY_DEROGATORY = 1; HARM_CATEGORY_TOXICITY = 2
                HARM_CATEGORY_VIOLENCE = 3; HARM_CATEGORY_SEXUAL = 4; HARM_CATEGORY_MEDICAL = 5
                HARM_CATEGORY_DANGEROUS = 6; HARM_CATEGORY_HARASSMENT = 7; HARM_CATEGORY_HATE_SPEECH = 8
                HARM_CATEGORY_SEXUALLY_EXPLICIT = 9; HARM_CATEGORY_DANGEROUS_CONTENT = 10
            class HarmBlockThreshold(enum.Enum): # SafetySettingsで使用
                HARM_BLOCK_THRESHOLD_UNSPECIFIED = 0
                BLOCK_LOW_AND_ABOVE = 1
                BLOCK_MEDIUM_AND_ABOVE = 2
                BLOCK_ONLY_HIGH = 3
                BLOCK_NONE = 4
            class HarmProbability(enum.Enum):
                HARM_PROBABILITY_UNSPECIFIED = 0; NEGLIGIBLE = 1; LOW = 2; MEDIUM = 3; HIGH = 4


        @staticmethod
        def configure(*args: Any, **kwargs: Any) -> None:
            _part0_logger.info(f"Mock genai.configure called with args: {args}, kwargs: {kwargs}")
        class GenerativeModel: # type: ignore
            def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
                self.model_name = model_name
                _part0_logger.info(f"Mock GenerativeModel initialized for model: {model_name} with kwargs: {kwargs}")
            def generate_content(self, contents: Any, *args: Any, **kwargs: Any) -> 'genai_mock_class.types.GenerateContentResponse':
                _part0_logger.info(f"Mock generate_content called for model: {self.model_name} with contents: {str(contents)[:100]}..., kwargs: {kwargs}")
                mock_response = genai_mock_class.types.GenerateContentResponse()
                mock_candidate = genai_mock_class.types.Candidate()
                mock_candidate.content = genai_mock_class.types.Content()
                mock_part = genai_mock_class.types.Part()
                mock_part.text = "This is a mock response from the dummy GenerativeModel."
                mock_candidate.content.parts = [mock_part]
                mock_candidate.finish_reason = genai_mock_class.types.Candidate.FinishReason.STOP
                mock_candidate.safety_ratings = [
                    genai_mock_class.types.SafetyRating(
                        category=genai_mock_class.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        probability=genai_mock_class.types.HarmProbability.NEGLIGIBLE
                    )
                ]
                mock_response.candidates = [mock_candidate]
                mock_response.text = mock_part.text # Add .text attribute directly to response for easier access in mocks
                return mock_response
    genai = genai_mock_class()

# --- PyYAML ---
YAML_AVAILABLE = False
yaml: Any = None # モックまたは実際のモジュール
try:
    import yaml as yaml_actual
    yaml = yaml_actual
    YAML_AVAILABLE = True
    _part0_logger.info("PyYAML ライブラリのロードに成功しました。")
except ImportError:
    _part0_logger.critical(
        "致命的エラー: PyYAML ライブラリが見つかりません。NDGS v4.9α は設定ファイル処理にPyYAMLを必要とします。"
        "             'pip install PyYAML' でインストールしてください。処理を続行できません。"
    )
    sys.exit(1) # YAMLなしでは設定が読めないので終了

# --- tqdm, filelock, jsonschema ---
TQDM_AVAILABLE = False
tqdm: Type[Any] = object # フォールバック型
try:
    from tqdm import tqdm as actual_tqdm
    tqdm = actual_tqdm
    TQDM_AVAILABLE = True
    _part0_logger.info("tqdm ライブラリのロードに成功しました。")
except ImportError:
    _part0_logger.warning("tqdm ライブラリが見つかりません。プログレスバーは表示されません。'pip install tqdm' を推奨します。")
    class tqdm_mock: # type: ignore
        def __init__(self, iterable: Optional[Any] = None, *args: Any, **kwargs: Any) -> None: self.iterable = iterable if iterable else []
        def __iter__(self) -> Any: return iter(self.iterable)
        def __enter__(self) -> 'tqdm_mock': return self
        def __exit__(self, *args: Any) -> None: pass
        def update(self, n: int = 1) -> None: pass
        def set_description(self, desc: Optional[str] = None) -> None: pass
        def close(self) -> None: pass
    tqdm = tqdm_mock

FILELOCK_AVAILABLE = False
FileLock: Optional[Type[Any]] = None
FileLockTimeout: Type[Exception] = type('FileLockTimeoutFallback', (TimeoutError,), {})
try:
    from filelock import FileLock as ActualFileLock, Timeout as ActualFileLockTimeout
    FileLock = ActualFileLock
    FileLockTimeout = ActualFileLockTimeout
    FILELOCK_AVAILABLE = True
    _part0_logger.info("filelock ライブラリのロードに成功しました。")
except ImportError:
    _part0_logger.warning("filelock ライブラリが見つかりません。一部のファイル操作の排他制御が無効になります。'pip install filelock' を推奨します。")

JSONSCHEMA_AVAILABLE = False
jsonschema: Any = None
try:
    import jsonschema as actual_jsonschema
    jsonschema = actual_jsonschema
    JSONSCHEMA_AVAILABLE = True
    _part0_logger.info("jsonschema ライブラリのロードに成功しました。")
except ImportError:
    _part0_logger.info("jsonschema ライブラリが見つかりません。JSONスキーマ検証はスキップされます。'pip install jsonschema' を推奨します。")

# --- NLP & ML 関連 (オプションだが機能に影響) ---
SPACY_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False
SKLEARN_AVAILABLE = False
SCIPY_AVAILABLE = False
NUMPY_AVAILABLE = False
np_module: Any = None # numpyモジュールまたはダミーを格納する変数
TfidfVectorizer_cls: Optional[Type] = None # TfidfVectorizerクラスまたはダミー
cosine_similarity_func: Optional[Callable[..., Any]] = None # cosine_similarity関数またはダミー

try:
    import spacy
    SPACY_AVAILABLE = True
    _part0_logger.info("spaCy ライブラリのロードに成功しました。")
except ImportError:
    _part0_logger.warning("spaCy ライブラリが見つかりません。高度なNLP分析機能が無効になります。'pip install spacy' およびモデルのダウンロードを推奨します。")

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
    _part0_logger.info("transformers ライブラリのロードに成功しました。")
    try:
        import torch
        TORCH_AVAILABLE = True
        _part0_logger.info("torch ライブラリのロードに成功しました。")
    except ImportError:
        _part0_logger.warning("torch ライブラリが見つかりません。transformers の一部機能が制限される可能性があります。")
except ImportError:
    _part0_logger.warning("transformers ライブラリが見つかりません。MLベースの感情分析などが無効になります。'pip install transformers[torch]' を推奨します。")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as SklearnCosineSimilarity
    TfidfVectorizer_cls = SklearnTfidfVectorizer
    cosine_similarity_func = SklearnCosineSimilarity
    SKLEARN_AVAILABLE = True
    _part0_logger.info("scikit-learn ライブラリのロードに成功しました。")
except ImportError:
    _part0_logger.warning("scikit-learn ライブラリが見つかりません。一部のテキスト分析機能 (TF-IDF, コサイン類似度) が代替実装または無効になります。")
    class _DummyTfidfVectorizerSklearn: # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def fit_transform(self, texts: Any) -> Any: return type('DummyMatrix', (), {'shape': (len(texts) if hasattr(texts,'__len__') else 0, 0), 'toarray': lambda: []})()
        def get_feature_names_out(self) -> List[str]: return []
    TfidfVectorizer_cls = _DummyTfidfVectorizerSklearn
    def _dummy_cosine_similarity_sklearn(X: Any, Y: Optional[Any] = None) -> Any:
        if Y is None: Y = X
        try:
            len_X = X.shape[0] if hasattr(X, 'shape') else len(X)
            len_Y = Y.shape[0] if hasattr(Y, 'shape') else len(Y)
            return [[0.5 for _ in range(len_Y)] for _ in range(len_X)] # 常に0.5を返す単純なモック
        except: return [[0.5]] # エラー時はさらに単純化
    cosine_similarity_func = _dummy_cosine_similarity_sklearn

try:
    import scipy
    SCIPY_AVAILABLE = True
    _part0_logger.info("scipy ライブラリのロードに成功しました。")
except ImportError:
    _part0_logger.info("scipy ライブラリが見つかりません。一部の高度な数値計算機能が制限される可能性があります。")

try:
    import numpy
    np_module = numpy
    NUMPY_AVAILABLE = True
    _part0_logger.info("numpy ライブラリのロードに成功しました。")
except ImportError:
    _part0_logger.warning("numpy ライブラリが見つかりません。多くの数値計算が代替実装または無効になります。'pip install numpy' を強く推奨します。")
    class _DummyNumpyModule: # type: ignore
        def array(self, data: Any, dtype: Any = None) -> Any: return list(data)
        def mean(self, a: Any, axis: Any = None) -> Any: return sum(a)/len(a) if hasattr(a, '__len__') and len(a) > 0 else 0.0
        def std(self, a: Any, axis: Any = None) -> Any:
             if not (hasattr(a, '__len__') and len(a) > 0): return 0.0
             m = self.mean(a)
             return math.sqrt(sum((x - m)**2 for x in a) / len(a)) if len(a) > 0 else 0.0
        def isnan(self, x: Any) -> bool: return math.isnan(x) if isinstance(x, (float, int)) else False
        def isinf(self, x: Any) -> bool: return math.isinf(x) if isinstance(x, (float, int)) else False
        def any(self, a: Any, axis: Any = None, out: Any = None, keepdims: Any = False) -> bool: return builtins.any(a)
        def abs(self, x: Any) -> Any: return builtins.abs(x)
        # ndarray は型なので、list を返すようにする (より安全なフォールバック)
        @property
        def ndarray(self) -> Type[list]: return list # type: ignore
        def interp(self, x_val: float, xp_list: List[float], fp_list: List[float]) -> float:
            if not (xp_list and fp_list and len(xp_list) == len(fp_list)):
                return fp_list[0] if fp_list else x_val
            if len(xp_list) == 1:
                return fp_list[0]
            # Ensure xp_list is sorted for correct interpolation logic
            sorted_indices = sorted(range(len(xp_list)), key=xp_list.__getitem__)
            xp_sorted = [xp_list[i] for i in sorted_indices]
            fp_sorted = [fp_list[i] for i in sorted_indices]

            if x_val <= xp_sorted[0]: return fp_sorted[0]
            if x_val >= xp_sorted[-1]: return fp_sorted[-1]
            for i_interp in range(len(xp_sorted) - 1):
                if xp_sorted[i_interp] <= x_val <= xp_sorted[i_interp+1]:
                    if abs(xp_sorted[i_interp+1] - xp_sorted[i_interp]) < 1e-9:
                        return fp_sorted[i_interp]
                    return fp_sorted[i_interp] + (x_val - xp_sorted[i_interp]) * \
                           (fp_sorted[i_interp+1] - fp_sorted[i_interp]) / (xp_sorted[i_interp+1] - xp_sorted[i_interp])
            return fp_sorted[-1]
        tanh = math.tanh
    np_module = _DummyNumpyModule()

np: Any = np_module # グローバル変数 np をここで定義

# -----------------------------------------------------------------------------
# -- グローバルユーティリティ関数
# -----------------------------------------------------------------------------
_logger_ggt = logging.getLogger(f"{__name__}._get_global_type_utils")

def _get_global_type(name: str, expected_meta_type: Optional[type] = None) -> Optional[Type[Any]]:
    target_type = globals().get(name)
    if target_type is None:
        _logger_ggt.debug(f"グローバルスコープで型/クラス '{name}' が見つかりません。")
        return None
    actual_meta_type = type(target_type)
    
    PydanticBaseModel_cls = globals().get('BaseModel') # PydanticのBaseModelを取得
    is_pydantic_model_type = False
    if PYDANTIC_AVAILABLE and PydanticBaseModel_cls and isinstance(target_type, type) and issubclass(target_type, PydanticBaseModel_cls):
        is_pydantic_model_type = True

    is_enum_meta_type = isinstance(target_type, enum.EnumMeta)

    if expected_meta_type:
        if actual_meta_type is not expected_meta_type:
            if expected_meta_type is enum.EnumMeta and is_enum_meta_type:
                 _logger_ggt.debug(f"'{name}' は enum.EnumMeta のインスタンス（Enumクラス）であるため、メタタイプチェックをパスとみなします。")
            elif expected_meta_type is type and (isinstance(target_type, type) or is_pydantic_model_type):
                 _logger_ggt.debug(f"'{name}' は type または Pydantic BaseModel のサブクラスであるため、メタタイプチェックをパスとみなします。")
            else:
                _logger_ggt.debug(
                    f"型/クラス '{name}' のメタタイプ ({actual_meta_type.__name__}) が "
                    f"期待されるメタタイプ ({expected_meta_type.__name__}) と一致しません。"
                )
                return None
    elif not (isinstance(target_type, type) or is_enum_meta_type): # expected_meta_typeがNoneの場合
        _logger_ggt.debug(
            f"グローバルスコープで見つかった '{name}' は型/クラスではありません "
            f"(実際の型: {actual_meta_type.__name__})。"
        )
        return None
    _logger_ggt.debug(f"グローバルスコープから型/クラス '{name}' を正常に取得しました。")
    return target_type

# =============================================================================
# -- Part 1a: Global Constants & Basic Utility Functions (v4.9α - Gemini最適化・修正版)
# =============================================================================

# --- グローバル定数 (Part 0 で定義済みのフラグ以外) ---
SYSTEM_VERSION_INFO: Final[str] = "NDGS v4.9α (Optimized)" # このスクリプトのバージョン情報
DEFAULT_ENCODING: Final[str] = "utf-8" # ファイルI/Oのデフォルトエンコーディング

# --- ログ関連の定数 (setup_loggingのデフォルト引数で使用) ---
# Bootstrap logger 用 (setup_logging が初回呼び出しより前にログ出力する場合のフォールバック)
LOG_FILENAME_DEFAULT_BOOTSTRAP: Final[str] = "ndgs_v49_alpha_bootstrap.log"
LOG_MAX_BYTES_DEFAULT_BOOTSTRAP: Final[int] = 1 * 1024 * 1024  # 1MB
LOG_BACKUP_COUNT_DEFAULT_BOOTSTRAP: Final[int] = 1

# 通常のロギング用 (AppConfigV49のクラス変数やインスタンス変数で上書きされる想定)
LOG_FILENAME_DEFAULT: Final[str] = "ndgs_v49_alpha_main.log" # setup_loggingのデフォルトファイル名
LOG_MAX_BYTES_DEFAULT: Final[int] = 10 * 1024 * 1024 # 10MB (setup_loggingのデフォルト)
LOG_BACKUP_COUNT_DEFAULT: Final[int] = 5 # (setup_loggingのデフォルト)

# Part 0 で try-except import された結果をここでグローバル変数として明確化
# (これにより、以降のコードでこれらのフラグを安全に参照できる)
# 注意: これらのフラグは Part 0 の最後で globals().get() を使って再代入されるため、
# ここでの初期値は実質的に無視される。ただし、型ヒントのためには有効。
_PYDANTIC_AVAILABLE: bool = globals().get('PYDANTIC_AVAILABLE', False)
_YAML_AVAILABLE: bool = globals().get('YAML_AVAILABLE', False)
_FILELOCK_AVAILABLE: bool = globals().get('FILELOCK_AVAILABLE', False)
_TQDM_AVAILABLE: bool = globals().get('TQDM_AVAILABLE', False)
_JSONSCHEMA_AVAILABLE: bool = globals().get('JSONSCHEMA_AVAILABLE', False)
_SPACY_AVAILABLE: bool = globals().get('SPACY_AVAILABLE', False)
_TRANSFORMERS_AVAILABLE: bool = globals().get('TRANSFORMERS_AVAILABLE', False)
_TORCH_AVAILABLE: bool = globals().get('TORCH_AVAILABLE', False)
_SKLEARN_AVAILABLE: bool = globals().get('SKLEARN_AVAILABLE', False)
_SCIPY_AVAILABLE: bool = globals().get('SCIPY_AVAILABLE', False)
_NUMPY_AVAILABLE: bool = globals().get('NUMPY_AVAILABLE', False)
_GOOGLE_API_AVAILABLE: bool = globals().get('GOOGLE_API_AVAILABLE', False)


# --- 基本的なユーティリティ関数 ---
_part1a_utils_logger = logging.getLogger(f"{__name__}.Part1a_Utils")

def load_json_file(file_path: Union[str, pathlib.Path], expected_type: Optional[Type] = None) -> Optional[Any]:
    p = pathlib.Path(file_path).resolve()
    _part1a_utils_logger.debug(f"JSONファイル '{p}' のロード試行...")
    if not p.is_file():
        _part1a_utils_logger.error(f"JSONファイル '{p}' が見つかりません。")
        return None
    try:
        with p.open('r', encoding=DEFAULT_ENCODING) as f: data = json.load(f)
        if expected_type is not None and not isinstance(data, expected_type):
            _part1a_utils_logger.error(f"JSON '{p}' 内容型不一致 (期待: {expected_type.__name__}, 実際: {type(data).__name__})。")
            return None
        _part1a_utils_logger.debug(f"JSONファイル '{p}' ロード成功。")
        return data
    except json.JSONDecodeError as e: _part1a_utils_logger.error(f"JSON '{p}' デコード失敗: {e}", exc_info=False)
    except IOError as e_io: _part1a_utils_logger.error(f"JSON '{p}' 読込中I/Oエラー: {e_io}", exc_info=False)
    except Exception as e_other: _part1a_utils_logger.error(f"JSON '{p}' 読込中予期せぬエラー: {e_other}", exc_info=True)
    return None

def save_json_file(data: Any, file_path: Union[str, pathlib.Path], indent: Optional[int] = 2, create_backup_first: bool = False) -> bool:
    p = pathlib.Path(file_path).resolve()
    _part1a_utils_logger.debug(f"JSONファイル '{p}' への保存試行...")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if create_backup_first and p.exists():
            create_backup_file_v49(str(p))
        with p.open('w', encoding=DEFAULT_ENCODING) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
        _part1a_utils_logger.info(f"JSONデータ '{p}' への保存成功。")
        return True
    except Exception as e:
        _part1a_utils_logger.error(f"JSONファイル '{p}' 書き込み中にエラー: {e}", exc_info=True)
        return False

def create_backup_file_v49(file_path_str: str, backup_subdir_name: str = "_archive_backups", max_backups_to_keep: int = 3) -> Optional[str]:
    try:
        original_file = pathlib.Path(file_path_str).resolve()
        if not original_file.exists():
            _part1a_utils_logger.debug(f"バックアップ対象ファイル '{original_file}' が存在しないためスキップします。")
            return None

        backup_dir = original_file.parent / backup_subdir_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        backup_filename = f"{original_file.stem}_{timestamp_str}{original_file.suffix}"
        backup_path = backup_dir / backup_filename

        shutil.copy2(original_file, backup_path)
        _part1a_utils_logger.info(f"ファイルのバックアップを作成しました: '{original_file.name}' -> '{backup_path.name}'")

        existing_backups = sorted(
            [f for f in backup_dir.glob(f"{original_file.stem}_*{original_file.suffix}") if f.is_file()],
            key=os.path.getmtime
        )
        
        num_backups_to_delete = len(existing_backups) - max_backups_to_keep
        if num_backups_to_delete > 0:
            for old_backup_file_to_delete in existing_backups[:num_backups_to_delete]:
                try:
                    old_backup_file_to_delete.unlink()
                    _part1a_utils_logger.info(f"古いバックアップファイルを削除しました: '{old_backup_file_to_delete.name}'")
                except Exception as e_unlink:
                    _part1a_utils_logger.warning(f"古いバックアップファイル '{old_backup_file_to_delete.name}' の削除中にエラー: {e_unlink}")
        return str(backup_path)
    except Exception as e_backup:
        _part1a_utils_logger.error(f"ファイル '{file_path_str}' のバックアップ作成中にエラーが発生しました: {e_backup}", exc_info=True)
        return None

_global_main_logger_instance: Optional[logging.Logger] = None

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, pathlib.Path]] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
    console_debug_mode: bool = False,
    force_reload_handlers: bool = False
) -> logging.Logger:
    global _global_main_logger_instance
    root_logger = logging.getLogger()

    if root_logger.hasHandlers() and not force_reload_handlers and _global_main_logger_instance:
        _part1a_utils_logger.debug("ロガーは既に初期化済みです。既存のインスタンスを返します。")
        return _global_main_logger_instance

    if root_logger.hasHandlers():
        _part1a_utils_logger.debug("既存のロガーハンドラをクリアします...")
        for handler in root_logger.handlers[:]:
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e_close_handler:
                _part1a_utils_logger.warning(f"既存ハンドラのクローズ/削除中にエラー: {e_close_handler}")
    
    log_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03dZ [%(levelname)8s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s',
        '%Y-%m-%dT%H:%M:%S'
    )
    log_formatter.converter = time.gmtime

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_output_level = level if console_debug_mode else max(level, logging.INFO)
    console_handler.setLevel(console_output_level)
    root_logger.addHandler(console_handler)

    file_handler: Optional[logging.handlers.RotatingFileHandler] = None
    # デフォルト値を LOG_FILENAME_DEFAULT_BOOTSTRAP から LOG_FILENAME_DEFAULT に変更
    actual_log_file_path_str = str(log_file if log_file else LOG_FILENAME_DEFAULT)
    actual_max_bytes = int(max_bytes if max_bytes is not None else LOG_MAX_BYTES_DEFAULT)
    actual_backup_count = int(backup_count if backup_count is not None else LOG_BACKUP_COUNT_DEFAULT)

    if not console_debug_mode:
        try:
            log_file_path_obj = pathlib.Path(actual_log_file_path_str).resolve()
            log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

            can_write_to_log = os.access(log_file_path_obj.parent, os.W_OK) and \
                               (not log_file_path_obj.exists() or os.access(log_file_path_obj, os.W_OK))

            if can_write_to_log:
                file_handler = logging.handlers.RotatingFileHandler(
                    str(log_file_path_obj),
                    maxBytes=actual_max_bytes,
                    backupCount=actual_backup_count,
                    encoding=DEFAULT_ENCODING
                )
                file_handler.setFormatter(log_formatter)
                file_handler.setLevel(level)
                root_logger.addHandler(file_handler)
                print(f"情報: ログを '{log_file_path_obj}' に出力します (ファイルログレベル: {logging.getLevelName(level)})。")
            else:
                print(f"警告: ログファイル '{log_file_path_obj}' への書き込み権限がありません。ファイルログは無効になります。", file=sys.stderr)
        except Exception as e_file_handler:
            print(f"警告: ファイルログハンドラの設定中にエラーが発生しました: {e_file_handler}。ファイルログは無効になります。", file=sys.stderr)
    
    root_logger.setLevel(min(level, console_output_level))

    libraries_to_suppress = {'urllib3': logging.WARNING, 'googleapiclient': logging.WARNING, 'filelock': logging.WARNING}
    for lib_name_suppress, lib_log_level_suppress in libraries_to_suppress.items():
        try:
            logging.getLogger(lib_name_suppress).setLevel(lib_log_level_suppress)
        except Exception:
            pass
        
    _global_main_logger_instance = logging.getLogger(__name__)
    
    console_level_name = logging.getLevelName(console_handler.level)
    file_level_name = logging.getLevelName(file_handler.level) if file_handler and file_handler in root_logger.handlers else "N/A"
    _global_main_logger_instance.info(f"ロガー設定完了 - RootLevel: {logging.getLevelName(root_logger.level)}, Console: {console_level_name}, File: {file_level_name}")
    return _global_main_logger_instance

logger: logging.Logger = setup_logging(
    level=logging.INFO,
    log_file=LOG_FILENAME_DEFAULT_BOOTSTRAP, # 初期はブートストラップログ
    max_bytes=LOG_MAX_BYTES_DEFAULT_BOOTSTRAP,
    backup_count=LOG_BACKUP_COUNT_DEFAULT_BOOTSTRAP,
    console_debug_mode=False
)

def create_argument_parser_v49() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{SYSTEM_VERSION_INFO} - 対話生成システム",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    input_group = parser.add_argument_group('Input/Output Files')
    config_group = parser.add_argument_group('Configuration Overrides')
    execution_group = parser.add_argument_group('Execution Control')

    input_group.add_argument(
        "data_file", nargs='?', default=None,
        help="入力キャラクター・シーン設定JSONファイルパス。"
    )
    input_group.add_argument(
        "--data", dest="data_file_opt", type=str, default=None,
        help="入力JSONファイルパス (--data オプション形式)。`data_file` と同じ効果。"
    )
    input_group.add_argument(
        "--out-dir", dest="appconfig_base_output_dir", type=str, default=None,
        help="ベース出力ディレクトリ。未指定時はYAML設定または内部デフォルト値に従います。"
    )

    config_group.add_argument(
        "--job-id", type=str, default=None,
        help="ジョブID。未指定時はタイムスタンプベースで自動生成されます。"
    )
    config_group.add_argument(
        "--log-level", type=str, default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="ログレベル。未指定時はYAML設定または内部デフォルト値 (INFO) に従います。"
    )
    config_group.add_argument(
        "--log-file", type=str, default=None,
        help="ログファイル名。未指定時はYAML設定または内部デフォルト値に従います。"
    )
    config_group.add_argument(
        "--model", dest="appconfig_default_model", type=str, default=None,
        help="使用する生成モデル名。未指定時はYAML設定または内部デフォルト値に従います。"
    )
    config_group.add_argument(
        "--length", type=int, default=None,
        help="目標対話長。未指定時はYAML設定または内部デフォルト値に従います。"
    )
    config_group.add_argument(
        "--loops", type=int, default=None, dest="settings_feedback_loops_override",
        help="フィードバックループ最大回数。未指定時はYAML設定または内部デフォルト値に従います。"
    )
    
    execution_group.add_argument(
        "--report", type=str, choices=['none', 'light', 'full', 'markdown'], default='markdown',
        help="最終レポートの形式。"
    )
    execution_group.add_argument(
        "--check-only", action="store_true",
        help="入力データの検証のみを実行し、対話生成は行いません。"
    )
    execution_group.add_argument(
        "--run-tests", action="store_true",
        help="組み込みのユニットテストを実行します。"
    )
    execution_group.add_argument(
        "--clear-cache", dest="appconfig_cache_cleanup_on_start", action=argparse.BooleanOptionalAction, default=None,
        help="起動時にキャッシュをクリアします。未指定時はYAML設定または内部デフォルト値に従います。"
    )
    execution_group.add_argument(
        "--cache-vacuum", dest="appconfig_cache_vacuum_on_clear", action=argparse.BooleanOptionalAction, default=None,
        help="キャッシュクリア時にVACUUMを実行します。未指定時はYAML設定または内部デフォルト値に従います。"
    )
    return parser
# =============================================================================
# -- Part 1a 終了点
# =============================================================================
# =============================================================================
# -- Part 1b: Core Enum Definitions (v4.9α - ScoreKeys.LLM修正版 v1b.1)
# =============================================================================
# v1b.1 Update:
# - ScoreKeys.LLM._missing_ メソッドを修正。未知のキー文字列が渡された場合に
#   UNKNOWNメンバーを返すのではなく、ValueErrorを送出するように変更。
#   これにより、Part 12の _select_final_version でのキー由来判定がより厳密になり、
#   DFRSキーが誤ってLLMキーとして解釈される問題を解消する。
# - 他のEnumの _missing_ メソッドは変更なし。
# - ログ出力やエイリアスマップのロジックは基本的に維持。

import enum
import logging
import sys # EmotionalToneV49 のロガー設定で使用
import time # EmotionalToneV49 のロガー設定で使用
from typing import TYPE_CHECKING, TypeVar, Dict, Optional, Any, Type

# --- 型ヒント用の設定 ---
if TYPE_CHECKING:
    # このブロック内のインポートは型チェック時にのみ有効
    pass

# --- Enumの基底となる型変数を定義 ---
TEnum = TypeVar('TEnum', bound=enum.Enum)

class PsychologicalPhaseV49(str, enum.Enum):
    INTRODUCTION = "introduction"
    DEVELOPMENT = "development"
    CONFLICT_CRISIS = "conflict_crisis"
    CLIMAX_TURNING_POINT = "climax_turning_point"
    RESOLUTION_CONCLUSION = "resolution_conclusion"
    INTERNAL_PROCESSING = "internal_processing"
    EMOTIONAL_FOCUS = "emotional_focus"
    MEMORY_FOCUS = "memory_focus"
    ACTION_EVENT = "action_event"
    SIGNIFICANT_STATE = "significant_state"
    UNKNOWN = "unknown"

    _logger_instance_pp: Optional[logging.Logger] = None
    _alias_map_cache_pp: Optional[Dict[str, 'PsychologicalPhaseV49']] = None

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        if cls._logger_instance_pp is None:
            cls._logger_instance_pp = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        return cls._logger_instance_pp

    @classmethod
    def _get_minimal_alias_map(cls) -> Dict[str, 'PsychologicalPhaseV49']:
        if cls._alias_map_cache_pp is None:
            cls._alias_map_cache_pp = {
                "intro": cls.INTRODUCTION, "climax": cls.CLIMAX_TURNING_POINT,
                "resolution": cls.RESOLUTION_CONCLUSION, "conclusion": cls.RESOLUTION_CONCLUSION,
            }
        return cls._alias_map_cache_pp

    @classmethod
    def _missing_(cls, value: Any) -> Optional['PsychologicalPhaseV49']:
        logger = cls._get_logger()
        if not isinstance(value, str):
            logger.warning(f"非文字列値 '{value!r}' (型: {type(value)}) が指定されました。'{cls.UNKNOWN.name}' を返します。")
            return cls.UNKNOWN
        if not value.strip():
            logger.warning(f"空または空白のみの文字列値が指定されました。'{cls.UNKNOWN.name}' を返します。")
            return cls.UNKNOWN
        original_input = value
        normalized_value = value.lower().strip().replace(' ', '_').replace('-', '_')
        for member in cls:
            if member.value == normalized_value:
                return member
        enum_name_candidate = normalized_value.upper()
        if hasattr(cls, enum_name_candidate) and isinstance(getattr(cls, enum_name_candidate), cls):
            return getattr(cls, enum_name_candidate) # type: ignore
        alias_map = cls._get_minimal_alias_map()
        if normalized_value in alias_map:
            return alias_map[normalized_value]
        logger.warning(f"値 '{original_input}' (正規化後: '{normalized_value}') はどのメンバー、メンバー名、エイリアスにもマップできませんでした。'{cls.UNKNOWN.name}' を返します。")
        return cls.UNKNOWN

# EmotionalToneV49 とそのヘルパー
ET = TypeVar('ET', bound='EmotionalToneV49')

class EmotionalToneV49(str, enum.Enum):
    HAPPY = "happy"; SAD = "sad"; ANGRY = "angry"; FEARFUL = "fearful"; SURPRISED = "surprised"
    DISGUSTED = "disgusted"; JOYFUL = "joyful"; EXCITED = "excited"; ECSTATIC = "ecstatic"
    CONTENT = "content"; FULFILLED = "fulfilled"; PLEASED = "pleased"; RELIEVED = "relieved"
    CALM = "calm"; SERENE = "serene"; HOPEFUL = "hopeful"; OPTIMISTIC = "optimistic"
    GRATEFUL = "grateful"; THANKFUL = "thankful"; TOUCHED = "touched"; BLISSFUL = "blissful"
    TRIUMPHANT = "triumphant"; PLAYFUL = "playful"; AFFECTIONATE = "affectionate"; LOVING = "loving"
    PROUD = "proud"; CONFIDENT = "confident"; AMUSED = "amused"; INSPIRED = "inspired"
    GLOOMY = "gloomy"; DEJECTED = "dejected"; MELANCHOLIC = "melancholic"; WISTFUL = "wistful"
    IRRITATED = "irritated"; FRUSTRATED = "frustrated"; RESENTFUL = "resentful"; ANXIOUS = "anxious"
    NERVOUS = "nervous"; TENSE = "tense"; WORRIED = "worried"; APPREHENSIVE = "apprehensive"
    DREADFUL = "dreadful"; EMBARRASSED = "embarrassed"; ASHAMED = "ashamed"; GUILTY = "guilty"
    DISAPPOINTED = "disappointed"; HOLLOW = "hollow"; NUMB = "numb"; OVERWHELMED = "overwhelmed"
    RESIGNED = "resigned"; DESPAIRING = "despairing"; PAINED = "pained"; STRESSED = "stressed"
    BITTER = "bitter"; CYNICAL = "cynical"; DEFENSIVE = "defensive"; JEALOUS = "jealous"
    ENVIOUS = "envious"; CONFUSED = "confused"; CURIOUS = "curious"; SKEPTICAL = "skeptical"
    SARCASTIC = "sarcastic"; NOSTALGIC = "nostalgic"; SENTIMENTAL = "sentimental"
    BITTERSWEET = "bittersweet"; POIGNANT = "poignant"; AWESTRUCK = "awestruck"
    IMPRESSED = "impressed"; DETERMINED = "determined"; SMUG = "smug"
    CONSIDERATE = "considerate"; COMPASSIONATE = "compassionate"; ADMIRING = "admiring"
    INTENSE = "intense"; SUBTLE = "subtle"; PASSIONATE = "passionate"; TIRED = "tired"
    PESSIMISTIC = "pessimistic"; SHOCKED = "shocked"; DEFIANT = "defiant"
    NEUTRAL = "neutral"; UNKNOWN = "unknown"

    _logger_instance_et: Optional[logging.Logger] = None
    _alias_map_cache_et: Optional[Dict[str, 'EmotionalToneV49']] = None

    @classmethod
    def _get_logger(cls: Type[ET]) -> logging.Logger:
        if cls._logger_instance_et is None:
            cls._logger_instance_et = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        return cls._logger_instance_et

    @classmethod
    def _get_alias_map(cls: Type[ET]) -> Dict[str, ET]:
        if cls._alias_map_cache_et is None:
            cls._alias_map_cache_et = {
                "joy": cls.JOYFUL, "sadness": cls.SAD, "anger": cls.ANGRY, "fear": cls.FEARFUL,
                # ... (提供されたエイリアスリストの残りをここに追加) ...
                "hostile": cls.ANGRY, "suspicious": cls.SKEPTICAL, "thoughtful": cls.CONSIDERATE,
                "lighthearted": cls.PLAYFUL, "empathetic": cls.COMPASSIONATE,
                "vulnerable": cls.SAD, "chaotic": cls.OVERWHELMED, "evasive": cls.DEFENSIVE,
                "firm": cls.DETERMINED, "quiet": cls.CALM,
            }
            # logger = cls._get_logger()
            # logger.info(f"EmotionalToneV49 エイリアスマップ生成・キャッシュ完了。登録数: {len(cls._alias_map_cache_et)}")
        return cls._alias_map_cache_et

    @classmethod
    def _missing_(cls: Type[ET], value: Any) -> Optional[ET]:
        logger = cls._get_logger()
        if not isinstance(value, str):
            logger.warning(f"非文字列値 '{value!r}' (型: {type(value)})。'{cls.UNKNOWN.name}' を返します。")
            return cls.UNKNOWN
        if not value.strip():
            logger.warning(f"空または空白のみの文字列値。'{cls.UNKNOWN.name}' を返します。")
            return cls.UNKNOWN
        original_input = value
        normalized_value_for_alias = value.lower().strip().replace('_', ' ')
        normalized_value_for_member = value.lower().strip().replace(' ', '_').replace('-', '_')
        for member in cls:
            if member.value == normalized_value_for_member: return member
        enum_name_candidate = normalized_value_for_member.upper()
        if hasattr(cls, enum_name_candidate) and isinstance(getattr(cls, enum_name_candidate), cls):
            return getattr(cls, enum_name_candidate) # type: ignore
        alias_map = cls._get_alias_map()
        if normalized_value_for_alias in alias_map:
            return alias_map[normalized_value_for_alias]
        logger.warning(f"値 '{original_input}' はどのメンバー、メンバー名、エイリアスにもマップ不可。'{cls.UNKNOWN.name}' を返します。")
        return cls.UNKNOWN

# SubjectivityCategoryV49, FluctuationCategoryV49, DFRSMetricsV49 は v1b.0 と同様なので省略 (ただし、_missing_ の実装は上記と同様のパターンを推奨)
# ... (SubjectivityCategoryV49 の定義) ...
# ... (FluctuationCategoryV49 の定義) ...
# ... (DFRSMetricsV49 の定義) ...
class SubjectivityCategoryV49(str, enum.Enum):
    EMOTION_POSITIVE = "emotion_positive"; EMOTION_NEGATIVE = "emotion_negative"
    EMOTION_SURPRISE_CURIOSITY = "emotion_surprise_curiosity"; EMOTION_OTHER = "emotion_other"
    PHYSICAL_REACTION = "physical_reaction"; SENSATION_PERCEPTION = "sensation_perception"
    EMOTIONAL_ACTION_EXPRESSION = "emotional_action_expression"; ATTITUDE_BEHAVIOR = "attitude_behavior"
    RELATIONSHIP_COMMUNICATION = "relationship_communication"; EVALUATION_JUDGMENT = "evaluation_judgment"
    MODALITY_DEGREE_CERTAINTY = "modality_degree_certainty"; INTROSPECTION_COGNITION = "introspection_cognition"
    METAPHORICAL_SUBJECTIVITY = "metaphorical_figurative_subjectivity"
    INTERACTION_DISCOURSE_MARKERS = "interaction_discourse_markers"
    FIRST_PERSON_EMPHASIS_MARKERS = "first_person_emphasis_markers"
    SELF_AWARENESS_IDENTITY = "self_awareness_identity"; UNKNOWN = "unknown"
    @classmethod
    def _missing_(cls, value: Any) -> Optional['SubjectivityCategoryV49']: return cls.UNKNOWN # 簡易版

class FluctuationCategoryV49(str, enum.Enum):
    VERBAL_HESITATION = "verbal_hesitation"; FILLER = "filler"; RESTART_REPHRASE = "restart_rephrase"
    STAMMERING = "stammering"; PAUSE_SILENCE = "pause_silence"; TEMPO_RHYTHM_CHANGE = "tempo_rhythm_change"
    STRUCTURAL_BREAK = "structural_break"; MISCELLANEOUS = "miscellaneous"; UNKNOWN = "unknown"
    @classmethod
    def _missing_(cls, value: Any) -> Optional['FluctuationCategoryV49']: return cls.UNKNOWN # 簡易版

class DFRSMetricsV49(str, enum.Enum):
    CSR = "csr"; DIT = "dit"; PVS = "pvs"; DDB = "ddb"; TS = "ts"
    PHASE_ALIGNMENT = "phase_alignment"; TONE_ALIGNMENT = "tone_alignment"; PTN = "ptn"; ECS = "ecs"
    RHYTHM_CONSISTENCY = "rhythm_consistency"; RHYTHM_VARIABILITY = "rhythm_variability"
    SUBJECTIVITY_SCORE = "subjectivity_score"; FLUCTUATION_INTENSITY = "fluctuation_intensity"
    INTERNAL_DEPTH = "internal_depth"; EMOTION_COMPLEXITY = "emotion_complexity"; ETI = "eti"
    SYMBOLIC_DENSITY = "symbolic_density"; CONTENT_NOVELTY = "content_novelty"
    EXPRESSION_RICHNESS = "expression_richness"; FINAL_EODF_V49 = "final_eodf_v49"; UNKNOWN = "unknown"
    @classmethod
    def _missing_(cls, value: Any) -> Optional['DFRSMetricsV49']: return cls.UNKNOWN # 簡易版

class ScoreKeys:
    class LLM(str, enum.Enum):
        CONSISTENCY = "consistency"
        NATURALNESS = "naturalness"
        EMOTIONAL_DEPTH = "emotionalDepth" # LLM応答のキーはキャメルケースの場合があるので注意
        CONSTRAINTS = "constraints"
        ATTRACTIVENESS = "attractiveness"
        COMPLEXITY = "complexity"
        OVERALL = "overall" # Part12 のログで 'overall' がキーとして使われている
        UNKNOWN = "unknown_llm_score" # _missing_ から返される場合の値

        _logger_instance_sk_llm: Optional[logging.Logger] = None

        @classmethod
        def _get_logger(cls) -> logging.Logger:
            if cls._logger_instance_sk_llm is None:
                cls._logger_instance_sk_llm = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
            return cls._logger_instance_sk_llm

        @classmethod
        def _missing_(cls, value: Any) -> 'ScoreKeys.LLM': # 修正: 未知の値の場合はValueErrorを送出
            logger = cls._get_logger()
            if isinstance(value, cls):
                return value # 既に正しいEnumメンバー

            if not isinstance(value, str):
                logger.error(f"ScoreKeys.LLM: 非文字列値 '{value!r}' (型: {type(value)}) でEnumを解決しようとしました。")
                raise ValueError(f"ScoreKeys.LLM は文字列値からのみ解決可能です。入力値: {value!r}")
            
            if not value.strip():
                logger.error(f"ScoreKeys.LLM: 空または空白のみの文字列値でEnumを解決しようとしました。")
                raise ValueError("ScoreKeys.LLM は空または空白のみの文字列からは解決できません。")

            original_input = str(value) # valueがAnyなのでstrにキャスト
            # LLMキーは様々な形式で来る可能性があるため、正規化を試みる
            # 一般的には settings.final_selection_weights のキー (FinalSelectionKeysV49.value) がそのまま使われる
            normalized_value = original_input.strip().lower() # 基本的な正規化

            for member in cls:
                if member.value.lower() == normalized_value: # メンバーの .value も小文字化して比較
                    # logger.debug(f"ScoreKeys.LLM: 値 '{original_input}' はメンバー値 '{member.value}' に一致。'{member.name}' を返します。")
                    return member
            
            # メンバー名 (大文字スネークケース) との比較
            enum_name_candidate = original_input.strip().upper().replace('-', '_').replace(' ', '_')
            if hasattr(cls, enum_name_candidate) and isinstance(getattr(cls, enum_name_candidate), cls):
                member_found_by_name = getattr(cls, enum_name_candidate)
                # logger.debug(f"ScoreKeys.LLM: 値 '{original_input}' はメンバー名 '{member_found_by_name.name}' に一致。'{member_found_by_name.name}' を返します。")
                return member_found_by_name # type: ignore

            # ここまで来たら、どのメンバーにも一致しなかった
            logger.warning(f"ScoreKeys.LLM: 値 '{original_input}' はどの定義済みメンバーにもマップできませんでした。ValueErrorを送出します。")
            raise ValueError(f"無効な ScoreKeys.LLM 値: '{original_input}'。有効な値は {[m.value for m in cls if m != cls.UNKNOWN]} です。")
            # UNKNOWNを返す代わりにValueErrorを送出することで、Part12のtry-exceptが機能するようにする

    DFRS = DFRSMetricsV49

    class SelectionKeys(str, enum.Enum): # 以前は ScoreKeys.LLM.OVERALL.value などを使っていたが、循環参照や複雑化を避けるため、直接文字列を指定
        LLM_OVERALL = "overall"
        LLM_CONSISTENCY = "consistency"
        LLM_COMPLEXITY = "complexity"
        # DFRS系のキーは、DFRSMetricsV49のvalueと一致させる
        DFRS_FINAL = "final_eodf_v49"
        DFRS_SUBJECTIVITY = "subjectivity_score"
        DFRS_RICHNESS = "expression_richness"
        UNKNOWN = "unknown_selection_key"

        _logger_instance_sk_sel: Optional[logging.Logger] = None
        @classmethod
        def _get_logger(cls) -> logging.Logger:
            if cls._logger_instance_sk_sel is None: cls._logger_instance_sk_sel = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
            return cls._logger_instance_sk_sel
        @classmethod
        def _missing_(cls, value: Any) -> Optional['ScoreKeys.SelectionKeys']:
            logger = cls._get_logger()
            if isinstance(value, cls): return value
            if not isinstance(value, str): logger.warning(f"非文字列値 '{value!r}'。'{cls.UNKNOWN.name}' を返します。"); return cls.UNKNOWN
            if not value.strip(): logger.warning(f"空文字列。'{cls.UNKNOWN.name}' を返します。"); return cls.UNKNOWN
            original_input = value; normalized_value = value.lower().strip().replace('-', '_').replace(' ', '_')
            for member in cls:
                if member.value.lower() == normalized_value: return member
            enum_name_candidate = normalized_value.upper()
            if hasattr(cls, enum_name_candidate) and isinstance(getattr(cls, enum_name_candidate), cls): return getattr(cls, enum_name_candidate) # type: ignore
            logger.warning(f"値 '{original_input}' をマップできませんでした。'{cls.UNKNOWN.name}' を返します。")
            return cls.UNKNOWN

class SubjectiveIntensityLevel(str, enum.Enum):
    OFF = "off"; LOW = "low"; MEDIUM = "medium"; HIGH = "high"; EXTREME = "extreme"; UNKNOWN = "unknown"
    @classmethod
    def _missing_(cls, value: Any) -> Optional['SubjectiveIntensityLevel']: return cls.UNKNOWN # 簡易版

class FinalSelectionKeysV49(str, enum.Enum):
    LLM_OVERALL = ScoreKeys.LLM.OVERALL.value
    LLM_CONSISTENCY = ScoreKeys.LLM.CONSISTENCY.value
    LLM_COMPLEXITY = ScoreKeys.LLM.COMPLEXITY.value
    DFRS_FINAL = ScoreKeys.DFRS.FINAL_EODF_V49.value # DFRSMetricsV49 の値と一致
    DFRS_SUBJECTIVITY = ScoreKeys.DFRS.SUBJECTIVITY_SCORE.value
    DFRS_RICHNESS = ScoreKeys.DFRS.EXPRESSION_RICHNESS.value
    UNKNOWN = "unknown_final_selection_key"

    @classmethod
    def _missing_(cls, value: Any) -> Optional['FinalSelectionKeysV49']: # 既存のロジックをベースに
        logger = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        if isinstance(value, cls): return value
        if not isinstance(value, str): logger.warning(f"非文字列値 '{value!r}'。'{cls.UNKNOWN.name}' を返します。"); return cls.UNKNOWN
        val_norm = value.lower().strip().replace('-', '_').replace(' ', '_')
        for member in cls:
            if member.value.lower() == val_norm: return member
        name_cand = val_norm.upper()
        if hasattr(cls, name_cand) and isinstance(getattr(cls, name_cand), cls): return getattr(cls, name_cand) # type: ignore
        logger.warning(f"値 '{value}' をマップできませんでした。'{cls.UNKNOWN.name}' を返します。")
        return cls.UNKNOWN

class InitialSelectionKeysV49(str, enum.Enum):
    OVERALL = ScoreKeys.LLM.OVERALL.value
    EMOTIONAL_DEPTH = ScoreKeys.LLM.EMOTIONAL_DEPTH.value
    EXPRESSION_RICHNESS = ScoreKeys.DFRS.EXPRESSION_RICHNESS.value
    CONTENT_NOVELTY = ScoreKeys.DFRS.CONTENT_NOVELTY.value
    UNKNOWN = "unknown_initial_selection_key"

    @classmethod
    def _missing_(cls, value: Any) -> Optional['InitialSelectionKeysV49']: # 既存のロジックをベースに
        logger = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        if isinstance(value, cls): return value
        if not isinstance(value, str): logger.warning(f"非文字列値 '{value!r}'。'{cls.UNKNOWN.name}' を返します。"); return cls.UNKNOWN
        val_norm = value.lower().strip().replace('-', '_').replace(' ', '_')
        for member in cls:
            if member.value.lower() == val_norm: return member
        name_cand = val_norm.upper()
        if hasattr(cls, name_cand) and isinstance(getattr(cls, name_cand), cls): return getattr(cls, name_cand) # type: ignore
        logger.warning(f"値 '{value}' をマップできませんでした。'{cls.UNKNOWN.name}' を返します。")
        return cls.UNKNOWN

# =============================================================================
# -- Part 1b 終了点 (: Core Enum Definitions クラス定義終了)
# =============================================================================
# -- Part 1 終了点
# (Part 0, Part 1a, Part 1b の定義がこの前にあることを強く推奨)
# 必要なインポート (例: typing, pathlib, os, sys, logging, inspect, enum, pydantic) は Part 0 にあるものとします。
from typing import Optional, Dict, Any, List, Union, TypeAlias, Literal, Type, Callable # Part 0 から
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator, RootModel # Part 0 から
import pathlib # Part 0 から
import os # Part 0 から
import sys # Part 0 から
import logging # Part 0 から
import inspect # Part 0 から
import enum # Part 0 から
from datetime import datetime, timezone # Part 0 から

# =============================================================================
# -- Part 2: Config Class & Utilities (v4.9α - 最適化・FIX版)
# =============================================================================

# --- グローバルスコープでの型エイリアス定義 (AppConfigV49 より前に必要なもの) ---
# (もしAppConfigV49がこれらに依存する場合。通常はAppConfigV49内部の型ヒントで文字列リテラルを使う)
if TYPE_CHECKING:
    # 基本的な型や、他の多くの場所で再利用される可能性のあるエイリアス
    CharacterInputType: TypeAlias = Dict[str, Any]
    SceneInfoInputType: TypeAlias = Dict[str, Any]

    # --- Part 3で定義されるPydanticモデルへの前方参照 ---
    # これらは AppConfigV49 の型ヒントで使用するため、ここで定義しておく
    # 実行時は文字列リテラルとして扱われるように、else節も定義する
    ExternalConfigsV49Type = 'ExternalConfigsV49'
    SubjectivityKeywordsFileV49Type = 'SubjectivityKeywordsFileV49'
    FluctuationPatternsFileV49Type = 'FluctuationPatternsFileV49'
    TemperatureStrategyConfigV49Type = 'TemperatureStrategyConfigV49'
    AdaptationStrategyConfigV49Type = 'AdaptationStrategyConfigV49'
    FeedbackStrategyConfigV49Type = 'FeedbackStrategyConfigV49'
    FeatureFlagConfigV49Type = 'FeatureFlagConfigV49'
    APISettingsV49Type = 'APISettingsV49'
    FileSettingsV49Type = 'FileSettingsV49' # FileSettingsV49も前方参照対象

    # --- Part 1b で定義される Enum (型ヒントのため再掲、または直接参照) ---
    # from __main__ import DFRSMetricsV49 # __main__からのインポートは最終手段
    DFRSMetricsV49EnumType = 'DFRSMetricsV49' # 文字列リテラルで

    # --- Part 11で定義されるクラスへの前方参照 ---
    ExceptionManagerV49Type = 'ExceptionManagerV49'
else:
    # 実行時はこれらの型エイリアスは文字列リテラルとして機能する
    ExternalConfigsV49Type = 'ExternalConfigsV49'
    SubjectivityKeywordsFileV49Type = 'SubjectivityKeywordsFileV49'
    FluctuationPatternsFileV49Type = 'FluctuationPatternsFileV49'
    TemperatureStrategyConfigV49Type = 'TemperatureStrategyConfigV49'
    AdaptationStrategyConfigV49Type = 'AdaptationStrategyConfigV49'
    FeedbackStrategyConfigV49Type = 'FeedbackStrategyConfigV49'
    FeatureFlagConfigV49Type = 'FeatureFlagConfigV49'
    APISettingsV49Type = 'APISettingsV49'
    FileSettingsV49Type = 'FileSettingsV49'
    DFRSMetricsV49EnumType = 'DFRSMetricsV49'
    ExceptionManagerV49Type = 'ExceptionManagerV49'

# -----------------------------------------------------------------------------
# ヘルパー関数: _is_readonly_property (AppConfigV49クラス定義の前に配置)
# -----------------------------------------------------------------------------
def _is_readonly_property(cls: Type[Any], attr_name: str) -> bool:
    """指定された属性がセッターを持たない読み取り専用の@propertyであるかを判定します。"""
    if not inspect.isclass(cls):
        return False
    attr = getattr(cls, attr_name, None)
    return isinstance(attr, property) and attr.fset is None

# -----------------------------------------------------------------------------
# AppConfigV49 クラス定義
# -----------------------------------------------------------------------------
class AppConfigV49:
    """
    システム全体の設定を管理します (v4.9α - 最適化・FIX版)。
    デフォルト値をクラス変数として持ち、外部設定ファイルやCLI引数で上書き可能です。
    読み取り専用プロパティを通じて、動的に決定される設定値も提供します。
    """
    # --- クラス変数 (デフォルト値) ---
    SYSTEM_VERSION: str = "NDGS v4.9α (Optimized)"
    DEFAULT_MODEL: str = "models/gemini-2.5-flash-preview-04-17-thinking"
    API_KEY_ENV_VAR: str = "GENAI_API_KEY"
    RPM_LIMIT: int = 50
    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY: float = 1.5
    MAX_RETRY_DELAY: float = 45.0
    RATE_LIMIT_DELAY: float = 10.0
    API_TIMEOUT: int = 300
    EVALUATION_TEMPERATURE: float = 0.15
    INITIAL_CANDIDATE_COUNT: int = 3

    DEFAULT_BASE_OUTPUT_DIR_STR: str = "./output_v49_alpha_final_gemini_fixed_v2"
    CONFIG_DIR_STR: str = "./configs"
    RESOURCES_DIR_NAME_STR: str = "resources"
    CACHE_DIR_STR: str = "./cache/ndgs_v49_alpha_final_gemini_fixed_v2"
    PERSISTENT_CACHE_SUBDIR_NAME_DEFAULT: str = "persistent_v2"

    APP_CONFIG_YAML_FILENAME: str = "app_config_v49.yaml"
    SUBJECTIVITY_KEYWORDS_FILENAME: str = "subjectivity_keywords_v49.yaml"
    FLUCTUATION_PATTERNS_FILENAME: str = "fluctuation_patterns_v49.yaml"
    ANALYZER_KEYWORDS_FILENAME: str = "analyzer_keywords_v49.yaml"

    LOG_FILENAME_DEFAULT_STR: str = "ndgs_v49_alpha_final_fixed_v2.log"
    LOG_MAX_BYTES_DEFAULT_VAL: int = 25 * 1024 * 1024
    LOG_BACKUP_COUNT_DEFAULT_VAL: int = 10

    DEFAULT_JOB_ID_PREFIX: str = "ndgs49fix2_job_"
    FILENAME_MAX_LENGTH_DEFAULT: int = 150

    RESUME_DIR_NAME_DEFAULT: str = "resume_states"
    STATS_DIR_NAME_DEFAULT: str = "generation_stats"
    PROMPT_DIR_NAME_DEFAULT: str = "saved_prompts"
    EVAL_DIR_NAME_DEFAULT: str = "llm_evaluations"
    JSON_EXPORT_DIR_NAME_DEFAULT: str = "json_exports"
    REJECTED_DIR_NAME_DEFAULT: str = "rejected_dialogues"
    RL_MODEL_DIR_NAME_DEFAULT: str = "rl_training_data"

    CACHE_NLP_TABLE_NAME: str = "nlp_cache_v49_optimized_fix2"
    CACHE_DFRS_TABLE_NAME: str = "dfrs_cache_v49_optimized_fix2"

    LOCK_SUFFIX_DEFAULT: str = ".lockfile"
    RESUME_SUFFIX_DEFAULT: str = ".resume.json"
    STATS_FILENAME_DEFAULT: str = "ndgs_overall_stats_v49_fix2.jsonl"

    CACHE_CLEANUP_ON_START_DEFAULT: bool = False
    CACHE_VACUUM_ON_CLEAR_DEFAULT: bool = True
    EXTERNAL_CONFIG_ENABLED_DEFAULT: bool = True
    ENABLE_FILELOCK_DEFAULT: bool = getattr(sys.modules.get("__main__"), 'FILELOCK_AVAILABLE', True)

    logger: logging.Logger
    log_level_str: str
    log_filename: str
    log_max_bytes: int
    log_backup_count: int
    base_output_dir: pathlib.Path
    config_dir: pathlib.Path
    resources_dir: pathlib.Path
    cache_dir: pathlib.Path
    persistent_cache_dir: pathlib.Path
    filename_max_length: int
    enable_filelock: bool
    cache_cleanup_on_start: bool
    cache_vacuum_on_clear: bool
    external_config_enabled: bool

    loaded_external_configs: Optional['ExternalConfigsV49Type'] # type: ignore
    subjectivity_data: Optional['SubjectivityKeywordsFileV49Type'] # type: ignore
    fluctuation_data: Optional['FluctuationPatternsFileV49Type'] # type: ignore
    analyzer_keywords_data: Optional[Dict[str, Any]]

    _exception_manager: Optional['ExceptionManagerV49Type'] # type: ignore
    _external_config_loaded_successfully: bool

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        self.logger.info(f"AppConfigV49 (System: {self.SYSTEM_VERSION}) インスタンス初期化開始...")

        self.log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
        self.log_filename = self.LOG_FILENAME_DEFAULT_STR
        self.log_max_bytes = self.LOG_MAX_BYTES_DEFAULT_VAL
        self.log_backup_count = self.LOG_BACKUP_COUNT_DEFAULT_VAL

        self.base_output_dir = pathlib.Path(self.DEFAULT_BASE_OUTPUT_DIR_STR).resolve()
        self.config_dir = pathlib.Path(self.CONFIG_DIR_STR).resolve()
        self.resources_dir = (self.config_dir / self.RESOURCES_DIR_NAME_STR).resolve()
        self.cache_dir = pathlib.Path(self.CACHE_DIR_STR).resolve()
        self.persistent_cache_dir = (self.cache_dir / self.PERSISTENT_CACHE_SUBDIR_NAME_DEFAULT).resolve()

        self.filename_max_length = self.FILENAME_MAX_LENGTH_DEFAULT
        self.enable_filelock = self.ENABLE_FILELOCK_DEFAULT
        self.cache_cleanup_on_start = self.CACHE_CLEANUP_ON_START_DEFAULT
        self.cache_vacuum_on_clear = self.CACHE_VACUUM_ON_CLEAR_DEFAULT
        self.external_config_enabled = self.EXTERNAL_CONFIG_ENABLED_DEFAULT

        self.loaded_external_configs = None
        self.subjectivity_data = None
        self.fluctuation_data = None
        self.analyzer_keywords_data = None
        self._exception_manager = None
        self._external_config_loaded_successfully = False

        self.logger.debug(f"AppConfigV49 インスタンスの基本デフォルト設定完了。")

    @staticmethod
    def _ensure_dir_exists(dir_path: pathlib.Path) -> bool:
        logger_ensure = logging.getLogger(f"{AppConfigV49.__module__}.AppConfigV49._ensure_dir_exists")
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger_ensure.debug(f"ディレクトリ '{dir_path}' の存在確認/作成成功。")
            return True
        except OSError as e_os:
            logger_ensure.error(f"ディレクトリ '{dir_path}' の作成中にOSエラー ({e_os.errno}): {e_os.strerror}", exc_info=False)
        except Exception as e:
            logger_ensure.error(f"ディレクトリ '{dir_path}' の作成中に予期せぬエラー: {e}", exc_info=True)
        return False

    def _resolve_path_from_file_settings(
        self,
        path_input: Optional[Union[str, pathlib.Path]],
        default_segment: Optional[str] = None,
        base_directory: Optional[pathlib.Path] = None
    ) -> Optional[pathlib.Path]:
        final_path: Optional[pathlib.Path] = None
        if isinstance(path_input, pathlib.Path):
            final_path = path_input
        elif isinstance(path_input, str) and path_input.strip():
            final_path = pathlib.Path(path_input.strip())
        elif default_segment:
            current_base = base_directory if base_directory else pathlib.Path(".").resolve()
            final_path = current_base / default_segment

        if final_path:
            try:
                return final_path.resolve()
            except Exception as e_resolve:
                self.logger.warning(f"パス '{final_path}' の解決中にエラー: {e_resolve}")
                return None
        return None

    def load_external_configs(self) -> bool:
        if not self.external_config_enabled:
            self.logger.info("外部設定ファイルのロードは無効化されています。")
            self._external_config_loaded_successfully = False
            return False
        if self._external_config_loaded_successfully:
            self.logger.debug("外部設定ファイルは既にロード済みです。")
            return True
        if not YAML_AVAILABLE: # YAML_AVAILABLE は Part 0 で定義
            self.logger.critical("PyYAMLライブラリが見つからないため、外部設定ファイルをロードできません。")
            self._external_config_loaded_successfully = False
            return False

        self.logger.info("外部設定ファイルおよびリソースの読み込みを開始します...")
        if not self._ensure_dir_exists(self.config_dir) or \
           not self._ensure_dir_exists(self.resources_dir):
            self.logger.error("設定ディレクトリまたはリソースディレクトリの作成に失敗しました。")
            self._external_config_loaded_successfully = False
            return False

        # AppConfig YAML のロード
        app_config_yaml_path = self.config_dir / self.APP_CONFIG_YAML_FILENAME
        raw_app_cfg_data = load_yaml_file(app_config_yaml_path, expected_type=dict)

        ExternalConfigsV49_cls = globals().get('ExternalConfigsV49')
        FileSettingsV49_cls = globals().get('FileSettingsV49')

        if not ExternalConfigsV49_cls:
            self.logger.critical("ExternalConfigsV49 モデルクラスが見つかりません。外部設定を処理できません。")
            self._external_config_loaded_successfully = False
            return False

        if raw_app_cfg_data:
            try:
                ExternalConfigsV49_cls.model_rebuild(force=True) # type: ignore[attr-defined]
                loaded_cfg = ExternalConfigsV49_cls.model_validate(raw_app_cfg_data)
                self.loaded_external_configs = loaded_cfg
                self.logger.info(f"アプリケーション設定ファイル '{app_config_yaml_path}' の読み込みと検証が完了しました。")
                self._external_config_loaded_successfully = True

                if FileSettingsV49_cls and loaded_cfg.file_settings: # type: ignore[union-attr]
                    fs = loaded_cfg.file_settings # type: ignore[union-attr]
                    self.base_output_dir = self._resolve_path_from_file_settings(fs.base_output_dir, base_directory=self.config_dir) or self.base_output_dir
                    self.config_dir = self._resolve_path_from_file_settings(fs.config_dir, base_directory=pathlib.Path(".")) or self.config_dir
                    self.resources_dir = self._resolve_path_from_file_settings(fs.resources_dir, self.RESOURCES_DIR_NAME_STR, self.config_dir) or self.resources_dir
                    self.cache_dir = self._resolve_path_from_file_settings(fs.cache_dir, base_directory=pathlib.Path(".")) or self.cache_dir
                    self.log_filename = fs.log_filename if fs.log_filename else self.log_filename
                    self.log_max_bytes = fs.log_max_bytes if fs.log_max_bytes > 0 else self.log_max_bytes
                    self.log_backup_count = fs.log_backup_count if fs.log_backup_count >= 0 else self.log_backup_count
                    self.filename_max_length = fs.filename_max_length if fs.filename_max_length > 0 else self.filename_max_length
                    self.logger.info("AppConfigのパスとログ設定をFileSettingsに基づいて更新しました。")
                elif FileSettingsV49_cls:
                    self.logger.warning("ExternalConfigsV49内にfile_settingsが見つかりませんでしたが、FileSettingsV49モデルはロードされています。")
                else:
                    self.logger.warning("FileSettingsV49モデルクラスが見つかりません。ファイル設定の処理をスキップします。")

            except ValidationError as e_val_app:
                self.logger.error(f"アプリケーション設定ファイル '{app_config_yaml_path}' の検証中にエラーが発生しました: {e_val_app.errors(include_url=False)}", exc_info=False)
                self._external_config_loaded_successfully = False
            except Exception as e_proc_app:
                self.logger.error(f"アプリケーション設定ファイル '{app_config_yaml_path}' の処理中に予期せぬエラーが発生しました: {e_proc_app}", exc_info=True)
                self._external_config_loaded_successfully = False
        elif not raw_app_cfg_data and app_config_yaml_path.exists():
            self.logger.warning(f"アプリケーション設定ファイル '{app_config_yaml_path}' は空または不正な形式です。")
            self._external_config_loaded_successfully = False
        elif not app_config_yaml_path.exists():
            self.logger.info(f"アプリケーション設定ファイル '{app_config_yaml_path}' が見つかりません。デフォルト設定を使用します。")
            try:
                self.loaded_external_configs = ExternalConfigsV49_cls() # type: ignore[operator]
                self._external_config_loaded_successfully = True
            except Exception as e_default_cfg:
                self.logger.error(f"デフォルトのExternalConfigsV49の作成に失敗しました: {e_default_cfg}", exc_info=True)
                self._external_config_loaded_successfully = False


        if not self._external_config_loaded_successfully and app_config_yaml_path.exists():
            self.logger.error("アプリケーション設定ファイルのロードに失敗したため、後続のリソースファイルのロードを中止します。")
            return False

        # subjectivity_keywords_v49.yaml のロード
        SubjectivityKeywordsFileV49_cls = globals().get('SubjectivityKeywordsFileV49')
        if SubjectivityKeywordsFileV49_cls:
            subj_path = self.resources_dir / self.SUBJECTIVITY_KEYWORDS_FILENAME
            raw_subj_data = load_yaml_file(subj_path, expected_type=dict)
            if raw_subj_data:
                try:
                    SubjectivityKeywordsFileV49_cls.model_rebuild(force=True) # type: ignore[attr-defined]
                    self.subjectivity_data = SubjectivityKeywordsFileV49_cls.model_validate(raw_subj_data)
                    self.logger.info(f"主観性キーワードファイル '{subj_path}' の読み込みと検証が完了しました。")
                except ValidationError as e_val_subj:
                    self.logger.error(f"主観性キーワードファイル '{subj_path}' の検証エラー: {e_val_subj.errors(include_url=False)}", exc_info=False)
                except Exception as e_proc_subj:
                    self.logger.error(f"主観性キーワードファイル '{subj_path}' の処理中に予期せぬエラー: {e_proc_subj}", exc_info=True)
            elif subj_path.exists():
                 self.logger.warning(f"主観性キーワードファイル '{subj_path}' は空または不正な形式です。")
            else:
                 self.logger.info(f"主観性キーワードファイル '{subj_path}' が見つかりません。")
        else:
            self.logger.warning("SubjectivityKeywordsFileV49 モデルクラスが見つかりません。主観性キーワードのロードをスキップします。")


        # fluctuation_patterns_v49.yaml のロード
        FluctuationPatternsFileV49_cls = globals().get('FluctuationPatternsFileV49')
        if FluctuationPatternsFileV49_cls:
            fluc_path = self.resources_dir / self.FLUCTUATION_PATTERNS_FILENAME
            raw_fluc_data = load_yaml_file(fluc_path, expected_type=dict)
            if raw_fluc_data:
                try:
                    FluctuationPatternsFileV49_cls.model_rebuild(force=True) # type: ignore[attr-defined]
                    self.fluctuation_data = FluctuationPatternsFileV49_cls.model_validate(raw_fluc_data)
                    self.logger.info(f"揺らぎパターンファイル '{fluc_path}' の読み込みと検証が完了しました。")
                except ValidationError as e_val_fluc:
                    self.logger.error(f"揺らぎパターンファイル '{fluc_path}' の検証エラー: {e_val_fluc.errors(include_url=False)}", exc_info=False)
                except Exception as e_proc_fluc:
                    self.logger.error(f"揺らぎパターンファイル '{fluc_path}' の処理中に予期せぬエラー: {e_proc_fluc}", exc_info=True)
            elif fluc_path.exists():
                self.logger.warning(f"揺らぎパターンファイル '{fluc_path}' は空または不正な形式です。")
            else:
                self.logger.info(f"揺らぎパターンファイル '{fluc_path}' が見つかりません。")
        else:
            self.logger.warning("FluctuationPatternsFileV49 モデルクラスが見つかりません。揺らぎパターンのロードをスキップします。")

        # analyzer_keywords_v49.yaml のロード
        analyzer_kw_path = self.resources_dir / self.ANALYZER_KEYWORDS_FILENAME
        raw_analyzer_kw_data = load_yaml_file(analyzer_kw_path, expected_type=dict)
        if isinstance(raw_analyzer_kw_data, dict):
            self.analyzer_keywords_data = raw_analyzer_kw_data
            self.logger.info(f"アナライザーキーワードファイル '{analyzer_kw_path}' の読み込みが完了しました。キーの数: {len(self.analyzer_keywords_data)}")
        elif analyzer_kw_path.exists():
            self.logger.warning(f"アナライザーキーワードファイル '{analyzer_kw_path}' は空、不正な形式、または辞書形式ではありません。")
        else:
            self.logger.info(f"アナライザーキーワードファイル '{analyzer_kw_path}' が見つかりません。")

        self.logger.info(f"外部設定ファイル・リソースの読み込み処理が完了しました。AppConfigロード成功状態: {self._external_config_loaded_successfully}")
        return self._external_config_loaded_successfully

    def initialize_base_directories(self, base_output_dir_override: Optional[pathlib.Path] = None) -> None:
        self.logger.info("ベースディレクトリの初期化または存在確認を開始します...")

        if base_output_dir_override:
            self.base_output_dir = base_output_dir_override.resolve()
            self.logger.info(f"  コマンドライン引数で指定されたベース出力ディレクトリを使用します: {self.base_output_dir}")
        else:
            # __init__ または load_external_configs で設定された self.base_output_dir を使用
            self.base_output_dir = self.base_output_dir.resolve()
            self.logger.info(f"  設定されたベース出力ディレクトリを使用します: {self.base_output_dir}")

        # 関連ディレクトリも resolve (既に __init__ や load_external_configs で resolve 済みの可能性あり)
        self.config_dir = self.config_dir.resolve()
        self.resources_dir = self.resources_dir.resolve()
        self.cache_dir = self.cache_dir.resolve()

        pcd_from_yaml_str: Optional[Union[str, pathlib.Path]] = None
        if self.loaded_external_configs and \
           hasattr(self.loaded_external_configs, 'file_settings') and \
           self.loaded_external_configs.file_settings and \
           hasattr(self.loaded_external_configs.file_settings, 'persistent_cache_dir'):
            pcd_from_yaml_str = getattr(self.loaded_external_configs.file_settings, 'persistent_cache_dir', None)

        if pcd_from_yaml_str: # 文字列またはPathオブジェクトの場合
             # YAMLに有効なパスがあればそれを使用 (cache_dir基準で解決)
            resolved_pcd = self._resolve_path_from_file_settings(pcd_from_yaml_str, base_directory=self.cache_dir)
            if resolved_pcd:
                self.persistent_cache_dir = resolved_pcd
            else:
                self.logger.warning(f"YAMLで指定された永続キャッシュディレクトリ '{pcd_from_yaml_str}' を解決できませんでした。デフォルトパスを使用します。")
                self.persistent_cache_dir = (self.cache_dir / self.PERSISTENT_CACHE_SUBDIR_NAME_DEFAULT).resolve()
        else: # YAMLに指定がないか、無効な場合はデフォルト
            self.persistent_cache_dir = (self.cache_dir / self.PERSISTENT_CACHE_SUBDIR_NAME_DEFAULT).resolve()
        self.logger.info(f"  永続キャッシュディレクトリが '{self.persistent_cache_dir}' に設定されました。")

        dirs_to_ensure: List[pathlib.Path] = [
            self.base_output_dir, self.config_dir, self.resources_dir,
            self.cache_dir, self.persistent_cache_dir
        ]
        try:
            log_file_path_obj = pathlib.Path(self.log_filename)
            if not log_file_path_obj.is_absolute(): # 相対パスの場合、base_output_dir基準で解決することも検討できる
                                                 # ここではカレントディレクトリ基準で解決
                log_file_path_obj = log_file_path_obj.resolve()

            log_file_parent = log_file_path_obj.parent
            # カレントディレクトリやbase_output_dirと異なる場合のみ追加
            # （ログファイルがこれらのディレクトリ直下にない場合）
            if log_file_parent not in [pathlib.Path(".").resolve(), self.base_output_dir.resolve(), self.cache_dir.resolve()]:
                 dirs_to_ensure.append(log_file_parent)
        except Exception as e_log_path:
            self.logger.warning(f"ログファイルパス '{self.log_filename}' の親ディレクトリ取得中にエラー: {e_log_path}。ディレクトリ作成リストに追加されません。")

        unique_dirs_to_create = sorted(list(set(d for d in dirs_to_ensure if isinstance(d, pathlib.Path))))

        all_dirs_ok = True
        for dir_path_item in unique_dirs_to_create:
            self.logger.debug(f"  ディレクトリの存在確認/作成: {dir_path_item}")
            if not self._ensure_dir_exists(dir_path_item):
                all_dirs_ok = False
                # ベース出力ディレクトリの作成失敗は致命的エラーとして扱う
                if dir_path_item == self.base_output_dir:
                     # このエラーは呼び出し元でキャッチされることを期待
                     raise RuntimeError(f"ベース出力ディレクトリ '{self.base_output_dir}' の作成に失敗しました。処理を続行できません。")
                else:
                     self.logger.error(f"ディレクトリ '{dir_path_item}' の作成に失敗しました。関連機能が正しく動作しない可能性があります。")
        if all_dirs_ok:
            self.logger.info("ベースディレクトリの初期化または存在確認が正常に完了しました。")
        else:
            self.logger.warning("一部のベースディレクトリの初期化に失敗しました。")


    @property
    def api_key(self) -> Optional[str]:
        key_from_env = os.environ.get(self.API_KEY_ENV_VAR)
        if key_from_env:
            return key_from_env
        if self.loaded_external_configs and \
           hasattr(self.loaded_external_configs, 'api_settings') and \
           self.loaded_external_configs.api_settings and \
           isinstance(self.loaded_external_configs.api_settings.api_key, str):
            return self.loaded_external_configs.api_settings.api_key
        return None

    @property
    def generation_config(self) -> Dict[str, Any]:
        APISettingsV49_cls = globals().get('APISettingsV49')
        if not APISettingsV49_cls: # クラスが見つからない場合のフォールバック
            self.logger.error("APISettingsV49クラスが見つかりません。generation_configは空の辞書を返します。")
            return {}

        if self.loaded_external_configs and \
           hasattr(self.loaded_external_configs, 'api_settings') and \
           self.loaded_external_configs.api_settings and \
           isinstance(self.loaded_external_configs.api_settings.generation_config, dict):
            return self.loaded_external_configs.api_settings.generation_config.copy()
        # APISettingsV49 のデフォルトファクトリを使用
        return APISettingsV49_cls().generation_config.copy()

    @property
    def safety_settings(self) -> Optional[Dict[str, str]]:
        APISettingsV49_cls = globals().get('APISettingsV49')
        if not APISettingsV49_cls:
            self.logger.error("APISettingsV49クラスが見つかりません。safety_settingsはNoneを返します。")
            return None

        if self.loaded_external_configs and \
           hasattr(self.loaded_external_configs, 'api_settings') and \
           self.loaded_external_configs.api_settings and \
           isinstance(self.loaded_external_configs.api_settings.safety_settings, dict):
            return self.loaded_external_configs.api_settings.safety_settings.copy()
        return APISettingsV49_cls().safety_settings.copy()

    @property
    def dfrs_weights(self) -> Dict['DFRSMetricsV49', float]: # type: ignore[name-defined]
        if hasattr(self, '_cached_dfrs_weights'):
            return self._cached_dfrs_weights.copy() # キャッシュされたものを返す

        self._cached_dfrs_weights: Dict['DFRSMetricsV49', float] = {} # type: ignore[name-defined]
        DFRSMetricsV49_cls = globals().get('DFRSMetricsV49')

        if not DFRSMetricsV49_cls:
            self.logger.error("DFRSMetricsV49 Enumクラスが見つかりません。DFRS重みは空になります。")
            return {}

        if self.loaded_external_configs and \
           hasattr(self.loaded_external_configs, 'dfrs_weights_config') and \
           self.loaded_external_configs.dfrs_weights_config and \
           isinstance(self.loaded_external_configs.dfrs_weights_config.weights, dict):
            raw_weights = self.loaded_external_configs.dfrs_weights_config.weights
            for key_str, value_float in raw_weights.items():
                if not isinstance(key_str, str):
                    self.logger.warning(f"DFRS重み設定内のキーが文字列ではありません: {key_str} (型: {type(key_str)})。スキップします。")
                    continue
                try:
                    # 文字列キーをDFRSMetricsV49 Enumメンバーに変換
                    enum_key = DFRSMetricsV49_cls(key_str) # Enumの_missing_が呼ばれる
                    if isinstance(value_float, (int, float)) and enum_key != DFRSMetricsV49_cls.UNKNOWN:
                        self._cached_dfrs_weights[enum_key] = float(value_float)
                    elif enum_key == DFRSMetricsV49_cls.UNKNOWN and key_str.lower().strip() != DFRSMetricsV49_cls.UNKNOWN.value:
                         self.logger.warning(f"DFRS重みキー '{key_str}' は UNKNOWN にマップされました。重み設定から除外します。")
                    elif not isinstance(value_float, (int, float)):
                         self.logger.warning(f"DFRS重みキー '{key_str}' (Enum: {enum_key.value}) の値 '{value_float}' が数値ではありません。無視します。")

                except ValueError: # Enum変換失敗
                    self.logger.warning(f"DFRS重み設定内のキー '{key_str}' をDFRSMetricsV49 Enumに変換できませんでした。無視します。")
                except Exception as e:
                    self.logger.error(f"DFRS重み処理中に予期せぬエラー (キー: {key_str}): {e}", exc_info=True)
        else:
             self.logger.debug("DFRS重み設定がロードされていないか、形式が不正です。空の重みセットを使用します。")
        return self._cached_dfrs_weights.copy()

    @property
    def subjectivity_keywords(self) -> Dict['SubjectivityCategoryV49', List['SubjectivityKeywordEntryV49']]: # type: ignore[name-defined]
        """
        キャッシュされた主観性キーワードデータを返します。
        データは SubjectivityKeywordsFileV49 バリデータによって既に検証・変換済みであることを期待します。
        """
        if hasattr(self, '_cached_subjectivity_keywords'):
            return self._cached_subjectivity_keywords # 既にキャッシュされていればそれを返す

        # SubjectivityKeywordsFileV49.root は Dict[SubjectivityCategoryV49, List[SubjectivityKeywordEntryV49]] 型のはず
        if self.subjectivity_data and \
           hasattr(self.subjectivity_data, 'root') and \
           isinstance(self.subjectivity_data.root, dict):
            # 型チェックとロギングを強化
            SubjectivityCategoryEnum = globals().get('SubjectivityCategoryV49')
            SubjectivityKeywordEntryModel = globals().get('SubjectivityKeywordEntryV49')

            if not (SubjectivityCategoryEnum and SubjectivityKeywordEntryModel):
                self.logger.error("SubjectivityCategoryV49 Enum または SubjectivityKeywordEntryV49 モデルが見つかりません。subjectivity_keywords は空になります。")
                self._cached_subjectivity_keywords = {}
                return {}

            processed_data: Dict['SubjectivityCategoryV49', List['SubjectivityKeywordEntryV49']] = {} # type: ignore[name-defined]
            valid_load = True
            for cat_enum, entries_list in self.subjectivity_data.root.items():
                if not isinstance(cat_enum, SubjectivityCategoryEnum):
                    self.logger.warning(f"AppConfig.subjectivity_keywords: 不正なカテゴリキー型 '{type(cat_enum)}' (値: {cat_enum!r})。期待: {SubjectivityCategoryEnum.__name__}。スキップ。")
                    valid_load = False
                    continue
                if not isinstance(entries_list, list):
                    self.logger.warning(f"AppConfig.subjectivity_keywords: カテゴリ '{cat_enum.value}' のデータがリストではありません (型: {type(entries_list)})。スキップ。")
                    valid_load = False
                    continue
                
                # entries_list 内の各要素が SubjectivityKeywordEntryModel インスタンスであるかを確認
                # SubjectivityKeywordsFileV49 バリデータで変換済みのはず
                validated_entries_for_cat: List['SubjectivityKeywordEntryV49'] = [] # type: ignore[name-defined]
                for entry_item in entries_list:
                    if isinstance(entry_item, SubjectivityKeywordEntryModel):
                        validated_entries_for_cat.append(entry_item)
                    else:
                        self.logger.warning(f"AppConfig.subjectivity_keywords: カテゴリ '{cat_enum.value}' 内に不正なエントリ型 '{type(entry_item)}' を検出。期待: {SubjectivityKeywordEntryModel.__name__}。このエントリはスキップ。")
                        valid_load = False # 一つでも不正なものがあれば、ロード全体が不完全とみなすか検討
                
                if validated_entries_for_cat: # 有効なエントリがあった場合のみ追加
                    processed_data[cat_enum] = validated_entries_for_cat
            
            if not valid_load:
                self.logger.error("AppConfig.subjectivity_keywords: データの一部に型不一致または問題がありました。結果が不完全である可能性があります。")

            self._cached_subjectivity_keywords = processed_data
        else:
            self.logger.debug("AppConfig.subjectivity_keywords: self.subjectivity_data が未ロードまたは不正な形式のため、空のデータを返します。")
            self._cached_subjectivity_keywords = {}
        return self._cached_subjectivity_keywords

    @property
    def fluctuation_patterns(self) -> Dict['FluctuationCategoryV49', List['FluctuationPatternEntryV49']]: # type: ignore[name-defined]
        """
        キャッシュされた揺らぎパターンデータを返します。
        データは FluctuationPatternsFileV49 バリデータによって既に検証・変換済みであることを期待します。
        """
        if hasattr(self, '_cached_fluctuation_patterns'):
            return self._cached_fluctuation_patterns

        if self.fluctuation_data and \
           hasattr(self.fluctuation_data, 'root') and \
           isinstance(self.fluctuation_data.root, dict):

            FluctuationCategoryEnum = globals().get('FluctuationCategoryV49')
            FluctuationPatternEntryModel = globals().get('FluctuationPatternEntryV49')

            if not (FluctuationCategoryEnum and FluctuationPatternEntryModel):
                self.logger.error("FluctuationCategoryV49 Enum または FluctuationPatternEntryV49 モデルが見つかりません。fluctuation_patterns は空になります。")
                self._cached_fluctuation_patterns = {}
                return {}

            processed_data: Dict['FluctuationCategoryV49', List['FluctuationPatternEntryV49']] = {} # type: ignore[name-defined]
            valid_load = True
            for cat_enum, entries_list in self.fluctuation_data.root.items():
                if not isinstance(cat_enum, FluctuationCategoryEnum):
                    self.logger.warning(f"AppConfig.fluctuation_patterns: 不正なカテゴリキー型 '{type(cat_enum)}' (値: {cat_enum!r})。期待: {FluctuationCategoryEnum.__name__}。スキップ。")
                    valid_load = False
                    continue
                if not isinstance(entries_list, list):
                    self.logger.warning(f"AppConfig.fluctuation_patterns: カテゴリ '{cat_enum.value}' のデータがリストではありません (型: {type(entries_list)})。スキップ。")
                    valid_load = False
                    continue
                
                validated_entries_for_cat: List['FluctuationPatternEntryV49'] = [] # type: ignore[name-defined]
                for entry_item in entries_list:
                    if isinstance(entry_item, FluctuationPatternEntryModel):
                        validated_entries_for_cat.append(entry_item)
                    else:
                        self.logger.warning(f"AppConfig.fluctuation_patterns: カテゴリ '{cat_enum.value}' 内に不正なエントリ型 '{type(entry_item)}' を検出。期待: {FluctuationPatternEntryModel.__name__}。このエントリはスキップ。")
                        valid_load = False
                
                if validated_entries_for_cat:
                    processed_data[cat_enum] = validated_entries_for_cat
            
            if not valid_load:
                 self.logger.error("AppConfig.fluctuation_patterns: データの一部に型不一致または問題がありました。結果が不完全である可能性があります。")

            self._cached_fluctuation_patterns = processed_data
        else:
            self.logger.debug("AppConfig.fluctuation_patterns: self.fluctuation_data が未ロードまたは不正な形式のため、空のデータを返します。")
            self._cached_fluctuation_patterns = {}
        return self._cached_fluctuation_patterns

    @property
    def temperature_config(self) -> Optional['TemperatureStrategyConfigV49Type']: # type: ignore[name-defined]
        if self.loaded_external_configs and \
           hasattr(self.loaded_external_configs, 'temperature_config'):
            # TemperatureStrategyConfigV49 モデルがロードされているか確認
            TemperatureStrategyConfigV49_cls = globals().get('TemperatureStrategyConfigV49')
            if TemperatureStrategyConfigV49_cls and \
               isinstance(self.loaded_external_configs.temperature_config, TemperatureStrategyConfigV49_cls):
                return self.loaded_external_configs.temperature_config
            elif TemperatureStrategyConfigV49_cls:
                self.logger.warning(f"loaded_external_configs.temperature_config の型が不正です (型: {type(self.loaded_external_configs.temperature_config)})。TemperatureStrategyConfigV49を期待。")
            else:
                self.logger.error("TemperatureStrategyConfigV49モデルクラスが見つかりません。")
        return None

    @property
    def feature_flags(self) -> Optional['FeatureFlagConfigV49Type']: # type: ignore[name-defined]
        if self.loaded_external_configs and \
           hasattr(self.loaded_external_configs, 'feature_flags'):
            FeatureFlagConfigV49_cls = globals().get('FeatureFlagConfigV49')
            if FeatureFlagConfigV49_cls and \
               isinstance(self.loaded_external_configs.feature_flags, FeatureFlagConfigV49_cls):
                return self.loaded_external_configs.feature_flags
            elif FeatureFlagConfigV49_cls:
                 self.logger.warning(f"loaded_external_configs.feature_flags の型が不正です (型: {type(self.loaded_external_configs.feature_flags)})。FeatureFlagConfigV49を期待。")
            else:
                self.logger.error("FeatureFlagConfigV49モデルクラスが見つかりません。")
        return None

    @property
    def adaptation_config(self) -> Optional['AdaptationStrategyConfigV49Type']: # type: ignore[name-defined]
        if self.loaded_external_configs and \
           hasattr(self.loaded_external_configs, 'adaptation_config'):
            AdaptationStrategyConfigV49_cls = globals().get('AdaptationStrategyConfigV49')
            if AdaptationStrategyConfigV49_cls and \
               isinstance(self.loaded_external_configs.adaptation_config, AdaptationStrategyConfigV49_cls):
                return self.loaded_external_configs.adaptation_config
            elif AdaptationStrategyConfigV49_cls:
                self.logger.warning(f"loaded_external_configs.adaptation_config の型が不正です (型: {type(self.loaded_external_configs.adaptation_config)})。AdaptationStrategyConfigV49を期待。")
            else:
                self.logger.error("AdaptationStrategyConfigV49モデルクラスが見つかりません。")
        return None

    @property
    def feedback_config(self) -> Optional['FeedbackStrategyConfigV49Type']: # type: ignore[name-defined]
        if self.loaded_external_configs and \
           hasattr(self.loaded_external_configs, 'feedback_config'):
            FeedbackStrategyConfigV49_cls = globals().get('FeedbackStrategyConfigV49')
            if FeedbackStrategyConfigV49_cls and \
               isinstance(self.loaded_external_configs.feedback_config, FeedbackStrategyConfigV49_cls):
                return self.loaded_external_configs.feedback_config
            elif FeedbackStrategyConfigV49_cls:
                 self.logger.warning(f"loaded_external_configs.feedback_config の型が不正です (型: {type(self.loaded_external_configs.feedback_config)})。FeedbackStrategyConfigV49を期待。")
            else:
                self.logger.error("FeedbackStrategyConfigV49モデルクラスが見つかりません。")
        return None

    def update_from_args(self, args: Union[argparse.Namespace, Mapping[str, Any], None]) -> None:
        if args is None:
            self.logger.debug("update_from_args: argsがNoneのためスキップします。")
            return

        arg_dict: Dict[str, Any] = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
        self.logger.debug(f"コマンドライン引数からのAppConfig設定オーバーライドを開始します (対象引数: {len(arg_dict)}個)。")

        attr_to_cli_map = {
            "log_level_str": "log_level",
            "log_filename": "log_file",
            "log_max_bytes": "log_max_bytes", # このキー名はargparseで定義されていれば
            "log_backup_count": "log_backup_count", # 同上
            "enable_filelock": "enable_filelock", # 同上 (action=argparse.BooleanOptionalAction を想定)
            "cache_cleanup_on_start": "clear_cache", # action='store_true' を想定
            "cache_vacuum_on_clear": "cache_vacuum", # action='store_true' を想定
            "external_config_enabled": "EXTERNAL_CONFIG_ENABLED" # action=argparse.BooleanOptionalAction を想定
        }
        updated_attributes_log: List[str] = []

        for app_config_attr, cli_arg_key in attr_to_cli_map.items():
            if cli_arg_key in arg_dict and arg_dict[cli_arg_key] is not None:
                new_value_from_arg = arg_dict[cli_arg_key]
                current_instance_value = getattr(self, app_config_attr, None)
                # 属性の期待される型を取得 (既存の値の型から推測、またはデフォルトはstr)
                expected_type = type(current_instance_value) if current_instance_value is not None else str

                try:
                    converted_value: Any = None
                    conversion_successful = False
                    if expected_type is bool:
                        # argparse.BooleanOptionalAction は直接bool値を設定する
                        if isinstance(new_value_from_arg, bool):
                            converted_value = new_value_from_arg
                            conversion_successful = True
                        else: # store_true のような場合 (文字列で True/False が来ることは稀)
                            converted_value = str(new_value_from_arg).lower() in ['true', '1', 'yes', 'on']
                            conversion_successful = True
                    elif expected_type is int:
                        converted_value = int(new_value_from_arg)
                        conversion_successful = True
                    elif expected_type is float:
                        converted_value = float(new_value_from_arg)
                        conversion_successful = True
                    elif expected_type is str:
                        converted_value = str(new_value_from_arg)
                        conversion_successful = True
                    # pathlib.Path はここでは直接扱わず、文字列として更新後、initialize_base_directoriesで処理

                    if conversion_successful:
                        setattr(self, app_config_attr, converted_value)
                        updated_attributes_log.append(f"{app_config_attr}={converted_value!r}")
                    elif new_value_from_arg is not None: # Noneでなく、変換対象でもなかった場合
                        self.logger.warning(
                            f"AppConfig属性 '{app_config_attr}' のコマンドライン引数値 '{new_value_from_arg}' (型: {type(new_value_from_arg)}) は、"
                            f"期待される型 ({expected_type.__name__}) に自動変換できませんでした。スキップします。"
                        )
                except (ValueError, TypeError) as e_conversion:
                    self.logger.warning(
                        f"AppConfig属性 '{app_config_attr}' のコマンドライン引数値 '{new_value_from_arg}' の型変換中にエラーが発生しました: {e_conversion}。スキップします。"
                    )

        # クラス変数 DEFAULT_MODEL の更新
        # args.model は create_argument_parser_v49 で 'DEFAULT_MODEL' というdest名で定義されている想定
        cli_default_model_value = arg_dict.get('DEFAULT_MODEL')
        if cli_default_model_value is not None and isinstance(cli_default_model_value, str) and cli_default_model_value.strip():
            AppConfigV49.DEFAULT_MODEL = cli_default_model_value.strip() # クラス変数を直接変更
            updated_attributes_log.append(f"DEFAULT_MODEL (ClassVar) = '{AppConfigV49.DEFAULT_MODEL}'")

        if updated_attributes_log:
            self.logger.info(f"AppConfigの属性をコマンドライン引数から更新しました: {', '.join(updated_attributes_log)}")
        else:
            self.logger.debug("AppConfig: コマンドライン引数による属性の更新はありませんでした。")

# =============================================================================
# -- AppConfigV49 クラス定義終了
# =============================================================================
# -----------------------------------------------------------------------------
# ユーティリティ関数 (クラス外 - 最適化版)
# -----------------------------------------------------------------------------
def load_yaml_file(
    file_path: Union[str, pathlib.Path],
    expected_type: Type[Union[Dict[Any, Any], List[Any]]] = dict # type: ignore # デフォルトをdictに
) -> Optional[Union[Dict[Any, Any], List[Any]]]:
    """
    YAMLファイルを安全に読み込み、期待される型（辞書またはリスト）か確認します。
    エラー発生時や型不一致の場合はNoneを返します。
    """
    path = pathlib.Path(file_path)
    logger_util = logging.getLogger(f"{__name__}.load_yaml_file") # ロガー名を修正

    if not path.is_file():
        logger_util.warning(f"YAMLファイルが見つかりません: '{path}'")
        return None
    if not YAML_AVAILABLE:
        logger_util.error(f"YAML読み込み不可 (PyYAMLライブラリがインポートされていません): '{path}'")
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as e_yaml:
        logger_util.error(f"YAMLファイルのパースエラー ({path}): {e_yaml}", exc_info=False) # トレースバックは不要なことが多い
        return None
    except Exception as e_io: # ファイルIOエラーなど
        logger_util.error(f"YAMLファイルの読み込み中にエラー ({path}): {e_io}", exc_info=True)
        return None

    if raw_data is None: # 空ファイルやコメントのみのファイルなど
        logger_util.warning(f"YAMLファイルが空または有効なYAMLコンテンツを含んでいません: '{path}'")
        return None

    if isinstance(raw_data, expected_type):
        logger_util.debug(f"YAMLファイル '{path}' の読み込みと型チェック成功 (期待型: {expected_type.__name__})。")
        return raw_data
    else:
        logger_util.error(
            f"YAMLファイルのトップレベルデータ型が不正です ({path})。 "
            f"期待された型: {expected_type.__name__}, 実際の型: {type(raw_data).__name__}。"
        )
        return None

def save_json(data: Any, filepath: Union[str, pathlib.Path], indent: Optional[int] = 2) -> bool:
    """JSONデータを指定されたファイルパスに保存します。成功すればTrue。"""
    path = pathlib.Path(filepath)
    logger_json = logging.getLogger(f"{__name__}.save_json")
    try:
        # 親ディレクトリの存在確認と作成 (AppConfigV49._ensure_dir_exists を再利用)
        if not AppConfigV49._ensure_dir_exists(path.parent): # 静的メソッドとして呼び出し
            logger_json.error(f"親ディレクトリ '{path.parent}' の作成に失敗したため、JSONを保存できません。")
            return False
            
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, default=str) # default=str で非シリアライズ可能型を文字列化
        logger_json.debug(f"JSONデータ保存成功: '{path}'")
        return True
    except TypeError as e_type:
        logger_json.error(f"JSONシリアライズエラー ({path}): {e_type}。データ内容を確認してください。", exc_info=True)
    except OSError as e_os:
        logger_json.error(f"JSONファイル書き込みOSエラー ({path}): {e_os}", exc_info=True)
    except Exception as e_unknown:
        logger_json.error(f"JSON保存中に予期せぬエラー ({path}): {e_unknown}", exc_info=True)
    return False

def load_json(filepath: Union[str, pathlib.Path]) -> Optional[Any]:
    """JSONファイルを読み込みます。エラーの場合はNoneを返します。"""
    path = pathlib.Path(filepath)
    logger_json = logging.getLogger(f"{__name__}.load_json")
    if not path.is_file():
        logger_json.warning(f"JSONファイルが見つかりません: '{path}'")
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        logger_json.debug(f"JSONファイル読み込み成功: '{path}'")
        return data
    except json.JSONDecodeError as e_decode:
        logger_json.error(f"JSONデコードエラー ({path}): {e_decode.msg} (line {e_decode.lineno}, col {e_decode.colno})")
    except OSError as e_os:
        logger_json.error(f"JSONファイル読み込みOSエラー ({path}): {e_os}", exc_info=True)
    except Exception as e_unknown:
        logger_json.error(f"JSON読み込み中に予期せぬエラー ({path}): {e_unknown}", exc_info=True)
    return None

def extract_json_from_text(text: Optional[str]) -> Optional[Union[Dict[Any, Any], List[Any]]]:
    """テキストからJSONコードブロックまたはJSON文字列を抽出・パースします。"""
    logger_extract = logging.getLogger(f"{__name__}.extract_json_from_text")
    if not isinstance(text, str) or not text.strip():
        logger_extract.debug("入力テキストがNoneまたは空のため、JSON抽出をスキップします。")
        return None

    # 1. JSONコードブロックの抽出 (```json ... ```)
    #    改行や空白の許容度を高めたパターン
    json_block_pattern = re.compile(r"[`~]{3,}\s*json\s*\n([\s\S]+?)\n\s*[`~]{3,}", re.MULTILINE)
    match = json_block_pattern.search(text)
    if match:
        json_str = match.group(1).strip()
        try:
            data = json.loads(json_str)
            logger_extract.debug(f"JSONコードブロック抽出・パース成功。コンテンツプレビュー: '{json_str[:100]}...'")
            return data
        except json.JSONDecodeError as e:
            logger_extract.warning(f"抽出されたJSONブロックのパースに失敗しました: {e.msg}。コードブロック内容: '{json_str[:100]}...'")
            # コードブロックがあってもパースできない場合は、次のステップに進む

    # 2. テキスト全体がJSON形式かどうかのチェック
    text_stripped = text.strip()
    if (text_stripped.startswith('{') and text_stripped.endswith('}')) or \
       (text_stripped.startswith('[') and text_stripped.endswith(']')):
        try:
            data = json.loads(text_stripped)
            logger_extract.debug("テキスト全体が有効なJSON形式であると判断し、パース成功。")
            return data
        except json.JSONDecodeError:
            logger_extract.debug("テキスト全体はJSON形式ではありませんでした（または不完全）。部分抽出を試みます。")

    # 3. 内部の部分的なJSON文字列の探索 (よりロバストなパターンを検討)
    #    単純な '{...}' や '[...]' のみだと誤マッチが多いため、
    #    より複雑なJSON構造を考慮したパターンや、複数候補から最も妥当なものを選択するロジックが必要になる場合がある。
    #    ここでは、最も外側の有効なJSONオブジェクト/配列を見つける試みを維持。
    try:
        # 期待される開始・終了文字のペアで探索
        json_candidates: List[str] = []
        # 正規表現で'{'と'}'のバランス、'['と']'のバランスが取れている部分を大まかに探す
        # (完璧ではないが、単純な findall よりはマシになる可能性)
        # これは非常に複雑で誤検知も多いため、ここでは簡易的なアプローチを維持し、
        # 信頼性の高いコードブロックや全体のパースを優先する。
        # 簡易的なものとして、最も外側にある可能性のあるものを探す。
        # 最初の '{' と最後の '}'、または最初の '[' と最後の ']'
        first_brace = text_stripped.find('{')
        last_brace = text_stripped.rfind('}')
        first_bracket = text_stripped.find('[')
        last_bracket = text_stripped.rfind(']')

        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_candidates.append(text_stripped[first_brace : last_brace + 1])
        if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
            json_candidates.append(text_stripped[first_bracket : last_bracket + 1])
        
        # 長いもの（より包括的である可能性が高い）から試す
        for potential_str in sorted(json_candidates, key=len, reverse=True):
            try:
                data = json.loads(potential_str)
                logger_extract.debug(f"内部部分文字列からのJSON抽出・パース成功。候補: '{potential_str[:100]}...'")
                return data
            except json.JSONDecodeError:
                continue # 次の候補へ
    except Exception as e_re_find: # 正規表現関連のエラーキャッチ
        logger_extract.warning(f"部分JSON文字列の探索中にエラー: {e_re_find}")

    logger_extract.debug("テキストから有効なJSONコンテンツを見つけられませんでした。")
    return None

def save_text(filepath: Union[str, pathlib.Path], content: str, encoding: str = 'utf-8') -> bool:
    """テキストデータを指定されたファイルパスに保存します。成功すればTrue。"""
    # AppConfigV49._ensure_dir_exists を参照するように変更
    path = pathlib.Path(filepath)
    logger_file = logging.getLogger(f"{__name__}.save_text")

    if not isinstance(content, str):
        logger_file.error(f"保存するコンテントが文字列ではありません (型: {type(content)})。パス: '{path}'")
        return False
    try:
        # 親ディレクトリの存在確認と作成
        if not AppConfigV49._ensure_dir_exists(path.parent):
            logger_file.error(f"親ディレクトリ '{path.parent}' の作成に失敗したため、ファイルを保存できません。")
            return False
            
        with path.open('w', encoding=encoding) as f:
            f.write(content)
        logger_file.debug(f"テキスト保存成功: '{path}' (エンコーディング: {encoding})")
        return True
    except OSError as e_os:
        logger_file.error(f"ファイル書き込みOSエラー ({path}): {e_os}", exc_info=True)
    except Exception as e_unknown:
        logger_file.error(f"テキスト保存中に予期せぬエラー ({path}): {e_unknown}", exc_info=True)
    return False

def create_backup(
    filepath: Union[str, pathlib.Path],
    backup_dir_name: str = ".archive", # より一般的な名前に変更
    max_backups: int = 5 # デフォルト値を調整
) -> Optional[pathlib.Path]:
    """
    指定されたファイルのバックアップを作成します。
    成功した場合はバックアップファイルのパスを、失敗した場合はNoneを返します。
    """
    path = pathlib.Path(filepath)
    logger_backup = logging.getLogger(f"{__name__}.create_backup")

    if not path.is_file():
        logger_backup.debug(f"バックアップ対象ファイルが存在しません: '{path}'")
        return None
    try:
        backup_dir = path.parent / backup_dir_name
        if not AppConfigV49._ensure_dir_exists(backup_dir):
            logger_backup.error(f"バックアップディレクトリ '{backup_dir}' の作成に失敗しました。")
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z" # ミリ秒まで
        backup_filename_stem = f"{path.stem}_{timestamp}"
        
        # 拡張子を保持しつつ、ファイル名衝突を避けるために連番を付加する可能性も考慮
        # ここでは単純にタイムスタンプ付きファイル名とする
        backup_path = backup_dir / f"{backup_filename_stem}{path.suffix}"
        
        shutil.copy2(path, backup_path) # メタデータもコピー
        logger_backup.info(f"バックアップ作成成功: '{path}' -> '{backup_path}'")

        # 古いバックアップの削除 (ファイル名パターンでソート)
        if max_backups > 0:
            # パターンはファイル名本体と拡張子で構成
            backup_pattern = f"{path.stem}_*{path.suffix}"
            existing_backups = sorted(
                [f for f in backup_dir.glob(backup_pattern) if f.is_file()],
                key=lambda p: p.name, # 名前でソート (タイムスタンプ順になるはず)
                reverse=True
            )
            
            if len(existing_backups) > max_backups:
                for old_backup_file in existing_backups[max_backups:]:
                    try:
                        old_backup_file.unlink()
                        logger_backup.debug(f"古いバックアップ削除: '{old_backup_file}'")
                    except Exception as e_unlink:
                        logger_backup.warning(f"古いバックアップ '{old_backup_file}' の削除中にエラー: {e_unlink}")
        return backup_path
    except OSError as e_os:
        logger_backup.error(f"バックアップ作成/整理OSエラー ({path}): {e_os}", exc_info=True)
    except Exception as e_unknown:
        logger_backup.error(f"バックアップ作成/整理中に予期せぬエラー ({path}): {e_unknown}", exc_info=True)
    return None


def sanitize_filename(filename: str, max_length: Optional[int] = None, replacement: str = '_') -> str:
    """
    ファイル名として安全な文字列に変換します。
    max_length は AppConfig から取得することを推奨。
    """
    logger_sanitize = logging.getLogger(f"{__name__}.sanitize_filename")
    if not isinstance(filename, str):
        logger_sanitize.warning(f"入力ファイル名が文字列ではありません (型: {type(filename)})。文字列に変換します: '{filename}'")
        filename = str(filename)

    # 1. 空白文字のトリム
    processed_filename = filename.strip()

    # 2. 不許可文字の置換 (Windows, macOS, Linux で安全な範囲で)
    #    ファイルシステムやOSに依存する予約文字・パターンをより厳密に処理
    #    制御文字 (ASCII 0-31 および 127)
    control_chars_pattern = "".join(map(chr, list(range(32)) + [127]))
    # Windows予約文字: < > : " / \ | ? *
    # Linux/macOS予約文字: / (ナル文字は制御文字でカバー)
    # 共通の不許可文字セット
    illegal_chars = r'[<>:"/\\|?*\x00-\x1F\x7F]' # 正規表現パターン
    processed_filename = re.sub(illegal_chars, replacement, processed_filename)

    # 3. Windows予約デバイス名 (大文字・小文字区別なし、拡張子の有無に関わらず)
    #    CON, PRN, AUX, NUL, COM1-9, LPT1-9
    reserved_names_windows = {
        'CON', 'PRN', 'AUX', 'NUL',
        *[f'COM{i}' for i in range(1, 10)],
        *[f'LPT{i}' for i in range(1, 10)]
    }
    name_part_check, ext_part_check = os.path.splitext(processed_filename)
    if name_part_check.upper() in reserved_names_windows:
        processed_filename = f"{name_part_check}{replacement}{ext_part_check}"
        logger_sanitize.debug(f"予約デバイス名を回避: '{name_part_check}' -> '{name_part_check}{replacement}'")


    # 4. 末尾のピリオドやスペースの除去 (Windowsで問題になることがある)
    processed_filename = processed_filename.rstrip('. ')

    # 5. 長さ制限 (拡張子を含めた全長)
    effective_max_length = max_length if max_length is not None else AppConfigV49.FILENAME_MAX_LENGTH_DEFAULT # クラス変数から取得
    
    if len(processed_filename) > effective_max_length:
        name_part, ext_part = os.path.splitext(processed_filename)
        # 拡張子の長さを考慮したファイル名本体の最大長
        max_name_part_len = effective_max_length - len(ext_part)
        if max_name_part_len < 0: # 拡張子が長すぎる場合 (稀だが)
            # 拡張子自体を切り詰めるか、エラーとするか。ここではファイル名全体を切り詰める
            processed_filename = processed_filename[:effective_max_length]
            logger_sanitize.warning(f"拡張子が長すぎるため、ファイル名全体を{effective_max_length}文字に切り詰め: '{processed_filename}'")
        elif len(name_part) > max_name_part_len:
            processed_filename = name_part[:max_name_part_len] + ext_part
            logger_sanitize.debug(f"ファイル名を最大長{effective_max_length}文字に切り詰め: '{processed_filename}'")

    # 6. 空ファイル名になった場合のフォールバック
    if not processed_filename.strip(replacement + '.'): # 置換文字やピリオドのみになった場合も空とみなす
        fallback_name = f"file_{hashlib.md5(filename.encode('utf-8', errors='replace')).hexdigest()[:10]}"
        logger_sanitize.warning(f"サニタイズの結果、ファイル名が空または無効になりました。フォールバック名 '{fallback_name}' を使用します。元の名前: '{filename}'")
        processed_filename = fallback_name
        if len(processed_filename) > effective_max_length: # フォールバック名も長すぎる場合
            processed_filename = processed_filename[:effective_max_length]

    if processed_filename != filename:
        logger_sanitize.debug(f"ファイル名サニタイズ: '{filename}' -> '{processed_filename}'")
    return processed_filename

def fmt(value: Optional[Union[float, int]], precision: int = 2, na_string: str = "N/A") -> str:
    """数値を指定された精度でフォーマットします。Noneや変換不可の場合はna_stringを返します。"""
    if value is None:
        return na_string
    if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)): # type: ignore # npのダミー実装を考慮
        try:
            return f"{float(value):.{precision}f}"
        except (ValueError, TypeError): # 変換エラーは稀だが念のため
            return str(value) # フォーマット失敗時は文字列として返す
    elif isinstance(value, str): # 文字列が渡された場合（数値変換可能か試みる）
        try:
            num_val = float(value)
            if not (np.isnan(num_val) or np.isinf(num_val)): # type: ignore
                return f"{num_val:.{precision}f}"
        except ValueError:
            pass # 文字列が数値に変換できなければそのまま返す
    return str(value) # 上記以外（NaN, Inf, 変換不可文字列など）は文字列としてそのまま返す

def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_fraction: float = 0.25 # ジッターの割合 (0.0 - 1.0)
) -> float:
    """
    指数バックオフによる待機時間を計算します。
    上限付き、ジッター付き。
    """
    if attempt < 0:
        attempt = 0
    # 指数増加: base_delay * (2^attempt)
    delay = base_delay * (2 ** attempt)
    
    # ジッター追加: delay の +/- (jitter_fraction * delay) の範囲でランダム
    # 例: jitter_fraction=0.25 なら、 +/- 25% のジッター
    if jitter_fraction > 0.0:
        jitter_amount = delay * jitter_fraction
        delay += random.uniform(-jitter_amount, jitter_amount)
        
    # 最小値0、最大値max_delayでクリップ
    final_delay = max(0.0, min(delay, max_delay))
    return final_delay

# =============================================================================
# -- Part 2c: Persistent Cache Utility (v4.9α - 最適化版)
# =============================================================================
# ▼▼▼ 修正箇所: 必要な型をここで明示的にインポート ▼▼▼
from typing import TypeVar, Generic, Optional, Union # Unionもここでインポート推奨
# ▲▲▲ 修正箇所 ▲▲▲
# (他の必要な標準ライブラリもPart 0でインポート済みと仮定: random, time, pickle, sqlite3, pathlib, logging)

# 型変数 T を定義 (キャッシュされる値の型)
T = TypeVar("T")

class PersistentCache(Generic[T]):
    """
    SQLite を使用した永続的なキーバリューストアキャッシュ。
    有効期限 (TTL) 付きのデータ保存と、期限切れエントリ削除機能を持つ。
    SQLインジェクション対策として、テーブル名は識別子として基本的なチェックのみ行い、
    キーや値はプレースホルダ経由で処理します。
    """
    DEFAULT_TTL_SECONDS: Optional[int] = 24 * 60 * 60 * 7  # 7日間
    DEFAULT_MAX_SIZE_MB: float = 100.0
    DEFAULT_TABLE_NAME: str = "ndgs_persistent_cache_data" # より具体的なデフォルト名

    def __init__(
        self,
        db_path: Union[str, pathlib.Path], # pathlib.Pathも許容
        table_name: str = DEFAULT_TABLE_NAME,
        max_size_mb: float = DEFAULT_MAX_SIZE_MB, # 現状は目安として使用
        auto_cleanup_on_set_chance: float = 0.1, # set操作時にクリーンアップを実行する確率
        default_ttl_seconds: Optional[int] = DEFAULT_TTL_SECONDS,
        logger_instance: Optional[logging.Logger] = None,
        # config: Optional[Any] = None, # AppConfigV49インスタンスを直接渡すことは避ける
                                        # 必要な設定値は個別に渡すか、プロパティで取得する
        db_timeout_seconds: float = 10.0, # SQLite接続タイムアウト
        wal_mode: bool = True # WALモードをデフォルトで有効化
    ):
        self.db_path: pathlib.Path = pathlib.Path(db_path).resolve()
        
        # table_name の初期化前に self.logger を使わないように修正
        _logger_name_suffix = table_name if table_name and table_name.isidentifier() and table_name.lower() not in {"select", "from", "where", "insert", "update", "delete", "drop", "table", "index", "sqlite_master"} else self.DEFAULT_TABLE_NAME
        self.logger = logger_instance or logging.getLogger(f"{__name__}.PersistentCache.{_logger_name_suffix}")

        if not table_name or not table_name.isidentifier() or table_name.lower() in {"select", "from", "where", "insert", "update", "delete", "drop", "table", "index", "sqlite_master"}:
            self.logger.error(f"無効なテーブル名 '{table_name}' が指定されました。デフォルト '{self.DEFAULT_TABLE_NAME}' を使用します。")
            self.table_name: str = self.DEFAULT_TABLE_NAME
        else:
            self.table_name = table_name

        self.max_size_bytes: int = int(max_size_mb * 1024 * 1024)
        self.auto_cleanup_on_set_chance: float = max(0.0, min(1.0, auto_cleanup_on_set_chance))
        self.default_ttl_seconds: Optional[int] = default_ttl_seconds
        self.db_timeout_seconds: float = db_timeout_seconds
        self.wal_mode: bool = wal_mode
        
        self.conn: Optional[sqlite3.Connection] = None
        self._is_initialized_successfully: bool = False

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"キャッシュDBディレクトリ '{self.db_path.parent}' の存在確認/作成完了。")
            
            self._connect() # 接続試行
            if self.conn:
                self._create_table_if_not_exists()
                self._is_initialized_successfully = True
                self.logger.info(f"永続キャッシュ '{self.table_name}' ({self.db_path}) の初期化成功。")
                if self.auto_cleanup_on_set_chance > 0: # 初期化時にも一度実行 (確率1.0で)
                    self._evict_expired(chance=1.0)
            else:
                self.logger.error(f"永続キャッシュ '{self.table_name}' ({self.db_path}) のDB接続に失敗したため、初期化不完全。")
            
        except Exception as e: # sqlite3.Error 以外もキャッチ
            self.logger.error(f"キャッシュDB '{self.db_path}' の初期化中に予期せぬエラー: {e}", exc_info=True)
            self._is_initialized_successfully = False
            if self.conn: # エラー発生後でもconnが残っている場合があるのでクローズ試行
                self._close()
                
    def _connect(self) -> None:
        """データベースに接続します。既に接続済みの場合は何もしません。"""
        if self.conn is not None:
            # 既存の接続が有効か簡単なチェック (より厳密なチェックは check_connection で)
            try:
                self.conn.execute("SELECT 1;").fetchone()
                self.logger.debug(f"既存のDB接続 ({self.db_path}) は有効です。")
                return
            except sqlite3.Error:
                self.logger.warning(f"既存のDB接続 ({self.db_path}) が無効になっていました。再接続を試みます。")
                self._close() # 無効な接続を閉じる

        try:
            self.conn = sqlite3.connect(
                str(self.db_path),
                timeout=self.db_timeout_seconds,
                check_same_thread=False # 注意: マルチスレッドアクセス時は外部での適切な同期が必要
            )
            if self.wal_mode:
                self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA foreign_keys = ON;") # 通常はデフォルトOFFなので明示的にON
            self.conn.execute("PRAGMA busy_timeout = 5000;") # 5秒のビジータイムアウト
            self.logger.debug(f"キャッシュDB '{self.db_path}' に新規接続しました (Timeout: {self.db_timeout_seconds}s, WAL: {self.wal_mode})。")
        except sqlite3.Error as e_connect:
            self.logger.error(f"キャッシュDB '{self.db_path}' への接続に失敗しました: {e_connect}", exc_info=True)
            self.conn = None # エラー時はNoneを保証
            raise # 接続失敗は致命的なので再送出

    def _close(self) -> None:
        """データベース接続を安全に閉じます。"""
        if self.conn:
            try:
                self.conn.close()
                self.logger.debug(f"キャッシュDB '{self.db_path}' との接続を正常に閉じました。")
            except sqlite3.Error as e_close:
                self.logger.error(f"キャッシュDB '{self.db_path}' のクローズ中にSQLiteエラー: {e_close}", exc_info=True)
            finally:
                self.conn = None # 接続状態を確実にリセット

    def __enter__(self) -> 'PersistentCache[T]':
        if not self._is_initialized_successfully:
            # 初期化に失敗している場合は、接続試行せずエラーを出すか、機能限定モードにする
            # ここではRuntimeErrorを発生させる
            raise RuntimeError(f"PersistentCache (table: {self.table_name}, db: {self.db_path}) は正常に初期化されていません。")
        if self.conn is None: # __init__で接続失敗した場合や、一度閉じた後など
            self._connect() # 再接続を試みる
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        self._close()

    def _create_table_if_not_exists(self) -> None:
        """キャッシュ用テーブルが存在しない場合に作成します。接続必須。"""
        if self.conn is None: # 接続がない場合はエラー
            self.logger.error("テーブル作成試行: DB接続がありません。")
            raise sqlite3.OperationalError("Cannot create table: database connection is not available.")
        
        # テーブル名とインデックス名はSQLインジェクションを防ぐため、プレースホルダではなく直接埋め込みます。
        # self.table_name は __init__ で基本的な識別子チェック済み。
        # しかし、万全を期すなら、さらに厳密なホワイトリスト検証やクォート処理を検討。
        # ここでは isidentifier() によるチェックを信頼する。
        
        # SQL文: value_blob は NOT NULL 制約を付加。
        # created_at も NOT NULL。expires_at は NULL を許容（無期限キャッシュのため）。
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS "{self.table_name}" (
            key_text   TEXT PRIMARY KEY NOT NULL,
            value_blob BLOB NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL
        );
        """
        # インデックス名もサニタイズまたは安全な生成方法を検討
        safe_index_name_part = re.sub(r'[^a-zA-Z0-9_]', '', self.table_name) # 英数字とアンダースコアのみ
        index_name = f"idx_{safe_index_name_part}_expires_at"
        create_index_sql = f"""
        CREATE INDEX IF NOT EXISTS "{index_name}"
        ON "{self.table_name}" (expires_at) WHERE expires_at IS NOT NULL;
        """ # expires_at が NULL でないエントリのみインデックス対象

        try:
            with self.conn: # コンテキストマネージャで自動コミット/ロールバック
                self.conn.execute(create_table_sql)
                self.logger.debug(f"キャッシュテーブル '{self.table_name}' の存在確認/作成SQLを実行しました。")
                self.conn.execute(create_index_sql)
                self.logger.debug(f"キャッシュテーブル '{self.table_name}' のインデックス '{index_name}' の存在確認/作成SQLを実行しました。")
            self.logger.info(f"キャッシュテーブル '{self.table_name}' および関連インデックスの準備完了。")
        except sqlite3.Error as e_create_table:
            self.logger.error(f"キャッシュテーブル '{self.table_name}' またはインデックスの作成/準備に失敗しました: {e_create_table}", exc_info=True)
            raise # エラーを再送出して初期化失敗を明確にする

    def get(self, key: str) -> Optional[T]:
        """
        指定されたキーに対応する値をキャッシュから取得します。
        期限切れ、デシリアライズ失敗、またはDBエラーの場合はNoneを返します。
        """
        if not self._is_initialized_successfully or self.conn is None:
            self.logger.warning(f"Get操作 ({key=}): キャッシュ未初期化またはDB未接続。Noneを返します。")
            return None
        
        current_time = time.time()
        try:
            select_sql = f"SELECT value_blob, expires_at FROM \"{self.table_name}\" WHERE key_text = ?;"
            cursor = self.conn.execute(select_sql, (key,))
            row = cursor.fetchone()

            if row:
                value_blob, expires_at_timestamp = row
                if expires_at_timestamp is not None and expires_at_timestamp < current_time:
                    self.logger.info(f"キャッシュキー '{key}' (期限: {datetime.fromtimestamp(expires_at_timestamp, tz=timezone.utc).isoformat()}) は現在時刻 ({datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat()}) より古いため期限切れ。削除します。")
                    self.delete(key) # 期限切れエントリを削除
                    return None
                try:
                    value: T = pickle.loads(value_blob)
                    self.logger.debug(f"キャッシュHIT: キー '{key}' (期限: {datetime.fromtimestamp(expires_at_timestamp, tz=timezone.utc).isoformat() if expires_at_timestamp else '無期限'})")
                    return value
                except pickle.UnpicklingError as e_unpickle:
                    self.logger.error(f"キー '{key}' の値のデシリアライズに失敗しました: {e_unpickle}。破損データとして削除します。", exc_info=False) # トレースバックはノイズが多い場合がある
                    self.delete(key) # 破損エントリを削除
                    return None
                except Exception as e_deserialize_other:
                    self.logger.error(f"キー '{key}' の値のデシリアライズ中に予期せぬエラー: {e_deserialize_other}。破損データとして削除します。", exc_info=True)
                    self.delete(key)
                    return None
            else:
                self.logger.debug(f"キャッシュMISS: キー '{key}' は見つかりませんでした。")
                return None
        except sqlite3.Error as e_sqlite_get:
            self.logger.error(f"キー '{key}' の取得中にSQLiteエラー: {e_sqlite_get}", exc_info=True)
            return None
        except Exception as e_get_other:
            self.logger.error(f"キー '{key}' の取得中に予期せぬエラー: {e_get_other}", exc_info=True)
            return None

    def set(self, key: str, value: T, ttl_seconds: Optional[int] = -1) -> bool:
        """
        指定されたキーと値をキャッシュに保存します。TTLも指定可能です。
        成功した場合はTrue、失敗した場合はFalseを返します。
        ttl_seconds = -1 (デフォルト): クラスのdefault_ttl_secondsを使用
        ttl_seconds = None: 無期限
        ttl_seconds = 0: 即時期限切れ (実質的に保存直後に期限切れ)
        ttl_seconds > 0: 指定秒数
        """
        if not self._is_initialized_successfully or self.conn is None:
            self.logger.error(f"Set操作 ({key=}): キャッシュ未初期化またはDB未接続。保存できません。")
            return False

        effective_ttl: Optional[int]
        if ttl_seconds == -1:
            effective_ttl = self.default_ttl_seconds
        else:
            effective_ttl = ttl_seconds
        
        current_time = time.time()
        expires_at_timestamp: Optional[float] = None
        if effective_ttl is not None:
            if effective_ttl < 0: # 負のTTLは無期限として扱う (またはエラーとするか要件次第)
                self.logger.warning(f"TTLに負の値 ({effective_ttl}) が指定されました。無期限として扱います。")
                expires_at_timestamp = None
            else:
                expires_at_timestamp = current_time + effective_ttl

        try:
            serialized_value: bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL) # protocol指定
        except pickle.PicklingError as e_pickle:
            self.logger.error(f"キー '{key}' の値のシリアライズに失敗しました。保存できません: {e_pickle}", exc_info=True)
            return False
        except Exception as e_serialize_other:
            self.logger.error(f"キー '{key}' の値のシリアライズ中に予期せぬエラー。保存できません: {e_serialize_other}", exc_info=True)
            return False

        try:
            replace_sql = f"REPLACE INTO \"{self.table_name}\" (key_text, value_blob, created_at, expires_at) VALUES (?, ?, ?, ?);"
            with self.conn: # 自動コミット/ロールバック
                self.conn.execute(replace_sql, (key, sqlite3.Binary(serialized_value), current_time, expires_at_timestamp))
            
            expires_display = (datetime.fromtimestamp(expires_at_timestamp, tz=timezone.utc).isoformat()
                               if expires_at_timestamp is not None else "無期限")
            self.logger.debug(f"キャッシュSET成功: キー '{key}', TTL: {effective_ttl}s (期限: {expires_display})")

            if self.auto_cleanup_on_set_chance > 0 and random.random() < self.auto_cleanup_on_set_chance:
                # TTLがNone (無期限) の場合でも、他の期限切れエントリを掃除する意味はある
                self.logger.debug(f"Set操作後の自動クリーンアップ実行 (確率: {self.auto_cleanup_on_set_chance*100:.1f}%)")
                self._evict_expired(chance=1.0) # 確実に実行
            return True
        except sqlite3.Error as e_sqlite_set:
            self.logger.error(f"キー '{key}' の設定中にSQLiteエラー: {e_sqlite_set}", exc_info=True)
            return False
        except Exception as e_set_other:
            self.logger.error(f"キー '{key}' の設定中に予期せぬエラー: {e_set_other}", exc_info=True)
            return False

    def delete(self, key: str) -> bool:
        """指定されたキーのエントリをキャッシュから削除します。成功すればTrue。"""
        if not self._is_initialized_successfully or self.conn is None:
            self.logger.warning(f"Delete操作 ({key=}): キャッシュ未初期化またはDB未接続。削除できません。")
            return False
        try:
            delete_sql = f"DELETE FROM \"{self.table_name}\" WHERE key_text = ?;"
            with self.conn: # 自動コミット/ロールバック
                cursor = self.conn.execute(delete_sql, (key,))
            
            if cursor.rowcount > 0:
                self.logger.debug(f"キャッシュDELETE成功: キー '{key}' (削除件数: {cursor.rowcount})")
            else:
                self.logger.debug(f"キャッシュDELETE: キー '{key}' は見つかりませんでした。削除処理は成功とみなします。")
            return True
        except sqlite3.Error as e_sqlite_delete:
            self.logger.error(f"キー '{key}' の削除中にSQLiteエラー: {e_sqlite_delete}", exc_info=True)
            return False
        except Exception as e_delete_other:
            self.logger.error(f"キー '{key}' の削除中に予期せぬエラー: {e_delete_other}", exc_info=True)
            return False

    def clear_cache(self, vacuum: bool = False) -> bool:
        """
        キャッシュテーブル内のすべてのエントリを削除します。
        vacuum=True の場合、SQLiteのVACUUMコマンドを実行してDBファイルを最適化します。
        成功すればTrue。
        """
        if not self._is_initialized_successfully or self.conn is None:
            self.logger.error("Clear操作: キャッシュ未初期化またはDB未接続。クリアできません。")
            return False
        try:
            clear_sql = f"DELETE FROM \"{self.table_name}\";" # テーブル自体は削除しない
            with self.conn: # 自動コミット/ロールバック
                self.conn.execute(clear_sql)
                self.logger.info(f"キャッシュテーブル '{self.table_name}' の全エントリをクリアしました。")
                if vacuum:
                    self.logger.info(f"キャッシュDB '{self.db_path}' のVACUUM処理を開始します...")
                    self.conn.execute("VACUUM;")
                    self.logger.info(f"キャッシュDB '{self.db_path}' のVACUUM処理が完了しました。")
            return True
        except sqlite3.Error as e_sqlite_clear:
            self.logger.error(f"キャッシュテーブル '{self.table_name}' のクリアまたはVACUUM中にSQLiteエラー: {e_sqlite_clear}", exc_info=True)
            return False
        except Exception as e_clear_other:
            self.logger.error(f"キャッシュクリアまたはVACUUM中に予期せぬエラー: {e_clear_other}", exc_info=True)
            return False

    def _evict_expired(self, chance: float = 1.0) -> int:
        """
        期限切れのキャッシュエントリを削除します。
        chanceは実行確率(0.0-1.0)。削除された件数を返します。
        """
        if not self._is_initialized_successfully or self.conn is None:
            self.logger.debug("_evict_expired: キャッシュ未初期化またはDB未接続のためスキップ。")
            return 0
        
        if not (0.0 < chance <= 1.0): # chanceが0以下なら実行しない
             self.logger.debug(f"_evict_expired: 実行確率 ({chance*100:.1f}%) が0のためスキップ。")
             return 0
        if chance < 1.0 and random.random() > chance:
            self.logger.debug(f"_evict_expired: 実行確率 ({chance*100:.1f}%) によりスキップ。")
            return 0

        current_time = time.time()
        deleted_count = 0
        try:
            # expires_at が NULL でない（つまり期限が設定されている）エントリのみを対象
            evict_sql = f"DELETE FROM \"{self.table_name}\" WHERE expires_at IS NOT NULL AND expires_at < ?;"
            with self.conn: # 自動コミット/ロールバック
                cursor = self.conn.execute(evict_sql, (current_time,))
                deleted_count = cursor.rowcount
            
            if deleted_count > 0:
                self.logger.info(f"期限切れキャッシュエントリを {deleted_count} 件削除しました。")
            else:
                self.logger.debug("削除対象の期限切れキャッシュエントリはありませんでした。")
        except sqlite3.Error as e_sqlite_evict:
            self.logger.error(f"期限切れキャッシュエントリの削除中にSQLiteエラー: {e_sqlite_evict}", exc_info=True)
        except Exception as e_evict_other:
            self.logger.error(f"期限切れキャッシュエントリの削除中に予期せぬエラー: {e_evict_other}", exc_info=True)
        return deleted_count

    def check_connection(self) -> bool:
        """データベース接続が現在有効か確認します。必要なら再接続を試みます。"""
        if self.conn is None:
            self.logger.debug("DB接続なし。再接続を試みます...")
            try:
                self._connect()
                return self.conn is not None
            except Exception:
                self.logger.warning("DB再接続試行中にエラー発生。")
                return False
        # 接続がある場合、簡単なクエリでテスト
        try:
            self.conn.execute("SELECT 1;").fetchone()
            self.logger.debug("既存のDB接続は有効です。")
            return True
        except sqlite3.Error as e_check: # OperationalError, ProgrammingErrorなど
            self.logger.warning(f"キャッシュDB接続チェック失敗 (エラー: {e_check})。接続が無効になっている可能性があります。")
            self._close() # 無効な接続を閉じる
            self.logger.debug("無効なDB接続を閉じました。次回アクセス時に再接続されます。")
            return False
        except Exception as e_check_other: # 予期せぬエラー
             self.logger.error(f"キャッシュDB接続チェック中に予期せぬエラー: {e_check_other}", exc_info=True)
             return False


    def check_table_exists(self) -> bool:
        """キャッシュテーブルが存在するか確認します。接続がない場合は接続を試みます。"""
        if self.conn is None:
            if not self.check_connection(): # 接続がなければ、まず接続を試みる
                self.logger.error(f"テーブル存在チェック ({self.table_name}): DB未接続のため確認できません。")
                return False
        
        # self.conn が None でないことをここで再度確認 (check_connectionが失敗した場合など)
        if self.conn is None:
             self.logger.error(f"テーブル存在チェック ({self.table_name}): check_connection後もDB接続がありません。")
             return False
             
        try:
            # SQLインジェクション対策のため、テーブル名はプレースホルダで渡す
            check_sql = "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;"
            cursor = self.conn.execute(check_sql, (self.table_name,))
            exists = cursor.fetchone() is not None
            self.logger.debug(f"キャッシュテーブル '{self.table_name}' の存在チェック結果: {exists}")
            return exists
        except sqlite3.Error as e_sqlite_check_table:
            self.logger.error(f"キャッシュテーブル '{self.table_name}' の存在確認中にSQLiteエラー: {e_sqlite_check_table}", exc_info=True)
            return False
        except Exception as e_check_table_other:
            self.logger.error(f"キャッシュテーブル '{self.table_name}' の存在確認中に予期せぬエラー: {e_check_table_other}", exc_info=True)
            return False

    def close(self) -> None:
        """明示的なクローズメソッド。インスタンスが不要になった際に呼び出します。"""
        self.logger.info(f"永続キャッシュ '{self.table_name}' ({self.db_path}) のクローズ処理を明示的に呼び出しました。")
        self._close()

# =============================================================================
# -- Part 2c 終了点 (PersistentCache クラス定義終了)
# =============================================================================
# =============================================================================
# -- Part 2 終了点 (AppConfigV49 クラス定義とユーティリティ関数定義終了)
# =============================================================================
# =============================================================================
# -- Part 3: Domain Models (Pydantic) (v4.9α - Pydantic V2 Optimized)
# =============================================================================
# Pydanticモデルを定義し、型安全性とバリデーションを強化。
# Pydantic V2 準拠の修正を適用。

# --- 型チェック用の前方参照 ---
# (Enum型はPart 1で定義され、BaseModelなどはPart 0でインポートされている前提)
if TYPE_CHECKING:
    # Enum Types (Assumed defined in Part 1b and globally available)
    EmotionalToneV49 = 'EmotionalToneV49'
    PsychologicalPhaseV49 = 'PsychologicalPhaseV49'
    SubjectivityCategoryV49 = 'SubjectivityCategoryV49'
    FluctuationCategoryV49 = 'FluctuationCategoryV49'
    DFRSMetricsV49 = 'DFRSMetricsV49'
    SubjectiveIntensityLevel = 'SubjectiveIntensityLevel'
    FinalSelectionKeysV49 = 'FinalSelectionKeysV49'
    InitialSelectionKeysV49 = 'InitialSelectionKeysV49'
    ScoreKeys = 'ScoreKeys' # Container for LLM, DFRS enums

    # Pydantic Models (Defined within this Part 3, string literals for forward refs)
    ExternalConfigsV49 = 'ExternalConfigsV49'
    APISettingsV49 = 'APISettingsV49'
    FileSettingsV49 = 'FileSettingsV49'
    FeatureFlagConfigV49 = 'FeatureFlagConfigV49'
    DFRSWeightsConfigV49 = 'DFRSWeightsConfigV49'
    TemperatureStrategyConfigV49 = 'TemperatureStrategyConfigV49'
    FixedTemperatureParams = 'FixedTemperatureParams'
    DecreasingTemperatureParams = 'DecreasingTemperatureParams'
    TwoDimensionalTemperatureParams = 'TwoDimensionalTemperatureParams'
    AdaptationStrategyConfigV49 = 'AdaptationStrategyConfigV49'
    ProbabilisticHistoryAdaptationParams = 'ProbabilisticHistoryAdaptationParams'
    FeedbackStrategyConfigV49 = 'FeedbackStrategyConfigV49'
    CompositeFeedbackParams = 'CompositeFeedbackParams'
    PhaseToneFeedbackParams = 'PhaseToneFeedbackParams'
    SelectionWeightsConfigV49 = 'SelectionWeightsConfigV49'
    InitialSelectionWeightsV49Model = 'InitialSelectionWeightsV49Model'
    FinalSelectionWeightsV49Model = 'FinalSelectionWeightsV49Model'
    ErrorConfigV49 = 'ErrorConfigV49'
    ErrorRecoveryStrategy = 'ErrorRecoveryStrategy'

    SubjectivityKeywordsFileV49 = 'SubjectivityKeywordsFileV49'
    SubjectivityKeywordEntryV49 = 'SubjectivityKeywordEntryV49'
    FluctuationPatternsFileV49 = 'FluctuationPatternsFileV49'
    FluctuationPatternEntryV49 = 'FluctuationPatternEntryV49'

    CharacterV49 = 'CharacterV49'
    SceneInfoV49 = 'SceneInfoV49'
    InputDataV49 = 'InputDataV49'
    DialogueTurnInputV49 = 'DialogueTurnInputV49'
    LLMEvaluationScoresV49 = 'LLMEvaluationScoresV49'
    DFRSSubScoresV49 = 'DFRSSubScoresV49'
    EmotionCurvePointV49 = 'EmotionCurvePointV49'
    PhaseTimelinePointV49 = 'PhaseTimelinePointV49'
    BlockAnalysisTagsV49 = 'BlockAnalysisTagsV49'
    DialogueBlockV49 = 'DialogueBlockV49'
    SpeechBlockV49 = 'SpeechBlockV49'
    DescriptionBlockV49 = 'DescriptionBlockV49'
    VersionStateV49 = 'VersionStateV49'
    PhaseTransitionRecordV49 = 'PhaseTransitionRecordV49'
    OutputEvaluationV49 = 'OutputEvaluationV49'
    GenerationStatsV49 = 'GenerationStatsV49'
    GeneratorStateV49 = 'GeneratorStateV49'
    FeedbackContextV49 = 'FeedbackContextV49'
    OutputCharacterContextV49 = 'OutputCharacterContextV49'
    OutputSceneContextV49 = 'OutputSceneContextV49'
    InputContextV49 = 'InputContextV49'
    OutputDialogueV49 = 'OutputDialogueV49'
    SettingsMetadataV49 = 'SettingsMetadataV49'
    OutputMetadataV49 = 'OutputMetadataV49'
    OutputJsonStructureV49 = 'OutputJsonStructureV49'
    CacheDataV49 = 'CacheDataV49'
    PromptComponentsV49 = 'PromptComponentsV49'

# ----------------------------------------
# -- Part 3a: External Config Models
# ----------------------------------------

class FixedTemperatureParams(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="固定する温度")
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class DecreasingTemperatureParams(BaseModel):
    initial_temperature: float = Field(default=0.9, ge=0.0, le=2.0, description="初期温度")
    final_temperature: float = Field(default=0.5, ge=0.0, le=2.0, description="最終温度")
    decay_rate: float = Field(default=0.05, gt=0.0, lt=1.0, description="減衰率")
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class TwoDimensionalTemperatureParams(BaseModel):
    subjectivity_high_temp: float = Field(default=0.95, ge=0.0, le=2.0)
    subjectivity_low_temp: float = Field(default=0.65, ge=0.0, le=2.0)
    flow_tension_high_temp: float = Field(default=0.9, ge=0.0, le=2.0)
    flow_tension_low_temp: float = Field(default=0.7, ge=0.0, le=2.0)
    low_subjectivity_threshold: float = Field(default=3.0, ge=0.0, le=5.0)
    high_subjectivity_threshold: float = Field(default=4.5, ge=0.0, le=5.0)
    low_fluctuation_threshold: float = Field(default=2.5, ge=0.0, le=5.0)
    high_fluctuation_threshold: float = Field(default=4.2, ge=0.0, le=5.0)
    low_novelty_threshold: float = Field(default=3.0, ge=0.0, le=5.0)
    low_richness_threshold: float = Field(default=3.5, ge=0.0, le=5.0)
    low_internal_depth_threshold: float = Field(default=2.8, ge=0.0, le=5.0)
    low_emotion_complexity_threshold: float = Field(default=2.8, ge=0.0, le=5.0)
    low_eti_threshold: float = Field(default=2.5, ge=0.0, le=5.0)
    low_symbolic_threshold: float = Field(default=2.0, ge=0.0, le=5.0)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class TemperatureStrategyConfigV49(BaseModel):
    strategy_type: Literal["fixed", "decreasing", "two_dimensional"] = Field(default="fixed")
    base_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    fixed_params: 'FixedTemperatureParams' = Field(default_factory=FixedTemperatureParams)
    decreasing_params: 'DecreasingTemperatureParams' = Field(default_factory=DecreasingTemperatureParams)
    two_dimensional_params: 'TwoDimensionalTemperatureParams' = Field(default_factory=TwoDimensionalTemperatureParams)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class ProbabilisticHistoryAdaptationParams(BaseModel):
    exploration_rate: float = Field(default=0.15, ge=0.0, le=1.0, alias="exploration_rate_init")
    exploration_decay_rate: float = Field(default=0.995, ge=0.0, le=1.0, alias="exploration_decay")
    min_exploration_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    alignment_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_history_size: Annotated[int, Field(ge=1)] = Field(default=1000)
    history_save_interval: Annotated[int, Field(ge=1)] = Field(default=10)
    model_config = ConfigDict(extra='ignore', populate_by_name=True, validate_assignment=True)

class AdaptationStrategyConfigV49(BaseModel):
    enabled: bool = Field(default=False)
    strategy_type: Literal["simple_threshold", "probabilistic_history", "advanced_rl"] = Field(default="probabilistic_history")
    log_transitions: bool = Field(default=True)
    probabilistic_history_params: 'ProbabilisticHistoryAdaptationParams' = Field(default_factory=ProbabilisticHistoryAdaptationParams)
    alignment_threshold: float = Field(0.6, ge=0.0, le=1.0)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

    @model_validator(mode='after')
    def sync_alignment_threshold(self) -> 'AdaptationStrategyConfigV49':
        if self.probabilistic_history_params:
            self.probabilistic_history_params.alignment_threshold = self.alignment_threshold
        return self

class CompositeFeedbackParams(BaseModel):
    use_numeric_score: bool = Field(default=True)
    use_phase_tone: bool = Field(default=True)
    use_keywords: bool = Field(default=False)
    keyword_count: Annotated[int, Field(ge=1, le=10)] = Field(default=3)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class PhaseToneFeedbackParams(BaseModel):
    ecs_low_threshold: float = Field(default=2.5, ge=0.0, le=5.0)
    ecs_high_threshold: float = Field(default=4.5, ge=0.0, le=5.0)
    ptn_low_threshold: float = Field(default=3.0, ge=0.0, le=5.0)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class FeedbackStrategyConfigV49(BaseModel):
    strategy_type: Literal["composite", "phase_tone_only", "subjectivity_only", "quality_only", "context_aware", "fluctuation_only"] = Field(default="composite")
    composite_params: 'CompositeFeedbackParams' = Field(default_factory=CompositeFeedbackParams)
    phase_tone_params: 'PhaseToneFeedbackParams' = Field(default_factory=PhaseToneFeedbackParams)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class APISettingsV49(BaseModel):
    api_key: Optional[str] = Field(default=None)
    default_model: str = Field(default="models/gemini-pro")
    rpm_limit: Annotated[int, Field(ge=1)] = Field(default=60)
    evaluation_model: Optional[str] = Field(default=None)
    evaluation_temperature: float = Field(default=0.15, ge=0.0, le=1.0)
    initial_candidate_count: Annotated[int, Field(ge=1, le=8)] = Field(default=3)
    generation_config: Dict[str, Any] = Field(default_factory=lambda: {
        "temperature": 0.8, "top_p": 0.95, "top_k": 50, "candidate_count": 1, "max_output_tokens": 8192
    })
    safety_settings: Dict[str, str] = Field(default_factory=lambda: {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
    })
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

    @model_validator(mode='before')
    @classmethod
    def _load_api_key_from_env(cls, values: Any) -> Any:
        if isinstance(values, dict) and not values.get('api_key'):
            api_key_env_name_default = "GENAI_API_KEY"
            app_config_cls: Optional[Type] = globals().get('AppConfigV49') # Ensure AppConfigV49 is loaded if this is used
            api_key_env_var_name = getattr(app_config_cls, 'API_KEY_ENV_VAR', api_key_env_name_default) if app_config_cls else api_key_env_name_default
            
            api_key_env = os.environ.get(api_key_env_var_name)
            if api_key_env:
                values['api_key'] = api_key_env
                logging.getLogger(f"{cls.__module__}.{cls.__qualname__}").info(
                    f"環境変数 '{api_key_env_var_name}' からAPIキーを読み込みました。"
                )
        return values

class DFRSWeightsConfigV49(BaseModel):
    description: Optional[str] = Field(default="DFRS評価指標の重み設定")
    weights: Dict['DFRSMetricsV49', float] = Field(default_factory=dict)
    model_config = ConfigDict(extra='ignore', validate_assignment=True, arbitrary_types_allowed=True)

    @field_validator('weights', mode='before')
    @classmethod
    def _convert_weight_keys_to_enum(cls, v: Any) -> Dict['DFRSMetricsV49', float]: # type: ignore
        if not isinstance(v, dict):
            if v is None: return {}
            raise ValueError("DFRS weights must be a dictionary.")
        
        logger = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        converted_dict: Dict['DFRSMetricsV49', float] = {} # type: ignore
        DFRSMetricsEnum: Optional[Type[enum.Enum]] = globals().get('DFRSMetricsV49') # type: ignore

        if not DFRSMetricsEnum or not issubclass(DFRSMetricsEnum, enum.Enum):
            logger.error("DFRSMetricsV49 Enum not found or not an Enum. Weights validation will be incomplete.")
            # Attempt a basic conversion if Enum is not found, to prevent outright failure if possible
            temp_dict = {}
            for key, value in v.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    temp_dict[key] = float(value)
            return temp_dict # type: ignore

        for key_str, weight_val in v.items():
            try:
                enum_member = DFRSMetricsEnum(str(key_str).lower().strip()) # type: ignore
                if not isinstance(weight_val, (int, float)):
                    logger.warning(f"Weight value for DFRS key '{key_str}' is not a number ({weight_val}). Skipping.")
                    continue
                converted_dict[enum_member] = float(weight_val)
            except ValueError:
                 # Try matching by name as a fallback if value match fails
                try:
                    enum_member_by_name = getattr(DFRSMetricsEnum, str(key_str).upper().strip().replace('-', '_'))
                    if not isinstance(weight_val, (int, float)):
                        logger.warning(f"Weight value for DFRS key '{key_str}' (matched by name) is not a number ({weight_val}). Skipping.")
                        continue
                    converted_dict[enum_member_by_name] = float(weight_val) # type: ignore
                except (AttributeError, ValueError):
                    logger.warning(f"Key '{key_str}' is not a valid DFRSMetricsV49 Enum member (value or name). This weight will be ignored.")
        return converted_dict

class InitialSelectionWeightsV49Model(BaseModel):
    weights: Dict['InitialSelectionKeysV49', float] = Field(default_factory=dict)
    model_config = ConfigDict(extra='ignore', validate_assignment=True, arbitrary_types_allowed=True)

    @field_validator('weights', mode='before')
    @classmethod
    def _convert_weight_keys_to_enum(cls, v: Any) -> Dict['InitialSelectionKeysV49', float]: # type: ignore
        if not isinstance(v, dict):
            if v is None: return {}
            raise ValueError("Initial selection weights must be a dictionary.")
        logger = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        converted_dict: Dict['InitialSelectionKeysV49', float] = {} # type: ignore
        KeysEnum: Optional[Type[enum.Enum]] = globals().get('InitialSelectionKeysV49') # type: ignore
        if not KeysEnum or not issubclass(KeysEnum, enum.Enum):
            logger.error("InitialSelectionKeysV49 Enum not found. Weights validation will be incomplete.")
            return v # type: ignore
        for key_str, weight_val in v.items():
            try:
                enum_member = KeysEnum(str(key_str).lower().strip()) # type: ignore
                if not isinstance(weight_val, (int, float)):
                    logger.warning(f"Weight value for InitialSelection key '{key_str}' is not a number ({weight_val}). Skipping.")
                    continue
                converted_dict[enum_member] = float(weight_val)
            except ValueError:
                try:
                    enum_member_by_name = getattr(KeysEnum, str(key_str).upper().strip().replace('-', '_'))
                    if not isinstance(weight_val, (int, float)):
                        logger.warning(f"Weight value for InitialSelection key '{key_str}' (matched by name) is not a number ({weight_val}). Skipping.")
                        continue
                    converted_dict[enum_member_by_name] = float(weight_val) # type: ignore
                except (AttributeError, ValueError):
                    logger.warning(f"Key '{key_str}' is not a valid InitialSelectionKeysV49 Enum member. This weight will be ignored.")
        return converted_dict

class FinalSelectionWeightsV49Model(BaseModel):
    weights: Dict['FinalSelectionKeysV49', float] = Field(default_factory=dict)
    model_config = ConfigDict(extra='ignore', validate_assignment=True, arbitrary_types_allowed=True)

    @field_validator('weights', mode='before')
    @classmethod
    def _convert_weight_keys_to_enum(cls, v: Any) -> Dict['FinalSelectionKeysV49', float]: # type: ignore
        if not isinstance(v, dict):
            if v is None: return {}
            raise ValueError("Final selection weights must be a dictionary.")
        logger = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        converted_dict: Dict['FinalSelectionKeysV49', float] = {} # type: ignore
        KeysEnum: Optional[Type[enum.Enum]] = globals().get('FinalSelectionKeysV49') # type: ignore
        if not KeysEnum or not issubclass(KeysEnum, enum.Enum):
            logger.error("FinalSelectionKeysV49 Enum not found. Weights validation will be incomplete.")
            return v # type: ignore
        for key_str, weight_val in v.items():
            try:
                enum_member = KeysEnum(str(key_str).lower().strip()) # type: ignore
                if not isinstance(weight_val, (int, float)):
                    logger.warning(f"Weight value for FinalSelection key '{key_str}' is not a number ({weight_val}). Skipping.")
                    continue
                converted_dict[enum_member] = float(weight_val)
            except ValueError:
                try:
                    enum_member_by_name = getattr(KeysEnum, str(key_str).upper().strip().replace('-', '_'))
                    if not isinstance(weight_val, (int, float)):
                        logger.warning(f"Weight value for FinalSelection key '{key_str}' (matched by name) is not a number ({weight_val}). Skipping.")
                        continue
                    converted_dict[enum_member_by_name] = float(weight_val) # type: ignore
                except (AttributeError, ValueError):
                     logger.warning(f"Key '{key_str}' is not a valid FinalSelectionKeysV49 Enum member. This weight will be ignored.")
        return converted_dict

class SelectionWeightsConfigV49(BaseModel):
    initial: 'InitialSelectionWeightsV49Model' = Field(default_factory=InitialSelectionWeightsV49Model)
    final: 'FinalSelectionWeightsV49Model' = Field(default_factory=FinalSelectionWeightsV49Model)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class FeatureFlagConfigV49(BaseModel):
    save_prompts: bool = Field(default=False)
    save_evaluations: bool = Field(default=False)
    save_rejected_candidates: bool = Field(default=True)
    dfrs_evaluation_enabled: bool = Field(default=True)
    dfrs_evaluate_all_loops: bool = Field(default=False)
    dfrs_for_initial_selection: bool = Field(default=False)
    advanced_nlp_enabled: bool = Field(default=True)
    ml_emotion_enabled: bool = Field(default=False)
    phase_tagging_enabled: bool = Field(default=True)
    persistent_cache_enabled: bool = Field(default=True)
    cache_cleanup_on_start: bool = Field(default=False)
    cache_vacuum_on_clear: bool = Field(default=True)
    json_export_enabled: bool = Field(default=True)
    json_schema_validation: bool = Field(default=False)
    dialogue_mode: Literal["normal", "delayed", "mixed", "auto"] = Field(default="auto")
    enable_resume_feature: bool = Field(default=True)
    resume_save_critical: bool = Field(default=False)
    llm_evaluation_enabled: bool = Field(default=True)
    subjectivity_analysis_enabled: bool = Field(default=True)
    fluctuation_analysis_enabled: bool = Field(default=True)
    subjective_focus_enabled: bool = Field(default=True)
    use_external_subjectivity_dict: bool = Field(default=True)
    use_external_fluctuation_dict: bool = Field(default=True)
    enable_filelock: bool = Field(default=True)
    enable_structured_output: bool = Field(default=True)
    enable_markdown_report: bool = Field(default=True)
    enable_stats_logging: bool = Field(default=True)
    phase_tone_prompt_modulation_enabled: bool = Field(default=True)
    normalize_subj_by_length: bool = Field(default=True)
    normalize_fluc_by_length: bool = Field(default=True)
    enable_subj_synonyms: bool = Field(default=True)
    enable_fluc_synonyms: bool = Field(default=False)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class FileSettingsV49(BaseModel):
    base_output_dir: Path = Field(default_factory=lambda: Path("./output_v49_alpha"))
    config_dir: Path = Field(default_factory=lambda: Path("./configs"))
    resources_dir: Path = Field(default_factory=lambda: Path("./configs/resources"))
    cache_dir: Path = Field(default_factory=lambda: Path("./cache/ndgs_v49_alpha"))
    persistent_cache_dir: Optional[Path] = Field(default_factory=lambda: Path("./cache/ndgs_v49_alpha/persistent"))
    log_filename: str = Field(default="ndgs_v49_alpha.log")
    log_max_bytes: int = Field(default=26214400) # 25MB
    log_backup_count: int = Field(default=10)
    stats_filename: str = Field(default="generation_stats_v49_alpha.jsonl")
    resume_dir_name: str = Field(default="resume")
    stats_dir_name: str = Field(default="stats")
    prompt_dir_name: str = Field(default="prompts")
    eval_dir_name: str = Field(default="evaluations")
    json_export_dir_name: str = Field(default="json_exports")
    rejected_dir_name: str = Field(default="rejected_candidates")
    rl_model_dir: str = Field(default="rl_models")
    subjectivity_keywords_path: Path = Field(default_factory=lambda: Path("./configs/resources/subjectivity_keywords_v49.yaml"))
    fluctuation_patterns_path: Path = Field(default_factory=lambda: Path("./configs/resources/fluctuation_patterns_v49.yaml"))
    analyzer_keywords_path: Optional[Path] = Field(default_factory=lambda: Path("./configs/resources/analyzer_keywords_v49.yaml"))
    lock_suffix: str = Field(default=".lock")
    resume_suffix: str = Field(default=".json")
    filename_max_length: Annotated[int, Field(gt=0, lt=256)] = Field(default=150)
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator(
        'base_output_dir', 'config_dir', 'resources_dir', 'cache_dir',
        'persistent_cache_dir', 'subjectivity_keywords_path',
        'fluctuation_patterns_path', 'analyzer_keywords_path',
        mode='before'
    )
    @classmethod
    def _ensure_path_objects(cls, v: Any, info: ValidationInfo) -> Optional[Path]:
        if v is None:
            # Allow None for optional Path fields like persistent_cache_dir and analyzer_keywords_path
            if info.field_name in ['persistent_cache_dir', 'analyzer_keywords_path']:
                return None
            else: # Other path fields are not optional by default based on usage
                raise ValueError(f"Path field '{info.field_name}' cannot be None.")
        if isinstance(v, str): return Path(v)
        if isinstance(v, Path): return v
        raise TypeError(f"Field '{info.field_name}' must be a string or Path object, got {type(v)}")

class ErrorRecoveryStrategy(BaseModel):
    retry_delay: Annotated[float, Field(ge=0.0)]
    max_retries: Annotated[int, Field(ge=0)]
    backoff_factor: float = Field(default=1.5, ge=1.0)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class ErrorConfigV49(BaseModel):
    severity: Dict[str, Literal["RECOVERABLE", "FATAL", "WARNING_ONLY"]] = Field(default_factory=dict)
    recovery_strategies: Dict[str, 'ErrorRecoveryStrategy'] = Field(default_factory=dict)
    max_consecutive_errors: Annotated[int, Field(ge=1)] = Field(default=3)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class ExternalConfigsV49(BaseModel):
    system_version_check: Optional[str] = None
    api_settings: 'APISettingsV49' = Field(default_factory=APISettingsV49)
    file_settings: 'FileSettingsV49' = Field(default_factory=FileSettingsV49)
    feature_flags: 'FeatureFlagConfigV49' = Field(default_factory=FeatureFlagConfigV49)
    dfrs_weights_config: 'DFRSWeightsConfigV49' = Field(default_factory=DFRSWeightsConfigV49)
    temperature_config: 'TemperatureStrategyConfigV49' = Field(default_factory=TemperatureStrategyConfigV49)
    adaptation_config: 'AdaptationStrategyConfigV49' = Field(default_factory=AdaptationStrategyConfigV49)
    feedback_config: 'FeedbackStrategyConfigV49' = Field(default_factory=FeedbackStrategyConfigV49)
    selection_weights_config: 'SelectionWeightsConfigV49' = Field(default_factory=SelectionWeightsConfigV49)
    error_config: 'ErrorConfigV49' = Field(default_factory=ErrorConfigV49)
    prompt_style_presets: Dict[str, Dict[str, Union[str, List[str]]]] = Field(default_factory=dict)
    cache_config: Dict[str, Any] = Field(default_factory=lambda: {"max_size_mb": 500.0, "auto_cleanup_enabled": True, "dfrs_cache_ttl_seconds": 259200})
    eti_keywords: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    symbolic_density_keywords: List[str] = Field(default_factory=list)
    phase_tone_prompt_templates: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    phase_transition_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    style_suggestion_keyword_map: Dict[str, List[Union[List[str], float]]] = Field(default_factory=dict)
    default_subjective_intensity: 'SubjectiveIntensityLevel' = Field(default_factory=lambda: globals().get('SubjectiveIntensityLevel').MEDIUM) # type: ignore
    default_style_template: Annotated[str, Field(min_length=1)] = Field(default="standard")
    nlp_model_name: Annotated[str, Field(min_length=1)] = Field(default="ja_core_news_lg")
    ml_emotion_model: Optional[Annotated[str, Field(min_length=1)]] = None
    use_lightweight_ml_model: bool = Field(default=False)
    generation_params: Dict[str, Any] = Field(default_factory=lambda: {
        "target_length": 4000, "feedback_loops": 3,
        "min_feedback_loops": 1, "min_score_threshold": 4.5
    })
    execution_config: Dict[str, Any] = Field(default_factory=lambda: {"max_consecutive_errors": 5})
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True, validate_assignment=True)

# ----------------------------------------
# -- Part 3b: Subjectivity/Fluctuation Resource Models
# ----------------------------------------
class SubjectivityKeywordEntryV49(BaseModel):
    keyword: Optional[Annotated[str, Field(min_length=1)]] = None
    pattern: Optional[Annotated[str, Field(min_length=1)]] = None
    use_regex: bool = Field(default=False)
    intensity: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    category: 'SubjectivityCategoryV49'
    related_phases: List['PsychologicalPhaseV49'] = Field(default_factory=list)
    related_tones: List['EmotionalToneV49'] = Field(default_factory=list)
    context_tags: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    example_usage: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True, validate_assignment=True, str_strip_whitespace=True)

    @model_validator(mode='before')
    @classmethod
    def _check_keyword_or_pattern(cls, values: Any) -> Any: # Renamed for clarity
        logger_val = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        if not isinstance(values, dict): raise ValueError(f"{cls.__name__}: Input must be a dict.")
        kw, pat = values.get('keyword'), values.get('pattern')
        has_kw = kw is not None and str(kw).strip() != ""
        has_pat = pat is not None and str(pat).strip() != ""
        if not has_kw and not has_pat: raise ValueError(f"{cls.__name__}: 'keyword' or 'pattern' is required.")
        if has_kw and has_pat: raise ValueError(f"{cls.__name__}: Cannot specify both 'keyword' and 'pattern'.")
        
        # Default use_regex to True if pattern is provided and use_regex is not explicitly set
        if has_pat and values.get('use_regex') is None:
            values['use_regex'] = True
        
        if has_pat and values.get('use_regex') is True:
            try: re.compile(str(pat))
            except re.error as e: raise ValueError(f"Invalid regex pattern '{pat}': {e}")
        elif has_kw: # If only keyword is present, use_regex should be False
            if values.get('use_regex') is True:
                logger_val.warning(f"{cls.__name__} (keyword: '{kw}'): 'use_regex' is True but only 'keyword' is provided. Setting 'use_regex' to False.")
            values['use_regex'] = False
        return values

    @field_validator('category', 'related_phases', 'related_tones', mode='before')
    @classmethod
    def _convert_enum_fields(cls, v: Any, info: ValidationInfo) -> Any:
        logger_val = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}._convert_enum_fields")
        enum_map = {
            'category': 'SubjectivityCategoryV49',
            'related_phases': 'PsychologicalPhaseV49',
            'related_tones': 'EmotionalToneV49'
        }
        field_name = str(info.field_name)
        target_enum_cls_name = enum_map.get(field_name)
        if not target_enum_cls_name:
            raise RuntimeError(f"Unknown field '{field_name}' in enum validator of {cls.__name__}.")

        target_enum_cls: Optional[Type[enum.Enum]] = globals().get(target_enum_cls_name) # type: ignore
        entry_id_parts = []
        if isinstance(info.data, dict):
            if info.data.get('keyword'): entry_id_parts.append(f"keyword='{info.data.get('keyword')}'")
            if info.data.get('pattern'): entry_id_parts.append(f"pattern='{info.data.get('pattern')}'")
        entry_id = ", ".join(entry_id_parts) if entry_id_parts else "UnknownEntry"


        if not target_enum_cls or not issubclass(target_enum_cls, enum.Enum):
            msg = f"CRITICAL ({cls.__name__}): Enum class '{target_enum_cls_name}' for field '{field_name}' not found or not an Enum. Entry: '{entry_id}'"
            logger_val.critical(msg)
            raise RuntimeError(msg)

        def _converter(item_str_raw: Any) -> Optional[enum.Enum]:
            if isinstance(item_str_raw, target_enum_cls): return item_str_raw
            if not isinstance(item_str_raw, str):
                logger_val.warning(f"{cls.__name__} - Field '{field_name}', entry '{entry_id}': Value '{item_str_raw!r}' (type: {type(item_str_raw)}) is not a string. Skipping.")
                return None
            normalized = item_str_raw.lower().strip().replace('-', '_').replace(' ', '_')
            try: return target_enum_cls(normalized) # type: ignore
            except ValueError:
                try: return getattr(target_enum_cls, item_str_raw.upper().strip().replace('-', '_').replace(' ', '_'))
                except (AttributeError, ValueError):
                    logger_val.warning(f"{cls.__name__} - Field '{field_name}', entry '{entry_id}': Invalid Enum value/name '{item_str_raw}' for {target_enum_cls_name}. Skipping.")
                    return None
        
        if field_name == 'category':
            converted = _converter(v)
            if converted is None: raise ValueError(f"Field '{field_name}' (entry: '{entry_id}') has invalid value '{v}' for Enum {target_enum_cls_name}.")
            return converted
        elif isinstance(v, list):
            return [converted_item for item in v if (converted_item := _converter(item)) is not None]
        elif v is None and field_name in ['related_phases', 'related_tones']: # Allow empty lists for optional list fields
            return []
        
        raise TypeError(f"{cls.__name__} - Field '{field_name}', entry '{entry_id}': Value '{v!r}' type invalid for {target_enum_cls_name} or List[{target_enum_cls_name}].")


class SubjectivityKeywordsFileV49(RootModel[Dict['SubjectivityCategoryV49', List['SubjectivityKeywordEntryV49']]]): # type: ignore
    root: Dict['SubjectivityCategoryV49', List['SubjectivityKeywordEntryV49']] # type: ignore
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    @classmethod
    def _parse_yaml_structure(cls, data: Any) -> Dict['SubjectivityCategoryV49', List[Dict[str, Any]]]: # type: ignore
        logger_val = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        if not isinstance(data, dict):
            raise ValueError("主観性キーワードファイルのトップレベルは辞書である必要があります。")

        SubjCatEnum: Optional[Type[enum.Enum]] = globals().get('SubjectivityCategoryV49') # type: ignore
        if not SubjCatEnum or not issubclass(SubjCatEnum, enum.Enum):
            raise ImportError("SubjectivityCategoryV49 Enum が見つかりません。")

        processed_data: Dict['SubjectivityCategoryV49', List[Dict[str, Any]]] = defaultdict(list) # type: ignore
        for cat_str, entries in data.items():
            cat_enum_member: Optional['SubjectivityCategoryV49'] = None # type: ignore
            try:
                cat_enum_member = SubjCatEnum(str(cat_str).lower().strip()) # type: ignore
            except ValueError:
                try: cat_enum_member = getattr(SubjCatEnum, str(cat_str).upper().strip().replace('-', '_')) # type: ignore
                except (AttributeError, ValueError):
                    logger_val.warning(f"YAML内の未知の主観性カテゴリキー: '{cat_str}'。このカテゴリは無視されます。")
                    continue
            
            if not isinstance(entries, list):
                logger_val.warning(f"カテゴリ '{cat_str}' のエントリがリストではありません。無視されます。")
                continue

            for entry_dict in entries:
                if isinstance(entry_dict, dict):
                    entry_copy = entry_dict.copy()
                    entry_copy['category'] = cat_enum_member.value # type: ignore # Ensure category matches key for sub-model validation
                    processed_data[cat_enum_member].append(entry_copy)
                else:
                    logger_val.warning(f"カテゴリ '{cat_str}' 内の不正なエントリ型: {type(entry_dict)}。無視されます。")
        return dict(processed_data)

# --- Fluctuation Pattern Models ---
class FluctuationPatternEntryV49(BaseModel):
    keyword: Optional[Annotated[str, Field(min_length=1)]] = None
    pattern: Optional[Annotated[str, Field(min_length=1)]] = None
    use_regex: bool = Field(default=False)
    intensity: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    category: 'FluctuationCategoryV49'
    related_phases: List['PsychologicalPhaseV49'] = Field(default_factory=list)
    related_tones: List['EmotionalToneV49'] = Field(default_factory=list)
    context_tags: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    example_usage: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True, validate_assignment=True, str_strip_whitespace=True)

    # Define the validator directly in this class
    @model_validator(mode='before')
    @classmethod
    def _check_keyword_or_pattern_for_fluctuation(cls, values: Any) -> Any: # Renamed to be specific
        logger_val = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        if not isinstance(values, dict): raise ValueError(f"{cls.__name__}: Input must be a dict.")
        kw, pat = values.get('keyword'), values.get('pattern')
        has_kw = kw is not None and str(kw).strip() != ""
        has_pat = pat is not None and str(pat).strip() != ""
        if not has_kw and not has_pat: raise ValueError(f"{cls.__name__}: 'keyword' or 'pattern' is required.")
        if has_kw and has_pat: raise ValueError(f"{cls.__name__}: Cannot specify both 'keyword' and 'pattern'.")
        
        if has_pat and values.get('use_regex') is None:
            values['use_regex'] = True
        
        if has_pat and values.get('use_regex') is True:
            try: re.compile(str(pat))
            except re.error as e: raise ValueError(f"Invalid regex pattern '{pat}': {e}")
        elif has_kw:
            if values.get('use_regex') is True:
                logger_val.warning(f"{cls.__name__} (keyword: '{kw}'): 'use_regex' is True but only 'keyword' is provided. Setting 'use_regex' to False.")
            values['use_regex'] = False
        return values

    @field_validator('category', 'related_phases', 'related_tones', mode='before')
    @classmethod
    def _convert_enum_fields_for_fluctuation(cls, v: Any, info: ValidationInfo) -> Any: # Renamed
        logger_val = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}._convert_enum_fields_for_fluctuation")
        enum_map = {
            'category': 'FluctuationCategoryV49',
            'related_phases': 'PsychologicalPhaseV49',
            'related_tones': 'EmotionalToneV49'
        }
        field_name = str(info.field_name)
        target_enum_cls_name = enum_map.get(field_name)
        if not target_enum_cls_name:
             raise RuntimeError(f"Unknown field '{field_name}' in enum validator of {cls.__name__}.")

        target_enum_cls: Optional[Type[enum.Enum]] = globals().get(target_enum_cls_name) # type: ignore
        entry_id = (info.data.get('keyword') or info.data.get('pattern') or "UnknownEntry") if isinstance(info.data, dict) else "UnknownEntry"

        if not target_enum_cls or not issubclass(target_enum_cls, enum.Enum):
            msg = f"CRITICAL ({cls.__name__}): Enum class '{target_enum_cls_name}' for field '{field_name}' not found or not an Enum. Entry: '{entry_id}'"
            logger_val.critical(msg)
            raise RuntimeError(msg)

        def _converter(item_str_raw: Any) -> Optional[enum.Enum]:
            if isinstance(item_str_raw, target_enum_cls): return item_str_raw
            if not isinstance(item_str_raw, str):
                logger_val.warning(f"{cls.__name__} - Field '{field_name}', entry '{entry_id}': Value '{item_str_raw!r}' is not a string. Skipping.")
                return None
            normalized = item_str_raw.lower().strip().replace('-', '_').replace(' ', '_')
            try: return target_enum_cls(normalized) # type: ignore
            except ValueError:
                try: return getattr(target_enum_cls, item_str_raw.upper().strip().replace('-', '_').replace(' ', '_'))
                except (AttributeError, ValueError):
                    logger_val.warning(f"{cls.__name__} - Field '{field_name}', entry '{entry_id}': Invalid Enum value/name '{item_str_raw}' for {target_enum_cls_name}. Skipping.")
                    return None
        
        if field_name == 'category':
            converted = _converter(v)
            if converted is None: raise ValueError(f"Field '{field_name}' (entry: '{entry_id}') has invalid value '{v}' for Enum {target_enum_cls_name}.")
            return converted
        elif isinstance(v, list):
            return [converted_item for item in v if (converted_item := _converter(item)) is not None]
        elif v is None and field_name in ['related_phases', 'related_tones']:
            return []
        raise TypeError(f"{cls.__name__} - Field '{field_name}', entry '{entry_id}': Value '{v!r}' type invalid for {target_enum_cls_name} or List[{target_enum_cls_name}].")

class FluctuationPatternsFileV49(RootModel[Dict['FluctuationCategoryV49', List['FluctuationPatternEntryV49']]]): # type: ignore
    root: Dict['FluctuationCategoryV49', List['FluctuationPatternEntryV49']] # type: ignore
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    @classmethod
    def _parse_yaml_structure_fluctuation(cls, data: Any) -> Dict['FluctuationCategoryV49', List[Dict[str, Any]]]: # type: ignore
        logger_val = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
        if not isinstance(data, dict):
            raise ValueError("揺らぎパターンファイルのトップレベルは辞書である必要があります。")

        FlucCatEnum: Optional[Type[enum.Enum]] = globals().get('FluctuationCategoryV49') # type: ignore
        if not FlucCatEnum or not issubclass(FlucCatEnum, enum.Enum):
            raise ImportError("FluctuationCategoryV49 Enum が見つかりません。")

        processed_data: Dict['FluctuationCategoryV49', List[Dict[str, Any]]] = defaultdict(list) # type: ignore
        for cat_str, entries in data.items():
            cat_enum_member: Optional['FluctuationCategoryV49'] = None # type: ignore
            try:
                cat_enum_member = FlucCatEnum(str(cat_str).lower().strip()) # type: ignore
            except ValueError:
                try: cat_enum_member = getattr(FlucCatEnum, str(cat_str).upper().strip().replace('-', '_')) # type: ignore
                except (AttributeError, ValueError):
                    logger_val.warning(f"YAML内の未知の揺らぎカテゴリキー: '{cat_str}'。このカテゴリは無視されます。")
                    continue
            
            if not isinstance(entries, list):
                logger_val.warning(f"カテゴリ '{cat_str}' のエントリがリストではありません。無視されます。")
                continue

            for entry_dict in entries:
                if isinstance(entry_dict, dict):
                    entry_copy = entry_dict.copy()
                    entry_copy['category'] = cat_enum_member.value # type: ignore
                    processed_data[cat_enum_member].append(entry_copy)
                else:
                    logger_val.warning(f"カテゴリ '{cat_str}' 内の不正なエントリ型: {type(entry_dict)}。無視されます。")
        return dict(processed_data)

# --- Part 3b 終了点 ---
# =============================================================================
# -- Part 3c: Core Domain Models (Pydantic V2 Optimized & Fixed)
# =============================================================================
# (Assumes Enums from Part 1b and BaseModel, Field, Annotated, etc. from Part 0 are available)
# (Assumes Part 3a and 3b have been defined before this section)

# --- Input Models (Should be defined before models that use them, e.g., GeneratorStateV49) ---
class CharacterV49(BaseModel):
    name: Annotated[str, Field(min_length=1)]
    age: Optional[Union[Annotated[int, Field(ge=0)], str]] = None
    gender: Optional[str] = None
    appearance: Optional[str] = None
    role: Optional[str] = None
    personalityTraits: List[str] = Field(default_factory=list)
    abilities: List[str] = Field(default_factory=list)
    storyRole: Optional[str] = None
    motivations: List[str] = Field(default_factory=list)
    internalConflicts: List[str] = Field(default_factory=list)
    values: List[str] = Field(default_factory=list)
    pastExperiences: Optional[str] = None
    duality: Optional[str] = None
    pronouns: Optional[str] = None
    speechPatterns: Optional[str] = None
    nonverbalCues: List[str] = Field(default_factory=list)
    relationships: Optional[str] = None
    communicationStyle: Optional[str] = None
    emotionalExpressions: Optional[str] = None
    emotionalPatterns: Optional[str] = None
    dialogueStyleShift: Optional[str] = None
    taboos: List[str] = Field(default_factory=list)
    subjective_tendencies: Optional[str] = None
    internal_monologue_style: Optional[str] = None
    perception_filter_bias: Optional[str] = None
    emotion_expression_method: Optional[str] = None
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class SceneInfoV49(BaseModel):
    name: Optional[Annotated[str, Field(min_length=1)]] = None
    location: Optional[str] = None
    time: Optional[str] = None
    atmosphere: Optional[str] = None
    environmentElements: List[str] = Field(default_factory=list)
    spatialCharacteristics: Optional[str] = None
    sensoryElements: Dict[str, str] = Field(default_factory=dict)
    purpose: Optional[str] = None
    previousEvents: Optional[str] = None
    characterPositions: Optional[str] = None
    emotionalStates: Dict[str, str] = Field(default_factory=dict)
    dialogueObjectives: Dict[str, str] = Field(default_factory=dict)
    foreshadowingElements: List[str] = Field(default_factory=list)
    dialogueStructure: Optional[str] = None
    dialoguePivots: Optional[str] = None
    constraints: Optional[str] = None
    notes: Optional[str] = None
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class InputDataV49(BaseModel):
    characterA: 'CharacterV49'
    characterB: 'CharacterV49'
    sceneInfo: 'SceneInfoV49'
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

class DialogueTurnInputV49(BaseModel):
    turn_id: Annotated[int, Field(ge=0)]
    speaker: Annotated[str, Field(min_length=1)]
    dialogue_text: Annotated[str, Field(min_length=1)]
    internal_monologue: Optional[str] = None
    action_description: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    target_phase: Optional[str] = None
    target_tone: Optional[str] = None
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

# --- LLM/DFRS Evaluation Score Models ---
class LLMEvaluationScoresV49(BaseModel):
    consistency: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    naturalness: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    emotionalDepth: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    constraints: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    attractiveness: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    complexity: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    overall: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    raw_output: Optional[str] = Field(default=None, exclude=True)
    model_config = ConfigDict(extra='allow', validate_assignment=True)

class DFRSSubScoresV49(BaseModel):
    _DFRSMetricsCls: ClassVar[Optional[Type[enum.Enum]]] = globals().get('DFRSMetricsV49')
    scores: Dict[str, Optional[float]] = Field(default_factory=dict, description="Raw scores dictionary, keys are DFRSMetricsV49.value")
    model_config = ConfigDict(extra='allow', populate_by_name=True, arbitrary_types_allowed=True, validate_assignment=True)

    @model_validator(mode='after')
    def _sync_individual_fields_with_scores(self) -> 'DFRSSubScoresV49':
        return self
    
    def get_score(self, metric: 'DFRSMetricsV49') -> Optional[float]: # type: ignore
        if not isinstance(metric, globals().get('DFRSMetricsV49', type(None))): # type: ignore
            logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}").warning(
                f"Invalid metric type provided to get_score: {type(metric)}. Expected DFRSMetricsV49 Enum member."
            )
            return None
        return self.scores.get(metric.value)


# --- Analysis Result Component Models ---
class EmotionCurvePointV49(BaseModel):
    block_index: Annotated[int, Field(ge=0)]
    tone: 'EmotionalToneV49' # type: ignore
    strength: Annotated[float, Field(ge=0.0, le=1.0)]
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True, validate_assignment=True)

class PhaseTimelinePointV49(BaseModel):
    block_index: Annotated[int, Field(ge=0)]
    phase: 'PsychologicalPhaseV49' # type: ignore
    confidence: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True, validate_assignment=True)

class BlockAnalysisTagsV49(BaseModel):
    matched_subjectivity_categories: List['SubjectivityCategoryV49'] = Field(default_factory=list) # type: ignore
    matched_fluctuation_categories: List['FluctuationCategoryV49'] = Field(default_factory=list) # type: ignore
    estimated_phase: Optional['PsychologicalPhaseV49'] = None # type: ignore
    estimated_tone: Optional['EmotionalToneV49'] = None # type: ignore
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True, validate_assignment=True)

class DialogueBlockV49(BaseModel):
    type: Literal["speech", "description"]
    text: str
    length: Optional[Annotated[int, Field(ge=0)]] = None
    speaker: Optional[str] = None
    analysis_tags: Optional['BlockAnalysisTagsV49'] = Field(default_factory=BlockAnalysisTagsV49)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

    @model_validator(mode='after')
    def _process_block_logic(self) -> 'DialogueBlockV49':
        # RecursionError 対策: object.__setattr__ を使用して Pydantic の __setattr__ をバイパス
        current_length = getattr(self, 'length', None) # 直接 self.length を参照しない
        if current_length is None and self.text is not None:
            object.__setattr__(self, 'length', len(self.text))
        
        if self.type == "description":
            if getattr(self, 'speaker', object()) is not None: # speakerがNoneでない場合のみ設定
                object.__setattr__(self, 'speaker', None)
        elif self.type == "speech":
            current_speaker = getattr(self, 'speaker', None)
            if not current_speaker or not str(current_speaker).strip():
                raise ValueError("Speech block must have a non-empty speaker.")
        return self

class SpeechBlockV49(DialogueBlockV49):
    type: Literal["speech"] = "speech"
    speaker: Annotated[str, Field(min_length=1)]

class DescriptionBlockV49(DialogueBlockV49):
    type: Literal["description"] = "description"
    # speaker is implicitly None due to DialogueBlockV49's validator

# --- Generator State Models ---
class VersionStateV49(BaseModel):
    version_id: Annotated[int, Field(ge=0)]
    generated_text: Optional[str] = None
    prompt_text: Optional[str] = Field(default=None, exclude=True)
    feedback_text: Optional[str] = Field(default=None, exclude=True)
    evaluation_text_raw: Optional[str] = None
    llm_scores: Optional['LLMEvaluationScoresV49'] = Field(default=None)
    dfrs_scores: Optional['DFRSSubScoresV49'] = Field(default=None)
    generation_time_ms: Optional[Annotated[float, Field(ge=0.0)]] = None
    evaluation_time_ms: Optional[Annotated[float, Field(ge=0.0)]] = None
    generation_model: Optional[str] = None
    status: Literal["pending", "generating", "evaluating", "completed", "error"] = Field(default="pending")
    error_info: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analyzer_results: Optional[Dict[str, Any]] = Field(default_factory=dict, exclude=True)
    estimated_subjectivity: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    estimated_fluctuation: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True, validate_assignment=True)

class PhaseTransitionRecordV49(BaseModel):
    loop_number: Annotated[int, Field(ge=0)]
    from_phase: Optional['PsychologicalPhaseV49'] = None # type: ignore
    from_tone: Optional['EmotionalToneV49'] = None # type: ignore
    to_phase: Optional['PsychologicalPhaseV49'] = None # type: ignore
    to_tone: Optional['EmotionalToneV49'] = None # type: ignore
    reward: Annotated[float, Field(ge=0.0, le=1.0)]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True, validate_assignment=True)

class OutputEvaluationV49(BaseModel):
    final_eodf_v49: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    llm_scores: Optional['LLMEvaluationScoresV49'] = None
    dfrs_scores: Optional['DFRSSubScoresV49'] = None
    emotion_curve: List['EmotionCurvePointV49'] = Field(default_factory=list)
    topic_shifts: List[Dict[str, Any]] = Field(default_factory=list)
    phase_timeline: List['PhaseTimelinePointV49'] = Field(default_factory=list)
    phase_tags_summary: List['PsychologicalPhaseV49'] = Field(default_factory=list) # type: ignore
    evaluation_feedback: Optional[str] = None
    model_config = ConfigDict(extra='allow', populate_by_name=True, arbitrary_types_allowed=True, validate_assignment=True)

    @model_validator(mode='after')
    def _populate_final_eodf_from_dfrs(self) -> 'OutputEvaluationV49':
        if self.final_eodf_v49 is None and self.dfrs_scores and self.dfrs_scores.scores:
            DFRSMetricsEnum_cls: Optional[Type[enum.Enum]] = globals().get('DFRSMetricsV49') # type: ignore
            if DFRSMetricsEnum_cls:
                final_eodf_key_enum = getattr(DFRSMetricsEnum_cls, 'FINAL_EODF_V49', None)
                if final_eodf_key_enum:
                    val = self.dfrs_scores.scores.get(final_eodf_key_enum.value) # type: ignore
                    if isinstance(val, (int, float)):
                        self.final_eodf_v49 = float(val)
        return self

class GenerationStatsV49(BaseModel):
    loops: Annotated[int, Field(ge=0)]
    overall: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    final_eodf_v49: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    subjectivity_score: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    fluctuation_intensity: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    expression_richness: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    content_novelty: Optional[Annotated[float, Field(ge=0.0, le=5.0)]] = None
    error_count: Annotated[int, Field(ge=0)] = Field(default=0)
    duration_seconds: Optional[Annotated[float, Field(ge=0.0)]] = None
    avg_api_call_time_ms: Optional[Annotated[float, Field(ge=0.0)]] = None
    total_tokens_processed: Optional[Annotated[int, Field(ge=0)]] = None
    model_config = ConfigDict(extra='allow', validate_assignment=True)

    @model_validator(mode='before')
    @classmethod
    def _populate_from_evaluation(cls, data: Any, info: ValidationInfo) -> Any:
        if isinstance(data, dict):
            if data.get('overall') is None and 'llm_scores' in data and isinstance(data['llm_scores'], dict):
                data['overall'] = data['llm_scores'].get('overall')
            
            if 'dfrs_scores' in data and isinstance(data['dfrs_scores'], dict) and 'scores' in data['dfrs_scores'] and isinstance(data['dfrs_scores']['scores'], dict):
                dfrs_values = data['dfrs_scores']['scores']
                DFRSMetricsEnum_cls: Optional[Type[enum.Enum]] = globals().get('DFRSMetricsV49') # type: ignore
                if DFRSMetricsEnum_cls:
                    metric_map = {
                        'final_eodf_v49': getattr(DFRSMetricsEnum_cls, 'FINAL_EODF_V49', None),
                        'subjectivity_score': getattr(DFRSMetricsEnum_cls, 'SUBJECTIVITY_SCORE', None),
                        'fluctuation_intensity': getattr(DFRSMetricsEnum_cls, 'FLUCTUATION_INTENSITY', None),
                        'expression_richness': getattr(DFRSMetricsEnum_cls, 'EXPRESSION_RICHNESS', None),
                        'content_novelty': getattr(DFRSMetricsEnum_cls, 'CONTENT_NOVELTY', None),
                    }
                    for field_name, enum_member in metric_map.items():
                        if data.get(field_name) is None and enum_member:
                            data[field_name] = dfrs_values.get(enum_member.value) # type: ignore
        return data

class GeneratorStateV49(BaseModel):
    job_id: Annotated[str, Field(min_length=1)]
    system_version: str
    model_name: str
    start_time: datetime
    input_data: 'InputDataV49'
    target_length: Annotated[int, Field(gt=0)]
    settings_snapshot: Dict[str, Any] = Field(default_factory=dict)
    current_loop: Annotated[int, Field(ge=0)] = Field(default=0)
    current_intended_phase: Optional['PsychologicalPhaseV49'] = None # type: ignore
    current_intended_tone: Optional['EmotionalToneV49'] = None # type: ignore
    temperature_history: List[float] = Field(default_factory=list)
    adj_factor_history: List[Dict[str, Any]] = Field(default_factory=list)
    versions: List['VersionStateV49'] = Field(default_factory=list)
    complete: bool = Field(default=False)
    completion_time: Optional[datetime] = None
    final_version: Optional[Annotated[int, Field(ge=0)]] = None
    final_score_llm: Optional[float] = None
    final_dfrs_scores: Optional[Dict[str, Optional[float]]] = Field(default_factory=dict)
    final_evaluation_summary: Optional['OutputEvaluationV49'] = None
    final_generation_stats: Optional['GenerationStatsV49'] = None
    final_output_json_path: Optional[str] = None
    last_error: Optional[Dict[str, Any]] = None
    error_records: List[Dict[str, Any]] = Field(default_factory=list)
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator('versions', mode='before')
    @classmethod
    def _convert_versions_dict_to_list(cls, v: Any, info: ValidationInfo) -> List[Any]:
        if isinstance(v, dict):
            logger_val = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}")
            try:
                return [item_data for _, item_data in sorted(v.items(), key=lambda x: int(str(x[0])))]
            except (ValueError, TypeError) as e:
                logger_val.warning(
                    f"Field '{info.field_name}': Error sorting version dictionary by keys: {e}. "
                    f"Data preview: {str(v)[:200]}. Returning empty list."
                )
                return []
        if v is None: return []
        if not isinstance(v, list):
            raise TypeError(f"Field '{info.field_name}' must be a list or a dict convertible to a list.")
        return v

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        dumped_data = super().model_dump(*args, **kwargs)
        if 'settings_snapshot' in dumped_data and isinstance(dumped_data['settings_snapshot'], dict):
            # AttributeError: 'GeneratorStateV49' object has no attribute '__qualname__' 修正
            logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}.model_dump")
            cleaned_snapshot = {
                k: v_item for k, v_item in dumped_data['settings_snapshot'].items()
                if not callable(v_item)
            }
            if len(cleaned_snapshot) < len(dumped_data['settings_snapshot']):
                 logger.debug("Callable items were excluded from settings_snapshot during model_dump.")
            dumped_data['settings_snapshot'] = cleaned_snapshot
        return dumped_data

# --- Feedback Context ---
class FeedbackContextV49(BaseModel):
    version: Annotated[int, Field(ge=0)]
    dialogue_text: str
    intended_phase: Optional['PsychologicalPhaseV49'] = None # type: ignore
    intended_tone: Optional['EmotionalToneV49'] = None # type: ignore
    inferred_phase: Optional['PsychologicalPhaseV49'] = None # type: ignore
    inferred_tone: Optional['EmotionalToneV49'] = None # type: ignore
    dfrs_scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    llm_scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True, validate_assignment=True)

# --- Final Output JSON Structure Models ---
class OutputCharacterContextV49(BaseModel):
    name: Annotated[str, Field(min_length=1)]
    description: Optional[str] = None
    personalityTraits: Optional[List[str]] = Field(default=None, alias="personality")
    speechPatterns: Optional[str] = Field(default=None, alias="speech_style")
    model_config = ConfigDict(populate_by_name=True, extra='ignore', validate_assignment=True)

class OutputSceneContextV49(BaseModel):
    location: Optional[str] = None
    time: Optional[str] = None
    atmosphere: Optional[str] = None
    purpose: Optional[str] = None
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class InputContextV49(BaseModel):
    character_sheets: List['OutputCharacterContextV49']
    scene_sheet: 'OutputSceneContextV49'
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class OutputDialogueV49(BaseModel):
    dialogue_blocks: List[Union['SpeechBlockV49', 'DescriptionBlockV49']]
    total_length: Annotated[int, Field(ge=0)]
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class SettingsMetadataV49(BaseModel):
    dialogue_mode: Optional[str] = None
    style_template: Optional[str] = None
    subjective_focus: Optional[bool] = None
    subjective_intensity: Optional[str] = None # Should store SubjectiveIntensityLevel.value
    adaptation_strategy_type: Optional[str] = None
    feedback_strategy_type: Optional[str] = None
    target_length: Optional[Annotated[int, Field(gt=0)]] = None
    feedback_loops: Optional[Annotated[int, Field(ge=0)]] = None
    model_config = ConfigDict(extra='allow', validate_assignment=True)

class OutputMetadataV49(BaseModel):
    job_id: Annotated[str, Field(min_length=1)]
    generation_time: datetime
    model_used: Optional[str] = None
    system_version: Optional[str] = None
    generation_stats: Optional['GenerationStatsV49'] = None
    settings_applied: Optional['SettingsMetadataV49'] = Field(default=None, alias="settings")
    model_config = ConfigDict(populate_by_name=True, extra='ignore', validate_assignment=True)

class OutputJsonStructureV49(BaseModel):
    metadata: 'OutputMetadataV49'
    input_context: 'InputContextV49'
    output_dialogue: 'OutputDialogueV49'
    evaluation: Optional['OutputEvaluationV49'] = None
    notes: Optional[str] = None
    error_records: List[Dict[str, Any]] = Field(default_factory=list)
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

# ----------------------------------------
# Utility Models
# ----------------------------------------
class CacheDataV49(BaseModel):
    key: Annotated[str, Field(min_length=1)]
    value: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

class PromptComponentsV49(BaseModel):
    system_prompt: str = Field(default="")
    scene_setting: str = Field(default="")
    character_profiles: str = Field(default="")
    dialogue_history: str = Field(default="")
    current_turn_context: str = Field(default="")
    adaptation_feedback: str = Field(default="")
    instruction_prompt: str = Field(default="")
    style_guide_prompt: str = Field(default="")
    custom_directives: List[str] = Field(default_factory=list)
    output_format_instruction: str = Field(default="")
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

_MODELS_DEFINED_IN_PART3 = [
    FixedTemperatureParams, DecreasingTemperatureParams, TwoDimensionalTemperatureParams,
    TemperatureStrategyConfigV49, ProbabilisticHistoryAdaptationParams, AdaptationStrategyConfigV49,
    CompositeFeedbackParams, PhaseToneFeedbackParams, FeedbackStrategyConfigV49, APISettingsV49,
    DFRSWeightsConfigV49, InitialSelectionWeightsV49Model, FinalSelectionWeightsV49Model,
    SelectionWeightsConfigV49, FeatureFlagConfigV49, FileSettingsV49, ErrorRecoveryStrategy,
    ErrorConfigV49, ExternalConfigsV49, SubjectivityKeywordEntryV49, SubjectivityKeywordsFileV49,
    FluctuationPatternEntryV49, FluctuationPatternsFileV49, CharacterV49, SceneInfoV49,
    InputDataV49, DialogueTurnInputV49, LLMEvaluationScoresV49, DFRSSubScoresV49,
    EmotionCurvePointV49, PhaseTimelinePointV49, BlockAnalysisTagsV49, DialogueBlockV49,
    SpeechBlockV49, DescriptionBlockV49, VersionStateV49, PhaseTransitionRecordV49,
    OutputEvaluationV49, GenerationStatsV49, GeneratorStateV49, FeedbackContextV49,
    OutputCharacterContextV49, OutputSceneContextV49, InputContextV49, OutputDialogueV49,
    SettingsMetadataV49, OutputMetadataV49, OutputJsonStructureV49, CacheDataV49,
    PromptComponentsV49
]

for model_cls_to_rebuild in _MODELS_DEFINED_IN_PART3:
    if hasattr(model_cls_to_rebuild, 'model_rebuild'):
        try:
            model_cls_to_rebuild.model_rebuild(force=True)
        except Exception as e_rebuild:
            logging.getLogger(__name__).warning(
                f"Warning during model_rebuild for {model_cls_to_rebuild.__name__}: {e_rebuild}"
            )

# =============================================================================
# -- Part 3c 終了点 (Core Domain Models)
# -- Part 3 終了点 (全体)
# =============================================================================
# =============================================================================
# -- Part 4: API Client & Interfaces (v4.9α - 最適化版)
# =============================================================================
# API Clientクラス定義と、主要コンポーネント間の依存性を定義する
# インターフェース(Protocol)を定義。
# 最適化版では、型ヒントの厳密化、ApiClientのエラーハンドリング強化、
# 設定参照の明確化を行う。

# --- 型チェック用の前方参照 ---
if TYPE_CHECKING:
    # Part 0, 1, 2, 3 で定義される型やEnum
    # これらはスクリプトの先頭部分で適切にインポートまたは定義されていることを期待
    from __main__ import (
        # Enums (Part 1b で定義されている想定)
        PsychologicalPhaseV49, EmotionalToneV49, SubjectivityCategoryV49,
        FluctuationCategoryV49, DFRSMetricsV49, ScoreKeys,
        SubjectiveIntensityLevel, FinalSelectionKeysV49, InitialSelectionKeysV49,
        # Pydantic Models (Part 3 で定義されている想定)
        ExternalConfigsV49, FeedbackContextV49, PhaseTransitionRecordV49,
        GeneratorStateV49, CharacterV49, SceneInfoV49, InputDataV49, # InputDataV49 を追加
        AppConfigV49, DialogueSettingsV49, # Part 2, 10 で定義
        DialogStyleManagerV49, PersistentCache, StructuredErrorV49, DialogueManagerV49,
        # Protocols that might be referenced by other protocols (though less common)
        # ConfigProtocol, SettingsProtocol, # Defined below in this part
    )
    from typing import TypeAlias

    # 型エイリアス (主にこのPart内で使用するものの明確化)
    # これらのエイリアスは、参照する型がこの時点で解決可能であるか、
    # 文字列リテラルとして正しく前方参照されている必要があります。
    ExternalConfigsV49Type: TypeAlias = ExternalConfigsV49
    GeneratorStateV49Type: TypeAlias = GeneratorStateV49
    FeedbackContextV49Type: TypeAlias = FeedbackContextV49
    PhaseTransitionRecordV49Type: TypeAlias = PhaseTransitionRecordV49
    StructuredErrorV49Type: TypeAlias = StructuredErrorV49
    CharacterInputType: TypeAlias = Dict[str, Any] # Part 12 から再掲
    SceneInfoInputType: TypeAlias = Dict[str, Any]   # Part 12 から再掲

    # Enumの型エイリアス (Part 1b で定義済みのものを直接参照するため、ここでは不要な場合もある)
    # スクリプト全体で一貫して Enum名を直接使う方が可読性が高い場合もある。
    # PsychologicalPhaseV49EnumType: TypeAlias = PsychologicalPhaseV49
    # EmotionalToneV49EnumType: TypeAlias = EmotionalToneV49
    # DFRSMetricsV49EnumType: TypeAlias = DFRSMetricsV49
    # SubjectivityCategoryV49EnumType: TypeAlias = SubjectivityCategoryV49
    # FluctuationCategoryV49EnumType: TypeAlias = FluctuationCategoryV49

    # ScoreKeys.LLM はネストされたEnumなので、アクセス方法に注意が必要
    if hasattr(ScoreKeys, 'LLM'):
        ScoreKeysLLMEnumType: TypeAlias = ScoreKeys.LLM # type: ignore
    else: # フォールバックとして一般的なEnum型
        ScoreKeysLLMEnumType: TypeAlias = enum.Enum # type: ignore


# -----------------------------------------------------------------------------
# -- Part 4a: Interfaces (protocols_v49.py 相当) (v4.9α - 最適化・修正版)
# -----------------------------------------------------------------------------

# --- 基本的な型定義 ---
T = TypeVar('T') # Part 0で定義済み想定だが、ここで再確認

# --- 設定関連プロトコル ---
class ConfigProtocol(Protocol):
    """
    設定クラス(AppConfigV49)が実装すべき属性・メソッドのインターフェース (v4.9 最適化版)。
    """
    SYSTEM_VERSION: str
    DEFAULT_MODEL: str
    API_KEY_ENV_VAR: str
    DEFAULT_JOB_ID_PREFIX: str
    MAX_CONSECUTIVE_ERRORS_DEFAULT: int
    CONFIG_FILENAME: str
    RESOURCES_DIR_NAME: str
    SUBJECTIVITY_KEYWORDS_FILENAME: str
    FLUCTUATION_PATTERNS_FILENAME: str
    ANALYZER_KEYWORDS_FILENAME: str
    # --- AppConfigV49で定義されている他の重要なクラス変数もここに追加 ---

    # インスタンス属性
    log_level_str: str
    log_filename: pathlib.Path # Pathオブジェクトであることを明示
    log_max_bytes: int
    log_backup_count: int
    base_output_dir: pathlib.Path
    config_dir: pathlib.Path
    resources_dir: pathlib.Path
    cache_dir: pathlib.Path
    persistent_cache_dir: pathlib.Path
    filename_max_length: int
    external_config_enabled: bool
    enable_filelock: bool
    
    # API関連 (APISettingsV49から移譲される属性の例)
    api_key: Optional[str]
    rpm_limit: int
    # max_retries: int # ErrorConfigV49に移行の可能性
    # base_retry_delay: float # ErrorConfigV49に移行の可能性
    # max_retry_delay: float # ErrorConfigV49に移行の可能性
    # rate_limit_delay: float # ErrorConfigV49に移行の可能性
    # api_timeout: int # ErrorConfigV49に移行の可能性

    # 実際の generation_config は APISettingsV49 経由になるため、
    # ConfigProtocol で直接持つかは設計による。ここでは例としてコメントアウト。
    # generation_config: Dict[str, Any]
    # safety_settings: Optional[Dict[str, str]]

    # ロードされた設定/リソースモデル
    loaded_external_configs: Optional['ExternalConfigsV49'] # TYPE_CHECKINGブロックで定義されたエイリアスを使用
    subjectivity_data: Optional['SubjectivityKeywordsFileV49'] # type: ignore
    fluctuation_data: Optional['FluctuationPatternsFileV49'] # type: ignore
    analyzer_keywords_data: Optional[Dict[str, Any]]

    # メソッド (AppConfigV49が持つべき主要メソッド)
    def load_external_configs(self) -> bool: ...
    def update_from_args(self, args: argparse.Namespace) -> None: ...
    def initialize_base_directories(self, base_output_dir_override: Optional[pathlib.Path] = None) -> None: ...
    # _ensure_dir_exists は内部メソッドなのでプロトコルには含めないことが多い


class SettingsProtocol(Protocol):
    """
    ジョブ固有設定クラス(DialogueSettingsV49)のインターフェース (v4.9 最適化版)。
    """
    # 主要な設定属性 (型ヒントは実際のEnumクラスを直接参照)
    dialogue_mode: Literal["normal", "delayed", "mixed", "auto"]
    style_template: str
    custom_style_file_path: Optional[str]
    subjective_focus: bool
    subjective_intensity: 'SubjectiveIntensityLevel' # type: ignore[name-defined]
    dfrs_evaluation_enabled: bool
    dfrs_evaluate_all_loops: bool
    dfrs_for_initial_selection: bool
    advanced_nlp_enabled: bool
    nlp_model_name: str
    ml_emotion_enabled: bool
    ml_emotion_model: Optional[str]
    use_lightweight_ml_model: bool
    phase_tagging_enabled: bool
    adaptation_strategy_enabled: bool
    adaptation_strategy_type: Literal["simple_threshold", "probabilistic_history", "advanced_rl"]
    log_phase_tone_transitions: bool
    feedback_strategy_type: Literal["composite", "phase_tone_only", "subjectivity_only", "quality_only", "context_aware", "fluctuation_only"]
    json_export_enabled: bool
    json_schema_validation: bool
    save_prompts: bool
    save_evaluations: bool
    save_rejected_candidates: bool
    persistent_cache_enabled: bool # FeatureFlagConfigV49から移譲
    phase_tone_prompt_modulation_enabled: bool # FeatureFlagConfigV49から移譲
    
    feedback_loops: int
    min_feedback_loops: int
    min_score_threshold: float
    target_length: int
    auto_normalize_weights: bool

    @property
    def config(self) -> ConfigProtocol: ...
    @property
    def enhanced_dfrs_weights(self) -> Dict['DFRSMetricsV49', float]: ... # type: ignore[name-defined]
    @property
    def final_selection_weights(self) -> Dict['FinalSelectionKeysV49', float]: ... # type: ignore[name-defined]
    @property
    def initial_candidate_weights(self) -> Dict['InitialSelectionKeysV49', float]: ... # type: ignore[name-defined]
    
    def update_settings_based_on_mode(self, mode: str, auto_suggested: bool = True) -> None: ...
    def update_from_args(self, args: argparse.Namespace) -> None: ...
    # model_dump のようなメソッドもプロトコルに含めるか検討

# --- コア機能プロトコル ---
class ApiClientProtocol(Protocol):
    model_name: str
    api_available: bool
    def generate_content(self, prompt: str, *, generation_config_overrides: Optional[Dict[str, Any]] = None) -> str: ...
    def generate_content_with_candidates(self, prompt: str, *, candidate_count: int, generation_config_overrides: Optional[Dict[str, Any]] = None) -> List[str]: ...

class ScorerProtocol(Protocol):
    """
    主観性および揺らぎスコア計算のためのインターフェース。
    このプロトコルを実装するクラス (例: SubjectivityFluctuationScorerV49) は、
    Part 0 および Part 1b で定義された Enum 型を正しく参照する必要があります。
    """
    # 型ヒントには、グローバルスコープで利用可能な実際のEnum型名を使用
    def calculate_subjectivity_score(self, text: str) -> Tuple[float, Dict['SubjectivityCategoryV49', int]]: ... # type: ignore[name-defined]
    def calculate_fluctuation_intensity(self, text: str) -> Tuple[float, Dict['FluctuationCategoryV49', int]]: ... # type: ignore[name-defined]

class AnalyzerProtocol(Protocol):
    def analyze_and_get_results(self, text: str) -> Dict[str, Any]: ...

class EvaluatorProtocol(Protocol):
    def get_dfrs_scores_v49(
        self,
        dialogue_text: Optional[str] = None,
        analyzer_results: Optional[Dict[str, Any]] = None,
        intended_phase: Optional['PsychologicalPhaseV49'] = None, # type: ignore[name-defined]
        intended_tone: Optional['EmotionalToneV49'] = None # type: ignore[name-defined]
    ) -> Dict[str, Optional[float]]: ...

class PromptBuilderProtocol(Protocol):
    def create_dialogue_prompt(
        self,
        character_a: 'CharacterInputType', # type: ignore[name-defined]
        character_b: 'CharacterInputType', # type: ignore[name-defined]
        scene_info: 'SceneInfoInputType', # type: ignore[name-defined]
        target_length: int,
        settings: SettingsProtocol, # SettingsProtocol を使用
        phase_val: Optional['PsychologicalPhaseV49'] = None, # type: ignore[name-defined]
        tone_val: Optional['EmotionalToneV49'] = None # type: ignore[name-defined]
    ) -> str: ...
    def create_improvement_prompt(
        self,
        prev_dialogue: str,
        prev_evaluation_text: Union[str, Dict[str, Any]],
        feedback_context: 'FeedbackContextV49Type', # type: ignore[name-defined]
        settings: SettingsProtocol # SettingsProtocol を使用
    ) -> str: ...
    def create_evaluation_prompt(self, dialogue_text: str) -> str: ...
    
    def extract_scores(self, evaluation_text: str, llm_score_keys_enum: Type['ScoreKeysLLMEnumType']) -> Dict['ScoreKeysLLMEnumType', float]: ... # type: ignore[name-defined]

# --- 戦略関連プロトコル ---
class AdaptationStrategyProtocol(Protocol):
    enabled: bool
    def suggest_next_state(
        self,
        current_intended_phase: Optional['PsychologicalPhaseV49'], # type: ignore[name-defined]
        current_intended_tone: Optional['EmotionalToneV49'], # type: ignore[name-defined]
        last_inferred_phase: Optional['PsychologicalPhaseV49'], # type: ignore[name-defined]
        last_inferred_tone: Optional['EmotionalToneV49'], # type: ignore[name-defined]
        last_alignment_scores: Tuple[Optional[float], Optional[float]],
        current_generator_state: Optional['GeneratorStateV49Type'] = None # type: ignore[name-defined]
    ) -> Tuple[Optional['PsychologicalPhaseV49'], Optional['EmotionalToneV49']]: ... # type: ignore[name-defined]
    def record_transition(self, record: 'PhaseTransitionRecordV49Type') -> None: ... # type: ignore[name-defined]
    def save_history(self) -> None: ...
    def load_history(self) -> None: ...

class FeedbackStrategyProtocol(Protocol):
    def generate(self, context: 'FeedbackContextV49Type') -> str: ... # type: ignore[name-defined]

class FeedbackManagerProtocol(Protocol):
    composite_strategy: Optional[FeedbackStrategyProtocol]
    def get_feedback(self, context: 'FeedbackContextV49Type', strategy_key: Optional[str] = None) -> str: ... # type: ignore[name-defined]

# --- スタイル管理プロトコル ---
class StyleManagerProtocol(Protocol):
    def register_custom_template(self, name: str, description: str, prompt_additions: Union[List[str], str]) -> None: ...
    def load_from_file(self, filepath: Union[str, pathlib.Path]) -> None: ...
    def get_template(self, style_name: str) -> Dict[str, Any]: ...
    def list_available_styles(self) -> List[Dict[str, str]]: ...
    def get_style_prompt_addition_text(self, style_name: str) -> str: ...
    @classmethod
    def suggest_style_for_scene(cls, scene_info: 'SceneInfoInputType', analysis_results: Optional[Dict[str, Any]]=None, config: Optional[ConfigProtocol]=None) -> str: ... # type: ignore[name-defined]

# --- I/O関連プロトコル ---
class DialogueManagerProtocol(Protocol):
    job_id: str
    job_output_dir: pathlib.Path
    base_output_dir: pathlib.Path
    def initialize_directories(self, settings: SettingsProtocol) -> None: ...
    def save_dialogue(self, version: int, text: str) -> None: ...
    def save_prompt(self, name_stem: str, text: str) -> None: ...
    def save_evaluation(self, version: int, text: str) -> None: ...
    def save_rejected_candidate(self, candidate_data: Dict[str, Any], reason: str) -> None: ...
    def save_resume_state(self, state: 'GeneratorStateV49Type') -> bool: ... # type: ignore[name-defined]
    def load_resume_state(self) -> Optional['GeneratorStateV49Type']: ... # type: ignore[name-defined]
    def append_stats(self, stats_data: Dict[str, Any]) -> None: ...
    def save_final_results(self, final_state: 'GeneratorStateV49Type', report_type: str) -> None: ... # type: ignore[name-defined]
    def get_final_json_path(self) -> Optional[pathlib.Path]: ...

# --- 例外管理プロトコル ---
class ExceptionManagerProtocol(Protocol):
    ERROR_RECOVERY_STRATEGIES: Dict[str, Dict[str, Any]]
    ERROR_SEVERITY: Dict[str, str]
    def log_error(self, error: Union[Exception, 'StructuredErrorV49Type'], source_override: Optional[str] = None, include_trace: Optional[bool] = None, context_data_override: Optional[Dict[str, Any]] = None, code: Optional[str] = None) -> 'StructuredErrorV49Type': ... # type: ignore[name-defined]
    def handle_with_retry(self, operation_name: str, operation: Callable[..., T], args: Optional[Tuple[Any, ...]] = None, kwargs: Optional[Dict[str, Any]] = None, extract_context_func: Optional[Callable[[Optional[Tuple[Any, ...]], Optional[Dict[str, Any]]], Dict[str, Any]]] = None) -> Tuple[bool, Optional[T], Optional['StructuredErrorV49Type']]: ... # type: ignore[name-defined]
    def safe_file_operation(self, operation_name: str, file_operation: Callable[..., T], args: Optional[Tuple[Any,...]] = None, kwargs: Optional[Dict[str,Any]] = None) -> Tuple[bool, Optional[T], Optional['StructuredErrorV49Type']]: ... # type: ignore[name-defined]
    def safe_api_call(self, operation_name: str, api_function: Callable[..., T], args: Optional[Tuple[Any,...]] = None, kwargs: Optional[Dict[str,Any]] = None) -> Tuple[bool, Optional[T], Optional['StructuredErrorV49Type']]: ... # type: ignore[name-defined]
    def safe_nlp_processing(self, operation_name: str, nlp_function: Callable[..., T], args: Optional[Tuple[Any,...]] = None, kwargs: Optional[Dict[str,Any]] = None) -> Tuple[bool, Optional[T], Optional['StructuredErrorV49Type']]: ... # type: ignore[name-defined]
    def is_retryable(self, error: Union[Exception, 'StructuredErrorV49Type']) -> bool: ... # type: ignore[name-defined]

# =============================================================================
# -- Part 4a 終了点
# =============================================================================
# =============================================================================
# -- Part 4b: API Client (api/client_v49.py 相当) (v4.9α - 最適化・堅牢化版)
# =============================================================================
class ApiClientV49: # Implicitly implements ApiClientProtocol
    """Google Generative AI API 通信を処理 (v4.9α - 最適化・堅牢化版)"""
    if TYPE_CHECKING:
        ConfigProtoType = ConfigProtocol
        ExceptionManagerProtoType = ExceptionManagerProtocol
        StructuredErrorV49Type = StructuredErrorV49
        GenerationConfigGoogleType: TypeAlias = genai.types.GenerationConfig
        GenerateContentResponseGoogleType: TypeAlias = genai.types.GenerateContentResponse
        GenerativeModelGoogleType: TypeAlias = genai.GenerativeModel
    else:
        ConfigProtoType = 'ConfigProtocol'
        ExceptionManagerProtoType = 'ExceptionManagerProtocol'
        StructuredErrorV49Type = 'StructuredErrorV49'
        GenerationConfigGoogleType = 'genai.types.GenerationConfig'
        GenerateContentResponseGoogleType = 'genai.types.GenerateContentResponse'
        GenerativeModelGoogleType = 'genai.GenerativeModel'

    _LLM_PROBLEMATIC_RESPONSE_PATTERNS: ClassVar[List[Pattern[str]]] = [
        re.compile(r"評価\S*ありがとうございま|評価対象\S*提供されていません", re.IGNORECASE),
        re.compile(r"具体的な対話文\S*提供されておりません|評価を行うためには", re.IGNORECASE),
        re.compile(r"採点できません|テキストをご提供いただければ", re.IGNORECASE),
        re.compile(r"Dummy candidate", re.IGNORECASE),
        re.compile(r"恐れ入りますが|申し訳ありませんが", re.IGNORECASE),
        re.compile(r"指示された評価を実行することができません|内容を理解できませんでした", re.IGNORECASE),
        re.compile(r"入力\S*テキストが必要です|提供された情報のみでは判断できません", re.IGNORECASE),
        re.compile(r"このプロンプトでは応答できません", re.IGNORECASE),
        re.compile(r"could not process the request|unable to generate a response", re.IGNORECASE),
        re.compile(r"コンテンツポリシーに違反する可能性", re.IGNORECASE),
        # ▼▼▼ ログで確認された新しいパターンを追加 ▼▼▼
        re.compile(r"\*\*/評価対象の対話文が提供されていないため、採点できません。\*\*", re.IGNORECASE)
        # ▲▲▲ 追加完了 ▲▲▲
    ]

    def __init__(self, config: ConfigProtoType, exception_manager: ExceptionManagerProtoType): # type: ignore
        self.logger = logging.getLogger(f"{self.__module__}.{self.__class__.__qualname__}")
        self.config = config
        self.exception_manager = exception_manager

        self.model_name: str = self.config.DEFAULT_MODEL # type: ignore
        self.api_key: Optional[str] = self.config.api_key # type: ignore
        self.rpm_limit: int = self.config.RPM_LIMIT # type: ignore
        self.max_retries: int = self.config.MAX_RETRIES # type: ignore
        self.base_retry_delay: float = self.config.BASE_RETRY_DELAY # type: ignore
        self.max_retry_delay: float = self.config.MAX_RETRY_DELAY # type: ignore
        self.rate_limit_delay: float = self.config.RATE_LIMIT_DELAY # type: ignore
        self.api_timeout: int = self.config.API_TIMEOUT # type: ignore

        self.generation_config_base: Dict[str, Any] = self.config.generation_config.copy() # type: ignore
        # safety_settings は genai.GenerativeModel に渡す際、List[Dict] または Dict[HarmCategory, HarmBlockThreshold] 形式を期待される。
        # AppConfigV49 からは Dict[str, str] で渡されるため、必要に応じて変換処理を挟むか、
        # ライブラリがこの形式を解釈できることを前提とする。
        self.safety_settings_config: Optional[Union[Dict[str, str], List[Dict[str, Any]]]] = None
        if self.config.safety_settings: # type: ignore
            self.safety_settings_config = self.config.safety_settings.copy() # type: ignore

        self.last_request_time: Optional[float] = None
        self.api_available: bool = False
        self.model: Optional[ApiClientV49.GenerativeModelGoogleType] = None

        if not GOOGLE_API_AVAILABLE:
            self.logger.warning("google-generativeai ライブラリが利用できないため、API関連機能は無効です。")
            return
        if not self.api_key:
            self.logger.warning(f"APIキーが設定されていません (環境変数 '{self.config.API_KEY_ENV_VAR}' を確認してください)。API関連機能は無効です。") # type: ignore
            return

        try:
            genai.configure(api_key=self.api_key)
            # safety_settingsの形式について genai ライブラリのドキュメントを参照し、適切な形式で渡すことが重要。
            # ここでは self.safety_settings_config がそのまま渡せる形式であると仮定。
            self.model = genai.GenerativeModel(self.model_name, safety_settings=self.safety_settings_config) # type: ignore
            self.api_available = True
            self.logger.info(f"ApiClientV49が正常に初期化されました (モデル: '{self.model_name}')。")
        except Exception as e_init_client:
            err_code_client_init = "API.INIT_FAILED"
            if GOOGLE_API_AVAILABLE:
                if isinstance(e_init_client, google_exceptions.PermissionDenied): err_code_client_init = "API.PERMISSION_DENIED" # type: ignore
                elif isinstance(e_init_client, google_exceptions.InvalidArgument): err_code_client_init = "API.INVALID_ARGUMENT_INIT" # type: ignore
            
            structured_error_client_init: ApiClientV49.StructuredErrorV49Type = self.exception_manager.log_error( # type: ignore
                e_init_client,
                source="ApiClientV49.__init__", code=err_code_client_init, context_data={"model_name": self.model_name}
            )
            raise RuntimeError(f"APIクライアントの初期化に失敗しました: {structured_error_client_init}") from e_init_client

    def _apply_adaptive_delay(self) -> None:
        if not self.api_available or self.rpm_limit <= 0: return
        min_interval_seconds = 60.0 / self.rpm_limit
        current_time_monotonic = time.monotonic()
        if self.last_request_time is not None:
            elapsed_seconds_since_last = current_time_monotonic - self.last_request_time
            wait_duration_adaptive = min_interval_seconds - elapsed_seconds_since_last
            if wait_duration_adaptive > 0:
                self.logger.debug(f"RPM制限のため {wait_duration_adaptive:.3f} 秒待機します。")
                time.sleep(wait_duration_adaptive)
        self.last_request_time = time.monotonic()

    def _handle_api_error_internal(self, exc_api_internal: Exception, attempt_api_internal: int) -> Tuple[bool, float, int]:
        structured_error_obj_internal: ApiClientV49.StructuredErrorV49Type = self.exception_manager.log_error(exc_api_internal, include_trace=(attempt_api_internal == 0)) # type: ignore
        should_retry_internal = self.exception_manager.is_retryable(structured_error_obj_internal) # type: ignore
        recovery_strategy_internal = self.exception_manager.ERROR_RECOVERY_STRATEGIES.get(structured_error_obj_internal.code, {}) # type: ignore
        max_retries_for_this_error_internal = int(recovery_strategy_internal.get("max_retries", self.max_retries))
        base_delay_for_this_error_internal = float(recovery_strategy_internal.get("retry_delay", self.base_retry_delay))
        delay_seconds_val_internal = 0.0
        if should_retry_internal:
            if structured_error_obj_internal.code == "API.RATE_LIMIT":
                delay_seconds_val_internal = max(self.rate_limit_delay, exponential_backoff(attempt_api_internal, base_delay_for_this_error_internal, self.max_retry_delay, jitter_fraction=0.3))
            else:
                delay_seconds_val_internal = exponential_backoff(attempt_api_internal, base_delay_for_this_error_internal, self.max_retry_delay, jitter_fraction=0.3)
        return should_retry_internal, max(0.0, delay_seconds_val_internal), max_retries_for_this_error_internal

    def _execute_api_call_with_retry(
        self, prompt_content: str, generation_config_obj: 'ApiClientV49.GenerationConfigGoogleType'
    ) -> 'ApiClientV49.GenerateContentResponseGoogleType':
        if not self.api_available or self.model is None:
            self.logger.error("APIクライアントが初期化されていないか利用不可のため、API呼び出しを実行できません。"); raise RuntimeError("APIクライアント未初期化/利用不可。")
        
        current_attempt_count = 0
        last_exception_for_retry: Optional[Exception] = None
        max_retries_for_current_call = self.max_retries
        api_request_options = {'timeout': self.api_timeout}

        while True:
            if current_attempt_count > max_retries_for_current_call:
                error_message_max_retry = f"API呼び出しが最大リトライ回数 ({max_retries_for_current_call}) を超過しました。最終エラー: {type(last_exception_for_retry).__name__ if last_exception_for_retry else 'N/A'}"
                self.logger.error(error_message_max_retry)
                raise RuntimeError(error_message_max_retry) from last_exception_for_retry
            try:
                self._apply_adaptive_delay()
                log_params_for_api = {}
                for param_name_log in ['temperature', 'candidate_count', 'top_k', 'top_p', 'max_output_tokens']:
                    if hasattr(generation_config_obj, param_name_log):
                        log_params_for_api[param_name_log] = getattr(generation_config_obj, param_name_log)
                
                self.logger.info(
                    f"APIリクエスト実行 (試行 {current_attempt_count + 1}/{max_retries_for_current_call + 1}), "
                    f"モデル: '{self.model_name}', パラメータ(抜粋): {log_params_for_api}, タイムアウト: {self.api_timeout}s"
                )
                api_call_time_start = time.monotonic()
                api_response_from_genai: ApiClientV49.GenerateContentResponseGoogleType = self.model.generate_content( # type: ignore
                    contents=prompt_content,
                    generation_config=generation_config_obj,
                    request_options=api_request_options
                )
                api_call_time_duration_ms = (time.monotonic() - api_call_time_start) * 1000
                self.logger.info(f"APIレスポンス受信 ({api_call_time_duration_ms:.1f}ms)。レスポンス型: {type(api_response_from_genai)}")

                if api_response_from_genai is None:
                    self.logger.error("APIレスポンスがNoneです。RuntimeErrorを送出します。"); raise RuntimeError("API returned a None response object.")
                if not hasattr(api_response_from_genai, 'candidates'):
                    self.logger.error("CRITICAL: APIレスポンスに必須の'candidates'属性がありません。AttributeErrorを送出します。"); raise AttributeError("API Response is missing the 'candidates' attribute.")
                
                response_prompt_feedback = getattr(api_response_from_genai, 'prompt_feedback', None)
                if response_prompt_feedback and getattr(response_prompt_feedback, 'block_reason', None):
                    block_reason_str = getattr(response_prompt_feedback.block_reason, 'name', str(response_prompt_feedback.block_reason))
                    self.logger.error(f"プロンプトがAPIによってブロックされました: {block_reason_str}")
                    if GOOGLE_API_AVAILABLE:
                         raise google_exceptions.FailedPrecondition(f"Prompt blocked by API due to: {block_reason_str}") # type: ignore
                    else:
                         raise RuntimeError(f"Prompt blocked by API due to: {block_reason_str}")
                return api_response_from_genai
            
            except AttributeError as e_attr_in_retry: # レスポンスオブジェクトの構造に関するエラー
                self.logger.error(f"APIレスポンス処理中にAttributeError (試行 {current_attempt_count + 1}): {e_attr_in_retry}", exc_info=True)
                last_exception_for_retry = e_attr_in_retry
                self.logger.error(f"リトライ不能なAttributeErrorと判断 (試行 {current_attempt_count + 1})。RuntimeErrorを送出します。")
                raise RuntimeError(f"APIリクエスト処理中にAttributeErrorが発生: {e_attr_in_retry}") from last_exception_for_retry
            except Exception as e_general_in_retry: # その他のAPI関連エラー
                last_exception_for_retry = e_general_in_retry
                should_retry_call, delay_seconds_for_call, max_retries_new = self._handle_api_error_internal(e_general_in_retry, current_attempt_count)
                max_retries_for_current_call = max_retries_new
                
                if not should_retry_call:
                    self.logger.error(f"リトライ不能なエラー ({type(last_exception_for_retry).__name__}) がAPI呼び出し中に発生しました (試行 {current_attempt_count + 1})。RuntimeErrorを送出します。")
                    raise RuntimeError(f"APIリクエスト中にリトライ不能なエラーが発生: {e_general_in_retry}") from last_exception_for_retry
                
                self.logger.warning(
                    f"API呼び出し失敗 ({type(last_exception_for_retry).__name__})。{delay_seconds_for_call:.3f}秒後にリトライします "
                    f"(試行 {current_attempt_count + 1}/{max_retries_for_current_call + 1})。"
                )
                time.sleep(delay_seconds_for_call)
            current_attempt_count += 1
            
        self.logger.critical(f"APIリトライループが予期せず終了 (論理エラーの可能性)。最終エラー: {type(last_exception_for_retry).__name__ if last_exception_for_retry else 'N/A'}")
        raise RuntimeError("APIリトライループが予期せず終了しました。") from last_exception_for_retry

    def _validate_and_prepare_gen_config(self, overrides: Optional[Dict[str, Any]] = None, candidate_count: int = 1) -> 'ApiClientV49.GenerationConfigGoogleType':
        config_dict_to_validate = self.generation_config_base.copy()
        if overrides: config_dict_to_validate.update({k: v for k, v in overrides.items() if k != 'candidate_count'})
        
        # Gemini API は candidate_count=1 のみをサポートすることが多い。複数指定しても最初の1つだけが返るかエラーになる。
        # そのため、要求された candidate_count によらず、API に渡す値は1に固定することを検討。
        # ここでは、バリデーションとしては受け付けるが、APIの仕様をコメントで注意喚起。
        config_dict_to_validate["candidate_count"] = max(1, min(candidate_count, 1)) # Geminiは通常1

        def _clip_param_for_config(params: Dict[str, Any], key_param: str, min_val_param: Optional[Union[int, float]], max_val_param: Optional[Union[int, float]], type_param: Type[Union[int, float]]):
            if key_param not in params or params[key_param] is None: return
            original_param_value = params[key_param]
            try:
                typed_param_value = type_param(original_param_value)
            except (ValueError, TypeError):
                self.logger.warning(f"生成設定'{key_param}'の値'{original_param_value}'は型({type_param.__name__})に変換できません。無視します。"); params.pop(key_param,None); return
            
            clipped_param_value = typed_param_value
            if min_val_param is not None and typed_param_value < min_val_param: clipped_param_value = min_val_param # type: ignore
            if max_val_param is not None and typed_param_value > max_val_param: clipped_param_value = max_val_param # type: ignore
            
            if clipped_param_value != typed_param_value:
                self.logger.warning(f"生成設定'{key_param}'の値'{original_param_value}'(変換後:{typed_param_value})は範囲外({min_val_param}-{max_val_param})です。'{clipped_param_value}'に調整しました。")
                params[key_param]=clipped_param_value
            else:
                params[key_param]=typed_param_value

        _clip_param_for_config(config_dict_to_validate, "temperature", 0.0, 2.0, float)
        _clip_param_for_config(config_dict_to_validate, "top_p", 0.0, 1.0, float)
        _clip_param_for_config(config_dict_to_validate, "top_k", 1, None, int) # 上限はモデル依存
        _clip_param_for_config(config_dict_to_validate, "candidate_count", 1, 1, int) # Gemini APIでは通常1
        # max_output_tokens の上限はモデルによって異なるため、configから取得できると良い
        model_max_tokens = getattr(self.config, 'MODEL_MAX_OUTPUT_TOKENS', 8192) # type: ignore
        _clip_param_for_config(config_dict_to_validate, "max_output_tokens", 1, model_max_tokens, int)

        stop_sequences_cfg = config_dict_to_validate.get("stop_sequences")
        if stop_sequences_cfg is not None:
            if not (isinstance(stop_sequences_cfg, list) and all(isinstance(s, str) for s in stop_sequences_cfg)):
                self.logger.warning(f"生成設定'stop_sequences'の値'{stop_sequences_cfg}'が不正(文字列リスト期待)。無視します。"); config_dict_to_validate.pop("stop_sequences",None)
        
        self.logger.debug(f"検証・準備済み生成設定: {config_dict_to_validate}")
        try:
            # genai.types.GenerationConfig で受け付けられるパラメータのみを抽出して渡す
            known_params_for_genai_config = {"temperature", "top_p", "top_k", "candidate_count", "max_output_tokens", "stop_sequences"}
            final_config_dict_for_genai_obj = {k: v for k, v in config_dict_to_validate.items() if k in known_params_for_genai_config and v is not None}
            return genai.types.GenerationConfig(**final_config_dict_for_genai_obj) # type: ignore
        except TypeError as e_gen_cfg_type:
            self.logger.error(f"GenerationConfigの作成に失敗しました ({final_config_dict_to_genai_obj}): {e_gen_cfg_type}", exc_info=True)
            raise ValueError(f"無効な生成設定パラメータが渡されました: {e_gen_cfg_type}") from e_gen_cfg_type

    def generate_content(self, prompt: str, *, generation_config_overrides: Optional[Dict[str, Any]] = None) -> str:
        if not self.api_available: self.logger.warning("API利用不可。コンテンツ生成スキップ。"); return ""
        try:
            prepared_config = self._validate_and_prepare_gen_config(generation_config_overrides, candidate_count=1)
        except ValueError as e_val_prep_cfg:
            self.logger.error(f"単一コンテンツ生成のための設定準備に失敗: {e_val_prep_cfg}"); return ""
        
        try:
            api_response = self._execute_api_call_with_retry(prompt, prepared_config)
            return self._process_single_response(api_response)
        except Exception as e_gen_single:
            self.logger.error(f"単一コンテンツ生成処理の最終段階でエラー: {type(e_gen_single).__name__} - {e_gen_single}", exc_info=True); return ""

    def generate_content_with_candidates(
        self, prompt: str, *, candidate_count: int, generation_config_overrides: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        if not self.api_available: self.logger.warning("API利用不可。複数候補生成スキップ。"); return []
        try:
            # Gemini APIのcandidate_countは通常1。リクエストは作成するがAPI側の挙動に注意。
            effective_cand_count = max(1, min(candidate_count, 1)) # 実質1
            prepared_config_multi = self._validate_and_prepare_gen_config(generation_config_overrides, candidate_count=effective_cand_count)
        except ValueError as e_val_prep_multi_cfg:
            self.logger.error(f"複数候補生成のための設定準備に失敗: {e_val_prep_multi_cfg}"); return []
        
        try:
            api_response_multi_cand = self._execute_api_call_with_retry(prompt, prepared_config_multi)
            return self._process_multiple_candidates(api_response_multi_cand)
        except Exception as e_gen_multi:
            self.logger.error(f"複数候補生成処理の最終段階でエラー: {type(e_gen_multi).__name__} - {e_gen_multi}", exc_info=True); return []

    def _filter_llm_problematic_response(self, text_to_filter: Optional[str], log_source_prefix: str) -> str:
        """LLMからの定型的なエラー応答や評価不能通知をフィルタリングする。"""
        if not text_to_filter or not text_to_filter.strip():
            return "" # 入力が空なら空を返す

        for pattern_regex in self._LLM_PROBLEMATIC_RESPONSE_PATTERNS:
            if pattern_regex.search(text_to_filter):
                self.logger.warning(
                    f"{log_source_prefix}: テキストが問題応答パターン「{pattern_regex.pattern}」に一致。フィルタリングします。 "
                    f"Preview: '{text_to_filter[:100].replace(chr(10),'/')}...'"
                )
                return "" # パターンに一致したら空文字列を返す
        return text_to_filter # 問題なければ元のテキストを返す

    def _process_single_response(self, response_obj: 'ApiClientV49.GenerateContentResponseGoogleType') -> str:
        self.logger.debug(f"_process_single_response: 受信レスポンス型: {type(response_obj)}")
        if not GOOGLE_API_AVAILABLE: self.logger.warning("_process_single_response: GOOGLE_API_AVAILABLE is False."); return ""
        if response_obj is None: self.logger.warning("_process_single_response: Response object is None."); return ""

        extracted_text_content: str = ""
        
        # 1. response.candidates からの抽出とフィルタリング
        if hasattr(response_obj, 'candidates') and response_obj.candidates is not None:
            candidates_from_response = response_obj.candidates
            candidate_object_to_process: Optional[Any] = None
            
            if hasattr(candidates_from_response, '__iter__') and not isinstance(candidates_from_response, (str, bytes)):
                try:
                    candidate_list_from_response = list(candidates_from_response)
                    if candidate_list_from_response: candidate_object_to_process = candidate_list_from_response[0]
                    else: self.logger.warning("_process_single_response: response.candidates は反復可能でしたが空でした。")
                except Exception as e_iter_cand_proc: self.logger.error(f"_process_single_response: response.candidates のリスト変換/アクセス中にエラー: {e_iter_cand_proc}", exc_info=True)
            elif candidates_from_response is not None: candidate_object_to_process = candidates_from_response
            
            if candidate_object_to_process is not None:
                # _extract_text_from_single_candidate_object は内部でフィルタリングも行う
                extracted_text_content = self._extract_text_from_single_candidate_object(
                    candidate_object_to_process, "_process_single_response (from candidate obj)"
                )
                if extracted_text_content:
                    self.logger.info(f"_process_single_response: 候補オブジェクトから有効なテキストを抽出・フィルタリング完了。")
                    return extracted_text_content
                else:
                    self.logger.warning("_process_single_response: 候補オブジェクトから有効なテキストを抽出できませんでした（フィルタリング後）。response.textフォールバックを試みます。")
        else:
            self.logger.warning("_process_single_response: Responseに'candidates'属性がないかNoneです。response.textフォールバックを試みます。")

        # 2. response.text をフォールバックとして使用し、フィルタリング
        if hasattr(response_obj, 'text') and isinstance(response_obj.text, str) and response_obj.text.strip():
            text_from_response_attr = response_obj.text.strip()
            self.logger.info(f"_process_single_response: response.text にフォールバック。Preview: '{text_from_response_attr[:150].replace(chr(10),'/')}...'")
            extracted_text_content = self._filter_llm_problematic_response(text_from_response_attr, "_process_single_response (response.text filter)")
            if extracted_text_content:
                self.logger.info(f"_process_single_response (response.text fallback): テキストを抽出・フィルタリング完了。")
                return extracted_text_content
            else:
                self.logger.warning("_process_single_response (response.text fallback): テキストはフィルタリングにより空と判断されました。")
        
        self.logger.warning("_process_single_response: 最終的にどのパスからも有効なテキストを抽出できませんでした。")
        return ""

    def _extract_text_from_single_candidate_object(self, candidate_obj_param: Any, log_source_prefix: str) -> str:
        if candidate_obj_param is None: self.logger.warning(f"{log_source_prefix}: 候補オブジェクトがNoneです。"); return ""
        self.logger.debug(f"{log_source_prefix}: 候補オブジェクト処理中。型: {type(candidate_obj_param)}")

        # 1. 安全性チェック (finish_reason, safety_ratings)
        # (この部分は前回の改善提案のものをベースに、Enumクラスの取得をより安全にする)
        finish_reason_from_cand = getattr(candidate_obj_param, 'finish_reason', None)
        finish_reason_name_for_log = "UNKNOWN_REASON"
        FinishReasonEnumClass: Optional[Type[enum.Enum]] = None
        if GOOGLE_API_AVAILABLE and hasattr(genai.types, 'Candidate') and hasattr(genai.types.Candidate, 'FinishReason'): # type: ignore
            FinishReasonEnumClass = genai.types.Candidate.FinishReason # type: ignore
        
        if FinishReasonEnumClass:
            if isinstance(finish_reason_from_cand, FinishReasonEnumClass): finish_reason_name_for_log = finish_reason_from_cand.name
            elif isinstance(finish_reason_from_cand, int):
                try: finish_reason_name_for_log = FinishReasonEnumClass(finish_reason_from_cand).name # type: ignore
                except ValueError: finish_reason_name_for_log = f"INT_VAL_{finish_reason_from_cand}"
            elif finish_reason_from_cand is not None: finish_reason_name_for_log = str(finish_reason_from_cand).upper()
        else:
            if isinstance(finish_reason_from_cand, enum.Enum): finish_reason_name_for_log = finish_reason_from_cand.name
            elif finish_reason_from_cand is not None: finish_reason_name_for_log = str(finish_reason_from_cand).upper()
        self.logger.debug(f"{log_source_prefix}: Candidate finish_reason='{finish_reason_name_for_log}' (raw: {finish_reason_from_cand})")

        problematic_finish_reasons = {"SAFETY", "RECITATION"}
        if FinishReasonEnumClass:
            problematic_finish_reasons.add(FinishReasonEnumClass.SAFETY.name) # type: ignore
            problematic_finish_reasons.add(FinishReasonEnumClass.RECITATION.name) # type: ignore
        if finish_reason_name_for_log in problematic_finish_reasons:
            self.logger.warning(f"{log_source_prefix}: 候補は不適切なfinish_reason ({finish_reason_name_for_log}) のためスキップ。"); return ""

        safety_ratings_from_cand = getattr(candidate_obj_param, 'safety_ratings', [])
        if hasattr(safety_ratings_from_cand, '__iter__') and not isinstance(safety_ratings_from_cand, str):
            HarmProbabilityEnumClass: Optional[Type[enum.Enum]] = None
            if GOOGLE_API_AVAILABLE and hasattr(genai.types, 'HarmProbability'): # type: ignore
                HarmProbabilityEnumClass = genai.types.HarmProbability # type: ignore
            blocking_harm_probabilities = {"HIGH", "MEDIUM"}
            if HarmProbabilityEnumClass:
                 blocking_harm_probabilities.add(HarmProbabilityEnumClass.HIGH.name) # type: ignore
                 blocking_harm_probabilities.add(HarmProbabilityEnumClass.MEDIUM.name) # type: ignore
            for rating_obj in safety_ratings_from_cand:
                probability_name_rating = getattr(getattr(rating_obj, 'probability', None), 'name', None)
                if probability_name_rating in blocking_harm_probabilities:
                    category_name_rating = getattr(getattr(rating_obj, 'category', None), 'name', "UnknownCategory")
                    self.logger.warning(f"{log_source_prefix}: 安全性評価によりブロック ({category_name_rating} is {probability_name_rating})。"); return ""
        
        # 2. テキスト抽出
        extracted_text_candidate = ""
        candidate_content = getattr(candidate_obj_param, 'content', None)
        if candidate_content and hasattr(candidate_content, 'parts') and candidate_content.parts:
            content_parts_list: List[Any] = []
            if isinstance(candidate_content.parts, list): content_parts_list = candidate_content.parts
            elif hasattr(candidate_content.parts, '__iter__') and not isinstance(candidate_content.parts, (str,bytes)):
                try: content_parts_list = list(candidate_content.parts)
                except Exception as e_list_parts: self.logger.warning(f"{log_source_prefix}: content.partsのリスト変換エラー: {e_list_parts}")
            
            if content_parts_list:
                text_segments_from_parts: List[str] = []
                for part_obj in content_parts_list:
                    if hasattr(part_obj, 'text') and isinstance(part_obj.text, str): text_segments_from_parts.append(part_obj.text)
                    elif isinstance(part_obj, str): text_segments_from_parts.append(part_obj) # part自体が文字列の場合
                extracted_text_candidate = "".join(text_segments_from_parts).strip()
                if extracted_text_candidate: self.logger.debug(f"{log_source_prefix}: content.partsからテキスト抽出: '{extracted_text_candidate[:50]}...'")
            elif hasattr(candidate_content.parts, 'text') and isinstance(candidate_content.parts.text, str): # partsが単一text属性を持つ場合
                extracted_text_candidate = candidate_content.parts.text.strip()
                if extracted_text_candidate: self.logger.debug(f"{log_source_prefix}: 単一content.parts.textから抽出: '{extracted_text_candidate[:50]}...'")
        
        if not extracted_text_candidate and hasattr(candidate_obj_param, 'text') and isinstance(candidate_obj_param.text, str): # candidate直下のtext属性
            extracted_text_candidate = candidate_obj_param.text.strip()
            if extracted_text_candidate: self.logger.debug(f"{log_source_prefix}: candidate.textから抽出 (フォールバック): '{extracted_text_candidate[:50]}...'")

        # 3. 抽出テキストのフィルタリング
        final_filtered_text = self._filter_llm_problematic_response(extracted_text_candidate, f"{log_source_prefix} (final filter)")
        
        if not final_filtered_text and extracted_text_candidate:
             self.logger.warning(f"{log_source_prefix}: テキストは抽出されましたが、フィルタリングにより空になりました。元のテキストプレビュー: '{extracted_text_candidate[:100]}'")
        elif not extracted_text_candidate:
             self.logger.warning(f"{log_source_prefix}: どのパスからもテキストを抽出できませんでした。")
        return final_filtered_text

    def _process_multiple_candidates(self, response_multi_obj: 'ApiClientV49.GenerateContentResponseGoogleType') -> List[str]:
        self.logger.debug(f"_process_multiple_candidates: 受信レスポンス型: {type(response_multi_obj)}")
        if not GOOGLE_API_AVAILABLE: self.logger.warning("_process_multiple_candidates: GOOGLE_API_AVAILABLE is False."); return []
        if response_multi_obj is None or not hasattr(response_multi_obj, 'candidates') or response_multi_obj.candidates is None:
            self.logger.warning("_process_multiple_candidates: Responseに'candidates'属性がないかNoneです。"); return []

        candidates_collection_from_multi_resp = response_multi_obj.candidates
        actual_candidates_list_to_process_multi: List[Any] = []
        if isinstance(candidates_collection_from_multi_resp, list):
            actual_candidates_list_to_process_multi = candidates_collection_from_multi_resp
        elif hasattr(candidates_collection_from_multi_resp, '__iter__') and not isinstance(candidates_collection_from_multi_resp, (str, bytes)):
            try: actual_candidates_list_to_process_multi = list(candidates_collection_from_multi_resp)
            except Exception as e_list_conv_multi_proc: self.logger.error(f"_process_multiple_candidates: candidates をリストに変換中エラー: {e_list_conv_multi_proc}"); return []
        elif candidates_collection_from_multi_resp is not None:
             actual_candidates_list_to_process_multi = [candidates_collection_from_multi_resp]
        
        if not actual_candidates_list_to_process_multi:
            self.logger.warning("_process_multiple_candidates: 有効な候補リストを取得できませんでした。")
            return []
        
        extracted_texts_from_multi: List[str] = []
        for idx_multi, cand_item_multi in enumerate(actual_candidates_list_to_process_multi):
            # _extract_text_from_single_candidate_object はフィルタリング済みのテキストを返す
            text_from_single_cand = self._extract_text_from_single_candidate_object(cand_item_multi, f"_process_multiple_candidates (Candidate {idx_multi})")
            if text_from_single_cand: # 空でなければ追加
                extracted_texts_from_multi.append(text_from_single_cand)
            else:
                self.logger.warning(f"  _process_multiple_candidates: 候補 {idx_multi} から有効なテキストを抽出できませんでした（フィルタリング後を含む）。")
        
        if not extracted_texts_from_multi:
            self.logger.warning("_process_multiple_candidates: どの候補からも有効なテキストを抽出できませんでした。")
        return extracted_texts_from_multi

# =============================================================================
# -- Part 4b 終了点 (ApiClientV49 クラス定義終了)
# =============================================================================
# =============================================================================
# -- Part 5: Prompt Builder (v4.9β – 最適化・堅牢性向上・プロンプト改善版 v5.5)
# =============================================================================
# v5.5 Update:
# - `extract_scores`メソッドをv5.2ベースで完全に復元し、さらにログ強化、マーカーベース検索、
#   Enumキーマッピング（.valueと.name両対応）、エラーハンドリングを改善。
#   これにより「version_state_data.llm_scores の型: <class 'NoneType'>」エラーの根本原因解決を目指す。
# - `create_dialogue_prompt`メソッドを全面的に再構築。
#   - ユーザー提示の【決定版プロンプト Part 5】（v5.4の自己評価で提示された詳細テキスト）の
#     階層構造（5.1～5.5）と内容を忠実にプロンプト文字列として生成するロジックに変更。
#   - 旧来のプロンプト部品（role_and_goal_instruction等）を新しい構造に統合・整理し、冗長性を排除。
#   - `_build_subjective_instruction_v49` と `_get_phase_tone_instruction_v49` の出力を
#     新しいプロンプト構造内の適切なセクションに、見出し等を調整して組み込む。
#   - `style_instruction_text` （DialogStyleManagerV49から取得）の挿入箇所を明確化。
# - ログ出力、バージョン情報をv5.5に更新。
# - NameError等の定義漏れがないよう、プロンプト部品の参照関係を慎重に構築。
# - その他メソッド (format_character_sheet, create_evaluation_prompt等) はv5.2/v5.4の安定版を維持。

from typing import TYPE_CHECKING, TypeVar, Set, List, Dict, Optional, Tuple, Union, Any, Type, ClassVar
import enum
import logging
import random
import re
import statistics # extract_scores で使用
import os # extract_scores で os.linesep を使用するため (ログ表示用)
# sys, time は __init__ 内でのクラスロードエラー時に使用される可能性があるため、念のため残す (直接は使われない)
import sys
import time


# --- グローバルスコープで利用可能であることを期待する変数 (Part 0 などで定義済み) ---
PYDANTIC_AVAILABLE = globals().get('PYDANTIC_AVAILABLE', False)
BaseModel: Type[Any] = globals().get('BaseModel', object) # type: ignore
ConfigDict: Type[Dict[str, Any]] = globals().get('ConfigDict', dict) # type: ignore
Field: Callable[..., Any] = globals().get('Field', lambda **kwargs: None) # type: ignore
ValidationError: Type[Exception] = globals().get('ValidationError', ValueError) # PydanticのValidationErrorまたはフォールバック
_get_global_type: Callable[[str, Optional[type]], Optional[Type[Any]]] = globals().get('_get_global_type', lambda name, meta=None: globals().get(name))
sanitize_filename: Callable[[str, Optional[int], str], str] = globals().get('sanitize_filename', lambda f, ml=None, r='_': str(f))
fmt: Callable[[Optional[Union[float, int]], int, str], str] = globals().get('fmt', lambda v, p=2, na="N/A": str(v))


TEnum = TypeVar('TEnum', bound=enum.Enum)

# ---------- type-checking-only imports ----------
if TYPE_CHECKING:
    from __main__ import ( # type: ignore[attr-defined]
        ConfigProtocol, SettingsProtocol, FeedbackStrategyProtocol, StyleManagerProtocol,
        SubjectiveIntensityLevel, PsychologicalPhaseV49, EmotionalToneV49,
        SubjectivityCategoryV49, FluctuationCategoryV49, ScoreKeys, DFRSMetricsV49,
        FeedbackContextV49, SubjectivityKeywordEntryV49, FluctuationPatternEntryV49
    )
    from typing import TypeAlias
    CharacterDictType: TypeAlias = Dict[str, Any]
    SceneInfoDictType: TypeAlias = Dict[str, Any]
    ScoreKeysLLMEnumType_hint: TypeAlias = ScoreKeys.LLM # type: ignore
    SubjectivityKeywordEntryV49Type_hint: TypeAlias = SubjectivityKeywordEntryV49
    FluctuationPatternEntryV49Type_hint: TypeAlias = FluctuationPatternEntryV49
else:
    ConfigProtocol = 'ConfigProtocol'
    SettingsProtocol = 'SettingsProtocol'
    FeedbackStrategyProtocol = 'FeedbackStrategyProtocol'
    StyleManagerProtocol = Optional['StyleManagerProtocol'] # type: ignore
    ScoreKeysLLMEnumType_hint = 'ScoreKeys.LLM' # Runtime fallback for type hint

class PromptBuilderV49: # Implicitly implements PromptBuilderProtocol
    logger: logging.Logger

    if TYPE_CHECKING:
        ConfigProtoType = ConfigProtocol
        SettingsProtoType = SettingsProtocol
        FeedbackStrategyProtoType = FeedbackStrategyProtocol
        StyleManagerProtoType = Optional[StyleManagerProtocol]
        SubjectiveIntensityLevel_cls_type: Type[SubjectiveIntensityLevel]
        PsychologicalPhaseV49_cls_type: Type[PsychologicalPhaseV49]
        EmotionalToneV49_cls_type: Type[EmotionalToneV49]
        SubjectivityCategoryV49_cls_type: Type[SubjectivityCategoryV49]
        FluctuationCategoryV49_cls_type: Type[FluctuationCategoryV49]
        DFRSMetrics_cls_type: Type[DFRSMetricsV49]
        SubjectivityKeywordEntry_cls_type: Type[SubjectivityKeywordEntryV49]
        FluctuationPatternEntry_cls_type: Type[FluctuationPatternEntryV49]
        FeedbackContextV49_cls_type: Type[FeedbackContextV49]
        LLMScoreKeys_cls_type: Type[ScoreKeysLLMEnumType_hint] # 正しくは ScoreKeys.LLM
        ValidationError_cls_type: Optional[Type[ValidationError]]
    else:
        ConfigProtoType = 'ConfigProtocol'
        SettingsProtoType = 'SettingsProtocol'
        FeedbackStrategyProtoType = 'FeedbackStrategyProtocol'
        StyleManagerProtoType = Optional['StyleManagerProtocol']

    def __init__(self,
                 config: ConfigProtoType, # type: ignore
                 feedback_strategy: FeedbackStrategyProtoType, # type: ignore
                 style_manager: StyleManagerProtoType = None): # type: ignore
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}.v5.5") # バージョン更新
        self.feedback_strategy = feedback_strategy
        self.style_manager = style_manager
        system_version_for_init = getattr(self.config, 'SYSTEM_VERSION', "N/A_PromptBuilderV49_v5.5")
        self.logger.info(f"PromptBuilderV49 (System Version: {system_version_for_init}) 初期化開始...")

        # 必須のEnum/Modelクラスをロード (v5.2/v5.4の堅牢なロード処理を維持)
        required_classes_map: Dict[str, Tuple[str, Optional[type]]] = {
            "SubjectiveIntensityLevel": ("SubjectiveIntensityLevel_cls", enum.EnumMeta),
            "PsychologicalPhaseV49": ("PsychologicalPhaseV49_cls", enum.EnumMeta),
            "EmotionalToneV49": ("EmotionalToneV49_cls", enum.EnumMeta),
            "SubjectivityCategoryV49": ("SubjectivityCategoryV49_cls", enum.EnumMeta),
            "FluctuationCategoryV49": ("FluctuationCategoryV49_cls", enum.EnumMeta),
            "DFRSMetricsV49": ("DFRSMetrics_cls", enum.EnumMeta),
            "SubjectivityKeywordEntryV49": ("SubjectivityKeywordEntry_cls", _get_global_type('BaseModel') if PYDANTIC_AVAILABLE else object),
            "FluctuationPatternEntryV49": ("FluctuationPatternEntry_cls", _get_global_type('BaseModel') if PYDANTIC_AVAILABLE else object),
            "FeedbackContextV49": ("FeedbackContextV49_cls", _get_global_type('BaseModel') if PYDANTIC_AVAILABLE else object),
            "ScoreKeys.LLM": ("LLMScoreKeys_cls", enum.EnumMeta),
        }
        if PYDANTIC_AVAILABLE:
            required_classes_map["ValidationError"] = ("ValidationError_cls", ValueError)

        error_summary_pb_init_list: List[str] = []
        for class_name_str, (attr_name_str, expected_meta_or_type) in required_classes_map.items():
            loaded_class = None
            if "." in class_name_str: # ScoreKeys.LLM のようなネストされた属性に対応
                parent_name, child_name = class_name_str.split(".", 1)
                parent_obj = _get_global_type(parent_name)
                if parent_obj: loaded_class = getattr(parent_obj, child_name, None)
            else:
                loaded_class = _get_global_type(class_name_str)

            if not loaded_class:
                error_summary_pb_init_list.append(f"'{class_name_str}' ({attr_name_str}) が見つかりません。")
                setattr(self, attr_name_str, None)
                continue
            
            actual_meta_or_type = type(loaded_class)
            type_check_ok = False
            if expected_meta_or_type is _get_global_type('BaseModel') and PYDANTIC_AVAILABLE:
                 if isinstance(loaded_class, type) and issubclass(loaded_class, _get_global_type('BaseModel')): type_check_ok = True # type: ignore
            elif expected_meta_or_type is enum.EnumMeta:
                if isinstance(loaded_class, enum.EnumMeta): type_check_ok = True
            elif expected_meta_or_type is ValueError and PYDANTIC_AVAILABLE: # ValidationError用
                 if isinstance(loaded_class, type) and issubclass(loaded_class, ValueError): type_check_ok = True # type: ignore
            elif expected_meta_or_type is object and not PYDANTIC_AVAILABLE: # Pydanticなしの場合のフォールバック
                type_check_ok = True
            
            if type_check_ok: setattr(self, attr_name_str, loaded_class)
            else:
                error_summary_pb_init_list.append(f"'{class_name_str}' ({attr_name_str})型不正.期待:{expected_meta_or_type.__name__ if expected_meta_or_type else 'N/A'},実際:{actual_meta_or_type.__name__}")
                setattr(self, attr_name_str, None)

        # LLMScoreKeys_cls のフォールバック処理 (v5.2と同様)
        if not hasattr(self, 'LLMScoreKeys_cls') or getattr(self, 'LLMScoreKeys_cls') is None:
            self.logger.critical("CRITICAL: ScoreKeys.LLM Enum がロードできませんでした。フォールバック用のダミーEnumを使用します。")
            @enum.unique
            class DummyLLMScoreKeysEnum(str, enum.Enum):
                OVERALL = "overall" # v5.2で "overall" に修正済み
                CONSISTENCY = "consistency"
                NATURALNESS = "naturalness"
                EMOTIONAL_DEPTH = "emotionalDepth"
                CONSTRAINTS = "constraints"
                ATTRACTIVENESS = "attractiveness"
                COMPLEXITY = "complexity"
                UNKNOWN = "unknown_llm_score"
            self.LLMScoreKeys_cls = DummyLLMScoreKeysEnum # type: ignore
            if 'ScoreKeys.LLM' not in [err.split("'")[1] for err in error_summary_pb_init_list if "'" in err]:
                 error_summary_pb_init_list.append("'ScoreKeys.LLM' がロードできなかったためダミーEnumを使用")

        if error_summary_pb_init_list:
            final_error_summary_pb = f"PromptBuilderV49初期化エラー: 必須クラス/Enumのロード/検証失敗: {'; '.join(error_summary_pb_init_list)}"
            self.logger.critical(final_error_summary_pb)
            raise ImportError(final_error_summary_pb)
        
        self.logger.info(f"PromptBuilderV49 (System Version: {system_version_for_init}) の初期化が正常に完了しました。")

    @staticmethod
    def format_character_sheet(character_data: Dict[str, Any]) -> str:
        # (このメソッドはv5.2/v5.4から変更なし)
        logger = logging.getLogger(f"{PromptBuilderV49.__module__}.PromptBuilderV49.format_character_sheet")
        if not isinstance(character_data, dict):
            logger.warning("キャラクターデータが辞書形式ではありません。空のシートを返します。")
            return "# キャラクター設定： (データ不正)"
        sheet_parts: List[str] = []
        name = str(character_data.get('name', '(キャラクター名不明)')).strip()
        sheet_parts.append(f"# キャラクター設定： {name if name else '(キャラクター名なし)'}")
        basic_info_parts = ["## 基本情報"]
        basic_fields = {"役割": 'role', "年齢": 'age', "性別": 'gender', "外見": 'appearance'}
        for label, key in basic_fields.items():
            value = character_data.get(key)
            if value is not None and str(value).strip():
                basic_info_parts.append(f"- **{label}:** {str(value).strip()}")
        if len(basic_info_parts) > 1: sheet_parts.append("\n".join(basic_info_parts))
        
        personality_traits = character_data.get("personalityTraits", [])
        if isinstance(personality_traits, list) and personality_traits:
            valid_traits = [str(t).strip() for t in personality_traits if str(t).strip()]
            if valid_traits:
                sheet_parts.append("\n## 性格・特性")
                sheet_parts.extend([f" - {trait}" for trait in valid_traits])
        
        detailed_sections_map: Dict[str, Dict[str, str]] = {
            "能力・役割": {"主な能力": 'abilities', "物語上の役割": 'storyRole'},
            "内面・背景": {"動機/目的": 'motivations', "内面的葛藤": 'internalConflicts',
                         "価値観/信念": 'values', "過去の重要な経験": 'pastExperiences',
                         "二面性/隠れた側面": 'duality'},
            "対話・行動様式": {"使用する一人称": 'pronouns', "口調/話し方の特徴": 'speechPatterns',
                             "特徴的な非言語的合図/仕草": 'nonverbalCues', "他者との関係性": 'relationships',
                             "コミュニケーションスタイル": 'communicationStyle',
                             "感情の主な表現方法": 'emotionalExpressions',
                             "典型的な感情パターン": 'emotionalPatterns',
                             "状況による対話スタイルの変化": 'dialogueStyleShift',
                             "タブー/NG行動": 'taboos'},
            "主観描写特性": {"主観描写の傾向": 'subjective_tendencies',
                             "内的モノローグのスタイル": 'internal_monologue_style',
                             "知覚フィルターの偏り": 'perception_filter_bias',
                             "感情を主にどう表現するか": 'emotion_expression_method'}
        }
        for section_title, fields_in_section_map in detailed_sections_map.items():
            section_content_parts = []
            for display_label, data_key in fields_in_section_map.items():
                value = character_data.get(data_key)
                if value is not None:
                    value_str = ""
                    if isinstance(value, list):
                        valid_list_items = [str(item).strip() for item in value if str(item).strip()]
                        if valid_list_items: value_str = ", ".join(valid_list_items)
                    elif isinstance(value, dict):
                        dict_items = [f"{k_}: {v_}" for k_, v_ in value.items() if v_ is not None and str(v_).strip()]
                        if dict_items: value_str = "; ".join(dict_items)
                    else: value_str = str(value).strip()
                    if value_str: section_content_parts.append(f"- **{display_label}:** {value_str}")
            if section_content_parts: sheet_parts.append(f"\n## {section_title}"); sheet_parts.extend(section_content_parts)
        return "\n".join(sheet_parts).strip()

    @staticmethod
    def format_scene_sheet(scene_data: Dict[str, Any]) -> str:
        # (このメソッドはv5.2/v5.4から変更なし)
        logger = logging.getLogger(f"{PromptBuilderV49.__module__}.PromptBuilderV49.format_scene_sheet")
        if not isinstance(scene_data, dict): logger.warning("シーンデータが辞書形式ではありません。空のシートを返します。"); return "# シーン設定： (データ不正)"
        sheet_parts: List[str] = []; name = str(scene_data.get('name', '(シーン名不明)')).strip(); sheet_parts.append(f"# シーン設定： {name if name else '(シーン名なし)'}")
        basic_info_parts = ["## 基本設定"]; basic_fields = {"場所": 'location', "時間帯": 'time', "全体的な雰囲気": 'atmosphere'}
        for label, key in basic_fields.items(): value = scene_data.get(key, '(未設定)'); basic_info_parts.append(f"- **{label}:** {str(value).strip()}")
        sheet_parts.append("\n".join(basic_info_parts))
        
        env_sensory_parts = ["\n## 環境・感覚情報"]; env_fields = {"主要な環境要素": 'environmentElements', "空間の特性": 'spatialCharacteristics'}
        for label, key in env_fields.items():
            value = scene_data.get(key); formatted_value_str = ""
            if value: formatted_value_str = PromptBuilderV49._format_complex_scene_field(value, list_prefix="  - ")
            if formatted_value_str.strip(): env_sensory_parts.append(f"- **{label}:**\n{formatted_value_str}")
        sensory_elements = scene_data.get('sensoryElements')
        if isinstance(sensory_elements, dict) and sensory_elements:
            env_sensory_parts.append("- **五感情報:**")
            for sense_type, sense_desc in sensory_elements.items():
                if str(sense_desc).strip(): env_sensory_parts.append(f"  - **{str(sense_type).capitalize()}:** {str(sense_desc).strip()}")
        if len(env_sensory_parts) > 1: sheet_parts.append("\n".join(env_sensory_parts))
        
        context_purpose_parts = ["\n## 状況と目的"]; context_purpose_parts.append(f"- **このシーンの主な目的・ゴール:** {str(scene_data.get('purpose', '(未設定)')).strip()}"); context_fields = {"直前の出来事や状況": 'previousEvents', "キャラクターの初期配置や位置関係": 'characterPositions', "キャラクターの初期感情状態": 'emotionalStates', "各キャラクターの対話目標": 'dialogueObjectives'}
        for label, key in context_fields.items():
            value = scene_data.get(key); formatted_value_str = ""
            if value: formatted_value_str = PromptBuilderV49._format_complex_scene_field(value, dict_prefix="  - ", list_prefix="  - ");
            if formatted_value_str.strip(): context_purpose_parts.append(f"- **{label}:**\n{formatted_value_str}")
        if len(context_purpose_parts) > 1: sheet_parts.append("\n".join(context_purpose_parts))
        
        narrative_constr_parts = ["\n## 物語要素と制約"]; narrative_fields = {"シーンに含めるべき伏線要素": 'foreshadowingElements', "期待される対話構造のアイデア": 'dialogueStructure', "重要な対話の転換点(プロットポイント)": 'dialoguePivots', "その他の制約条件や指示": 'constraints'}
        for label, key in narrative_fields.items():
            value = scene_data.get(key); formatted_value_str = ""
            if value: formatted_value_str = PromptBuilderV49._format_complex_scene_field(value, list_prefix="  - ");
            if formatted_value_str.strip(): narrative_constr_parts.append(f"- **{label}:**\n{formatted_value_str}")
        if len(narrative_constr_parts) > 1: sheet_parts.append("\n".join(narrative_constr_parts))
        
        notes = scene_data.get('notes')
        if notes and str(notes).strip(): sheet_parts.append("\n## その他特記事項・メモ"); sheet_parts.append(str(notes).strip())
        return "\n".join(sheet_parts).strip()

    @staticmethod
    def _format_complex_scene_field(value: Any, list_prefix: str = "- ", dict_prefix: str = "- ") -> str:
        # (このメソッドはv5.2/v5.4から変更なし)
        base_indent = "  "; parts = []
        try:
            if isinstance(value, dict) and value:
                for k, v_item in value.items():
                    v_str = str(v_item).strip() if v_item is not None else ""
                    if v_str: parts.append(f"{base_indent}{dict_prefix}{str(k)}: {v_str}")
            elif isinstance(value, list) and value:
                for item in value:
                    item_str = str(item).strip() if item is not None else ""
                    if item_str: parts.append(f"{base_indent}{list_prefix}{item_str}")
            elif isinstance(value, str) and value.strip():
                return "\n".join(f"{base_indent}{line}" for line in value.strip().splitlines())
            return "\n".join(parts)
        except Exception as e_format:
            logging.getLogger(f"{PromptBuilderV49.__module__}.PromptBuilderV49._format_complex_scene_field").warning(f"複雑なシーンフィールドの整形中にエラー (入力型: {type(value)}): {e_format}")
            return f"{base_indent}(フォーマットエラー: 予期せぬデータ型または内容)"

    def _get_enum_member_from_value( self, enum_class: Optional[Type[TEnum]], value: Any, default_member_override: Optional[TEnum] = None) -> Optional[TEnum]:
        # (このメソッドはv5.2/v5.4から変更なし、ログ強化済み)
        logger_gev = self.logger.getChild("_get_enum_member_from_value_v5.2") # Keep version for this stable helper
        if not enum_class or not isinstance(enum_class, enum.EnumMeta):
            logger_gev.error(f"無効なEnumクラス '{enum_class}' (型: {type(enum_class)}) が指定されました。Noneを返します。")
            return None
        
        default_enum_unknown: Optional[TEnum] = getattr(enum_class, 'UNKNOWN', None) # type: ignore
        effective_default_member = default_member_override if default_member_override is not None else default_enum_unknown

        if value is None:
            default_name = getattr(effective_default_member, 'name', 'None') if effective_default_member else 'None (default_enum_unknownもNone)'
            logger_gev.debug(f"入力値がNoneです。Enum '{enum_class.__name__}' のデフォルトメンバー '{default_name}' を返します。")
            return effective_default_member
        
        if isinstance(value, enum_class):
            return value
            
        val_str_to_convert = str(value).strip()
        try:
            member_instance: TEnum = enum_class(val_str_to_convert) # type: ignore
            member_name_for_log = getattr(member_instance, 'name', str(member_instance))
            
            is_input_str_literally_unknown = val_str_to_convert.lower() == 'unknown'
            is_resolved_to_actual_unknown_member = default_enum_unknown and member_instance == default_enum_unknown
            
            if is_resolved_to_actual_unknown_member and not is_input_str_literally_unknown and \
               val_str_to_convert.lower() != getattr(default_enum_unknown, 'value', '').lower() and \
               val_str_to_convert.upper() != getattr(default_enum_unknown, 'name', ''):
                logger_gev.info(f"値 '{value}' (文字列化: '{val_str_to_convert}') は、Enum '{enum_class.__name__}' の _missing_ メソッドまたは類似のフォールバック機構により、UNKNOWN ('{getattr(default_enum_unknown, 'value', 'N/A')}') にマップされました。")
            else:
                logger_gev.debug(f"値 '{value}' (文字列化: '{val_str_to_convert}') は Enum '{enum_class.__name__}' のメンバー '{member_name_for_log}' に正常に変換されました。")
            return member_instance
        except ValueError:
            default_name_for_log_val_err = getattr(effective_default_member, 'name', 'None') if effective_default_member else 'None'
            logger_gev.warning(f"値 '{value}' (文字列化: '{val_str_to_convert}') をEnum '{enum_class.__name__}' の有効なメンバーに変換できませんでした。デフォルトの '{default_name_for_log_val_err}' を返します。")
            return effective_default_member
        except Exception as e_unhandled_conversion:
            default_name_for_log_exc = getattr(effective_default_member, 'name', 'None') if effective_default_member else 'None'
            logger_gev.error(f"Enum '{enum_class.__name__}' の値 '{value}' 変換中に予期せぬエラーが発生: {e_unhandled_conversion}。デフォルトの '{default_name_for_log_exc}' を返します。", exc_info=True)
            return effective_default_member

    def _select_relevant_keywords( self, phase_val: Optional['PsychologicalPhaseV49'], tone_val: Optional['EmotionalToneV49'], top_k: int = 3) -> List['SubjectivityKeywordEntryV49Type_hint']: # type: ignore
        # (このメソッドはv5.2/v5.4から変更なし、ログ強化済み)
        logger_srk = self.logger.getChild("_select_relevant_keywords_v5.2") # Keep version for this stable helper
        if not self.SubjectivityKeywordEntry_cls:
            logger_srk.error("SubjectivityKeywordEntryV49モデルクラスがロードされていません。キーワード選択は不可能です。空のリストを返します。")
            return []
            
        phase_enum_member: Optional[PsychologicalPhaseV49] = self._get_enum_member_from_value(self.PsychologicalPhaseV49_cls, phase_val) # type: ignore
        tone_enum_member: Optional[EmotionalToneV49] = self._get_enum_member_from_value(self.EmotionalToneV49_cls, tone_val) # type: ignore
        logger_srk.debug(f"関連主観キーワード選択開始: Phase='{getattr(phase_enum_member,'value','N/A')}', Tone='{getattr(tone_enum_member,'value','N/A')}', TopK={top_k}")

        candidate_keywords_with_scores: List[Tuple[float, SubjectivityKeywordEntryV49]] = [] # type: ignore
        
        keyword_data_map_from_config: Optional[Dict[SubjectivityCategoryV49, List[Any]]] = getattr(self.config, 'subjectivity_keywords', None) # type: ignore

        if not isinstance(keyword_data_map_from_config, dict) or not keyword_data_map_from_config:
            logger_srk.debug("設定に主観性キーワードデータ ('subjectivity_keywords') が存在しないか空です。キーワードは選択されません。")
            return []

        for category_enum_key, keyword_entry_list_raw in keyword_data_map_from_config.items():
            if not (self.SubjectivityCategoryV49_cls and isinstance(category_enum_key, self.SubjectivityCategoryV49_cls)): # type: ignore
                logger_srk.warning(f"subjectivity_keywords内のキーが不正なカテゴリ型です (型: {type(category_enum_key)}, 値: '{getattr(category_enum_key, 'value', category_enum_key)}'). このカテゴリはスキップされます。")
                continue
            if not isinstance(keyword_entry_list_raw, list):
                logger_srk.warning(f"カテゴリ '{getattr(category_enum_key,'value','?')}' のキーワードデータがリスト形式ではありません (型: {type(keyword_entry_list_raw)})。このカテゴリはスキップされます。")
                continue

            for index, entry_dict_or_model in enumerate(keyword_entry_list_raw):
                keyword_model_instance: Optional[SubjectivityKeywordEntryV49] = None # type: ignore
                
                if isinstance(entry_dict_or_model, self.SubjectivityKeywordEntry_cls): # type: ignore
                    keyword_model_instance = entry_dict_or_model
                elif isinstance(entry_dict_or_model, dict) and self.ValidationError_cls and globals().get('PYDANTIC_AVAILABLE'): # type: ignore
                    try:
                        keyword_model_instance = self.SubjectivityKeywordEntry_cls.model_validate(entry_dict_or_model) # type: ignore
                    except self.ValidationError_cls as ve_keyword: # type: ignore
                        err_detail = ve_keyword.errors(include_input=False, include_url=False) if hasattr(ve_keyword, 'errors') else str(ve_keyword)
                        logger_srk.warning(f"  SubjectivityKeywordEntryV49 Pydantic検証エラー (カテゴリ: {getattr(category_enum_key,'value','?')}, インデックス: {index}). Error: {err_detail}")
                        continue
                    except Exception as e_model_conversion:
                        logger_srk.error(f"  SubjectivityKeywordEntryV49モデルへの変換中に予期せぬエラー (カテゴリ: {getattr(category_enum_key,'value','?')}, インデックス: {index}). Error: {e_model_conversion}", exc_info=True)
                        continue
                else:
                    logger_srk.warning(f"カテゴリ '{getattr(category_enum_key,'value','?')}' 内のキーワードエントリ (インデックス: {index}) が予期せぬ型です (型: {type(entry_dict_or_model)})。スキップします。")
                    continue
                
                if not keyword_model_instance: continue

                relevance_score = float(keyword_model_instance.intensity)
                debug_score_calc_parts = [f"base_intensity={relevance_score:.2f}"]

                if phase_enum_member and keyword_model_instance.related_phases and phase_enum_member in keyword_model_instance.related_phases:
                    relevance_score *= 1.5
                    debug_score_calc_parts.append("phase_match_bonus(x1.5)")
                if tone_enum_member and keyword_model_instance.related_tones and tone_enum_member in keyword_model_instance.related_tones:
                    relevance_score *= 1.2
                    debug_score_calc_parts.append("tone_match_bonus(x1.2)")
                
                relevance_score += random.uniform(-0.001, 0.001)
                debug_score_calc_parts.append(f"final_score={relevance_score:.3f}")
                logger_srk.debug(f"  キーワード候補: '{keyword_model_instance.keyword or keyword_model_instance.pattern}' (カテゴリ: {keyword_model_instance.category.value if keyword_model_instance.category else 'N/A'}), スコア計算: {', '.join(debug_score_calc_parts)}")
                candidate_keywords_with_scores.append((relevance_score, keyword_model_instance))
        
        selected_keyword_models = [model for score, model in sorted(candidate_keywords_with_scores, key=lambda x_item: x_item[0], reverse=True)[:top_k]]
        
        if selected_keyword_models:
            display_list_items = []
            for kw_model in selected_keyword_models:
                name_part = kw_model.keyword or kw_model.pattern or "N/A"
                cat_part = kw_model.category.value if kw_model.category and hasattr(kw_model.category, 'value') else '?'
                int_part = f"{kw_model.intensity:.1f}"
                display_list_items.append(f"'{name_part}'(Cat:{cat_part},Int:{int_part})")
            selected_keywords_display_list_str = ", ".join(display_list_items)
            logger_srk.info(f"選択された関連主観キーワード ({len(selected_keyword_models)}件): {selected_keywords_display_list_str}")
        else:
            logger_srk.info("関連する主観キーワードは見つかりませんでした (または選択されませんでした)。")
            
        return selected_keyword_models

    def _select_relevant_patterns( self, phase_val: Optional['PsychologicalPhaseV49'], tone_val: Optional['EmotionalToneV49'], top_k: int = 2) -> List['FluctuationPatternEntryV49Type_hint']: # type: ignore
        # (このメソッドはv5.2/v5.4から変更なし、ログ強化済み)
        logger_srp = self.logger.getChild("_select_relevant_patterns_v5.2") # Keep version for this stable helper
        if not self.FluctuationPatternEntry_cls:
            logger_srp.error("FluctuationPatternEntryV49モデルクラスがロードされていません。揺らぎパターン選択は不可能です。空のリストを返します。")
            return []

        phase_enum_member: Optional[PsychologicalPhaseV49] = self._get_enum_member_from_value(self.PsychologicalPhaseV49_cls, phase_val) # type: ignore
        tone_enum_member: Optional[EmotionalToneV49] = self._get_enum_member_from_value(self.EmotionalToneV49_cls, tone_val) # type: ignore
        logger_srp.debug(f"関連揺らぎパターン選択開始: Phase='{getattr(phase_enum_member,'value','N/A')}', Tone='{getattr(tone_enum_member,'value','N/A')}', TopK={top_k}")

        candidate_patterns_with_scores: List[Tuple[float, FluctuationPatternEntryV49]] = [] # type: ignore
        
        fluctuation_data_map_from_config: Optional[Dict[FluctuationCategoryV49, List[Any]]] = getattr(self.config, 'fluctuation_patterns', None) # type: ignore

        if not isinstance(fluctuation_data_map_from_config, dict) or not fluctuation_data_map_from_config:
            logger_srp.debug("設定に揺らぎパターンデータ ('fluctuation_patterns') が存在しないか空です。パターンは選択されません。")
            return []

        for category_enum_key, pattern_entry_list_raw in fluctuation_data_map_from_config.items():
            if not (self.FluctuationCategoryV49_cls and isinstance(category_enum_key, self.FluctuationCategoryV49_cls)): # type: ignore
                logger_srp.warning(f"fluctuation_patterns内のキーが不正なカテゴリ型です (型: {type(category_enum_key)}, 値: '{getattr(category_enum_key, 'value', category_enum_key)}'). このカテゴリはスキップされます。")
                continue
            if not isinstance(pattern_entry_list_raw, list):
                logger_srp.warning(f"カテゴリ '{getattr(category_enum_key,'value','?')}' の揺らぎパターンデータがリスト形式ではありません (型: {type(pattern_entry_list_raw)})。このカテゴリはスキップされます。")
                continue

            for index, entry_dict_or_model in enumerate(pattern_entry_list_raw):
                pattern_model_instance: Optional[FluctuationPatternEntryV49] = None # type: ignore
                
                if isinstance(entry_dict_or_model, self.FluctuationPatternEntry_cls): # type: ignore
                    pattern_model_instance = entry_dict_or_model
                elif isinstance(entry_dict_or_model, dict) and self.ValidationError_cls and globals().get('PYDANTIC_AVAILABLE'): # type: ignore
                    try:
                        pattern_model_instance = self.FluctuationPatternEntry_cls.model_validate(entry_dict_or_model) # type: ignore
                    except self.ValidationError_cls as ve_pattern: # type: ignore
                        err_detail = ve_pattern.errors(include_input=False, include_url=False) if hasattr(ve_pattern, 'errors') else str(ve_pattern)
                        logger_srp.warning(f"  FluctuationPatternEntryV49 Pydantic検証エラー (カテゴリ: {getattr(category_enum_key,'value','?')}, インデックス: {index}). Error: {err_detail}")
                        continue
                    except Exception as e_model_conversion_pt:
                        logger_srp.error(f"  FluctuationPatternEntryV49モデルへの変換中に予期せぬエラー (カテゴリ: {getattr(category_enum_key,'value','?')}, インデックス: {index}). Error: {e_model_conversion_pt}", exc_info=True)
                        continue
                else:
                    logger_srp.warning(f"カテゴリ '{getattr(category_enum_key,'value','?')}' 内の揺らぎパターンエントリ (インデックス: {index}) が予期せぬ型です (型: {type(entry_dict_or_model)})。スキップします。")
                    continue
                
                if not pattern_model_instance: continue

                relevance_score = float(pattern_model_instance.intensity)
                debug_score_calc_parts = [f"base_intensity={relevance_score:.2f}"]
                if phase_enum_member and pattern_model_instance.related_phases and phase_enum_member in pattern_model_instance.related_phases:
                    relevance_score *= 1.5
                    debug_score_calc_parts.append("phase_match_bonus(x1.5)")
                if tone_enum_member and pattern_model_instance.related_tones and tone_enum_member in pattern_model_instance.related_tones:
                    relevance_score *= 1.2
                    debug_score_calc_parts.append("tone_match_bonus(x1.2)")
                
                relevance_score += random.uniform(-0.001, 0.001)
                debug_score_calc_parts.append(f"final_score={relevance_score:.3f}")
                logger_srp.debug(f"  揺らぎパターン候補: '{pattern_model_instance.pattern or pattern_model_instance.keyword}' (カテゴリ: {pattern_model_instance.category.value if pattern_model_instance.category else 'N/A'}), スコア計算: {', '.join(debug_score_calc_parts)}")
                candidate_patterns_with_scores.append((relevance_score, pattern_model_instance))
        
        selected_pattern_models = [model for score, model in sorted(candidate_patterns_with_scores, key=lambda x_item: x_item[0], reverse=True)[:top_k]]
        
        if selected_pattern_models:
            display_list_items_pt = []
            for pt_model in selected_pattern_models:
                name_part_pt = pt_model.pattern or pt_model.keyword or "N/A"
                cat_part_pt = pt_model.category.value if pt_model.category and hasattr(pt_model.category, 'value') else '?'
                int_part_pt = f"{pt_model.intensity:.1f}"
                display_list_items_pt.append(f"'{name_part_pt}'(Cat:{cat_part_pt},Int:{int_part_pt})")
            selected_patterns_display_list_str = ", ".join(display_list_items_pt)
            logger_srp.info(f"選択された関連揺らぎパターン ({len(selected_pattern_models)}件): {selected_patterns_display_list_str}")
        else:
            logger_srp.info("関連する揺らぎパターンは見つかりませんでした (または選択されませんでした)。")
            
        return selected_pattern_models

    def _build_subjective_instruction_v49( self, intensity_val: 'SubjectiveIntensityLevelType_hint', phase_val: Optional['PsychologicalPhaseV49EnumType_hint'] = None, tone_val: Optional['EmotionalToneV49EnumType_hint'] = None, char_internal_style: Optional[str] = None, char_perception_bias: Optional[str] = None, char_emotion_method: Optional[str] = None) -> str: # type: ignore
        # (このメソッドはv5.2/v5.4から見出し調整のみ)
        if not self.SubjectiveIntensityLevel_cls: self.logger.critical("CRITICAL: SubjectiveIntensityLevel_cls未ロード。主観描写指示生成不可。"); return "**【主観描写と感情表現の深化】(5.2.3.1. および 5.3. の具体化):** (システムエラー: 強度Enum定義なし)"
        current_intensity_enum = self._get_enum_member_from_value( self.SubjectiveIntensityLevel_cls, intensity_val, self.SubjectiveIntensityLevel_cls.MEDIUM ); # type: ignore
        if not current_intensity_enum: self.logger.error(f"主観強度Enumメンバー変換失敗(入力:{intensity_val})。MEDIUMフォールバック。"); current_intensity_enum = self.SubjectiveIntensityLevel_cls.MEDIUM # type: ignore
        if current_intensity_enum == self.SubjectiveIntensityLevel_cls.OFF: return "**【主観描写と感情表現の深化】(5.2.3.1. および 5.3. の具体化):** 主観描写はオフです（客観的かつ行動中心の描写を基本とします）。" # type: ignore
        
        intensity_templates = {
            self.SubjectiveIntensityLevel_cls.LOW: "控えめな主観性（軽い思考や感覚、短い内省）を地の文に時折含める程度にしてください。",
            self.SubjectiveIntensityLevel_cls.MEDIUM: "適度な主観描写（思考、感情、知覚）を地の文に自然に織り交ぜ、キャラクターの視点や感情が読者に伝わるようにしてください。",
            self.SubjectiveIntensityLevel_cls.HIGH: "強い主観描写（内面の思考、深い感情、鮮明な記憶の断片、五感を通した鋭い感覚）を積極的に地の文に盛り込み、キャラクターの内面世界を深く掘り下げてください。",
            self.SubjectiveIntensityLevel_cls.EXTREME: "極めて強い主観描写（意識の流れに近い自由な連想、断片的な思考の連続、奔流のような感覚情報、現実と幻想の混濁など）を重視し、読者をキャラクターの精神世界に没入させてください。"
        }
        base_instruction = intensity_templates.get(current_intensity_enum, intensity_templates[self.SubjectiveIntensityLevel_cls.MEDIUM]) # type: ignore
        
        selected_keywords_models = self._select_relevant_keywords(phase_val, tone_val, top_k=3)
        selected_patterns_models = self._select_relevant_patterns(phase_val, tone_val, top_k=2)
        
        expression_guide_parts: List[str] = []
        kw_examples: List[str] = []
        pt_examples: List[str] = []
        
        if selected_keywords_models:
            temp_kw_examples = []
            for kw in selected_keywords_models:
                name = kw.keyword or kw.pattern
                cat = kw.category.value if kw.category and hasattr(kw.category, 'value') else '不明'
                temp_kw_examples.append(f"「{name}」(カテゴリ: {cat})")
            kw_examples = temp_kw_examples
        if kw_examples: expression_guide_parts.append(f"関連性の高い主観キーワード例: {', '.join(kw_examples)}")
        
        if selected_patterns_models:
            temp_pt_examples = []
            for pt in selected_patterns_models:
                name = pt.pattern or pt.keyword
                cat = pt.category.value if pt.category and hasattr(pt.category, 'value') else '不明'
                temp_pt_examples.append(f"「{name}」(カテゴリ: {cat})")
            pt_examples = temp_pt_examples
        if pt_examples: expression_guide_parts.append(f"関連性の高い表現揺らぎパターン例: {', '.join(pt_examples)}")
        
        expression_guide_str = "";
        if expression_guide_parts: expression_guide_str = "、".join(expression_guide_parts) + "。これらを参考に、描写に織り交ぜることを検討してください。"
        
        character_specific_hints_parts: List[str] = []
        if char_internal_style and char_internal_style.strip(): character_specific_hints_parts.append(f"内的モノローグのスタイルは「{char_internal_style}」を意識してください。")
        if char_perception_bias and char_perception_bias.strip(): character_specific_hints_parts.append(f"知覚フィルターの偏り（例：「{char_perception_bias}」）を反映させてください。")
        if char_emotion_method and char_emotion_method.strip(): character_specific_hints_parts.append(f"感情表現は主に「{char_emotion_method}」という方法で行われます。")
        if character_specific_hints_parts: expression_guide_str += " " + " ".join(character_specific_hints_parts)
        
        instruction_parts = [
            f"**【主観描写と感情表現の深化】(基本方針: {current_intensity_enum.value if hasattr(current_intensity_enum,'value') else 'N/A'}):** {base_instruction}"
        ]
        if expression_guide_str.strip():
            instruction_parts.append(f"  - **表現ガイド:** {expression_guide_str.strip()}")
        instruction_parts.extend([
            "  - 他者の言動や周囲の状況は、可能な限り視点キャラクターの知覚（五感）や解釈フィルター（例：疑念、期待、恐怖など）を通して描写してください。",
            "  - キャラクターの感情は、直接的な言葉（例：「悲しい」「嬉しい」）で説明するのではなく、その感情が引き起こす具体的な行動、表情の変化、声のトーンの変動、視線の動き、身体的な反応（例：心拍数の上昇、手の震え、息遣いの変化）、比喩表現、あるいは言葉にできない言外のニュアンス（サブテキスト）によって示唆することを強く意識してください。"
        ])
        return "\n".join(instruction_parts)

    def _get_phase_tone_instruction_v49( self, phase_val: Optional['PsychologicalPhaseV49EnumType_hint'], tone_val: Optional['EmotionalToneV49EnumType_hint']) -> str: # type: ignore
        # (このメソッドはv5.2/v5.4から見出し調整のみ)
        modulation_enabled = True;
        if hasattr(self.config, 'feature_flags') and self.config.feature_flags: modulation_enabled = getattr(self.config.feature_flags, 'phase_tone_prompt_modulation_enabled', True) # type: ignore
        
        current_phase_enum = self._get_enum_member_from_value(self.PsychologicalPhaseV49_cls, phase_val)
        current_tone_enum = self._get_enum_member_from_value(self.EmotionalToneV49_cls, tone_val)
        
        unknown_phase_val = getattr(self.PsychologicalPhaseV49_cls, "UNKNOWN", object()) # type: ignore
        unknown_tone_val = getattr(self.EmotionalToneV49_cls, "UNKNOWN", object()) # type: ignore
        
        if not modulation_enabled or not current_phase_enum or not current_tone_enum or current_phase_enum == unknown_phase_val or current_tone_enum == unknown_tone_val:
            phase_display = getattr(current_phase_enum, 'value', 'N/A') if current_phase_enum else 'N/A'
            tone_display = getattr(current_tone_enum, 'value', 'N/A') if current_tone_enum else 'N/A'
            self.logger.debug(f"位相/トーン指示生成スキップ: Modulation無効、または有効な位相/トーン未指定 (Phase: {phase_display}, Tone: {tone_display})")
            return ""
            
        templates: Optional[Dict[str, Dict[str, str]]] = None
        if hasattr(self.config, 'loaded_external_configs') and self.config.loaded_external_configs: # type: ignore
            templates = getattr(self.config.loaded_external_configs, 'phase_tone_prompt_templates', None) # type: ignore
            
        if not isinstance(templates, dict) or not templates:
            self.logger.warning("phase_tone_prompt_templates が設定にないか空です。位相/トーン指示は生成されません。")
            return ""
            
        phase_key_str = current_phase_enum.value # type: ignore
        tone_key_str = current_tone_enum.value # type: ignore
        instruction_text: str = ""
        
        if phase_key_str in templates and isinstance(templates[phase_key_str], dict):
            instruction_text = templates[phase_key_str].get(tone_key_str, "")
            
        if not instruction_text:
            available_tones = list(templates.get(phase_key_str, {}).keys()) if isinstance(templates.get(phase_key_str), dict) else "N/A"
            self.logger.debug(f"位相/トーン指示: テンプレートなし (P='{phase_key_str}', T='{tone_key_str}'). '{phase_key_str}'で利用可能なTones: {available_tones}")
        elif instruction_text.strip():
            # 見出しを【決定版プロンプト Part 5】の構造に合わせる
            return f"**目標とする心理的位相・感情トーンの強調指示 ({phase_key_str}/{tone_key_str}):** {instruction_text.strip()}\n"
        return ""

    def create_dialogue_prompt( self, character_a_data: Dict[str, Any], character_b_data: Dict[str, Any], scene_info_data: Dict[str, Any], target_length: int, settings: SettingsProtocol, phase_val: Optional['PsychologicalPhaseV49EnumType_hint'] = None, tone_val: Optional['EmotionalToneV49EnumType_hint'] = None) -> str: # type: ignore
        if not all([self.SubjectiveIntensityLevel_cls, self.PsychologicalPhaseV49_cls, self.EmotionalToneV49_cls]): self.logger.critical("PromptBuilder初期化エラー: 必須Enumクラス(Intensity/Phase/Tone)未ロード。"); raise RuntimeError("PromptBuilder is not properly initialized (required Enum classes not loaded).")
        
        subj_intensity_val_from_settings: Any = getattr(settings, 'subjective_intensity', self.SubjectiveIntensityLevel_cls.MEDIUM) # type: ignore
        subj_intensity_enum = self._get_enum_member_from_value( self.SubjectiveIntensityLevel_cls, subj_intensity_val_from_settings, self.SubjectiveIntensityLevel_cls.MEDIUM );
        if not subj_intensity_enum: self.logger.error(f"主観強度のEnumメンバー変換に失敗 (設定値: {subj_intensity_val_from_settings})。MEDIUMを使用します。"); subj_intensity_enum = self.SubjectiveIntensityLevel_cls.MEDIUM # type: ignore
        
        phase_obj_for_prompt = self._get_enum_member_from_value(self.PsychologicalPhaseV49_cls, phase_val)
        tone_obj_for_prompt = self._get_enum_member_from_value(self.EmotionalToneV49_cls, tone_val)
        subjective_focus_enabled = getattr(settings, 'subjective_focus', True)
        
        self.logger.info(f"初期対話プロンプト作成開始 (v5.5): TargetLength={target_length}, Mode='{getattr(settings, 'dialogue_mode', 'auto')}', SubjFocus={'ON' if subjective_focus_enabled else 'OFF'}, SubjIntensity='{subj_intensity_enum.value if hasattr(subj_intensity_enum, 'value') else 'N/A'}', IntendedPhase='{getattr(phase_obj_for_prompt, 'value', 'N/A')}', IntendedTone='{getattr(tone_obj_for_prompt, 'value', 'N/A')}'")

        char_a_name = str(character_a_data.get('name', 'キャラクターA')).strip()
        min_len = max(100, int(target_length * 0.75))
        max_len = int(target_length * 1.25)

        # --- 【決定版プロンプト Part 5】の構造に従ってプロンプト部品を定義 ---
        # 5.1. 基本哲学とAIロール
        # (ユーザー指示により「後続システムでの処理～」の事前説明は削除)
        prompt_part_5_1_philosophy_role = (
            "**5. 文体・表現スタイルと読者体験の最適化指針 (StoryWeaverAI-VisualPro準拠)**\n\n"
            "**5.1. 基本哲学とAIロール：読者中心の最高品質な物語体験の創造**\n"
            "   - **あなたの役割:** あなたは経験豊富な小説家「StoryWeaverAI-VisualPro」です。提供されたキャラクターおよびシーン設定（JSONデータ）を深く解釈し、現代の読者が直感的に理解し、感情移入できる、文学的価値を持つ卓越した小説シーンを創造してください。\n"
            "   - **至上の使命:** あなたの生成するテキストは、それ自体が完成された「読み物」として成立する品質を目指します。\n"
            "   - **核となるバランスの追求:**\n"
            "       - **魅力と深みの両立:** 漫画的・映像的な躍動感（テンポ、ダイナミズム、明確な感情表出）と、小説ならではの心理描写の深み、言葉選びの妙、自然で美しい読み心地を高い次元で両立させてください。\n"
            "       - **描写の戦略（Show & Tell）：** 感情や状況は、具体的な行動、表情、五感を通じた情景描写（Show）で読者に追体験させることを最優先とします。ただし、複雑な内面や行動だけでは伝達困難な機微、物語進行上不可欠な情報は、的確かつ簡潔な「語る描写（Tell）」（モノローグや地の文による思考提示や説明）を戦略的に活用してください。このバランスが読者の深い没入を促します。\n"
            "       - **表現の抑制と解放の調和：** 全体として自然で洗練された抑制の効いた表現を基本としつつ、シーンの目的（例：コメディ、クライマックス、シリアス）に応じて、読者の心を掴み物語を推進する「勢い」「感情強度」「面白さ」を、キャラクター、状況、作品トーンと完全に調和させ、唐突さや安易さを排除した上で効果的に発露させてください。\n"
            "   - **最優先事項：読者体験の最大化。** 常に「読者がどう感じるか」を第一に、読者の自然な受容性、共感度、没入感を最大化してください。\n"
            "   - **入力JSONデータの活用について:** 本プロンプトの各項目で『（入力JSON内の sceneInfo.パラメータ名 等を参照）』と記載されている箇所は、提供されたキャラクター設定（characterA, characterB）およびシーン設定（sceneInfo）内の該当するキーの値を最優先で参照し、指示に従ってください。該当するキーが存在しない、または値が未設定の場合は、本プロンプトの基本原則および前後の文脈から最も適切と思われる判断を下してください。"
        )

        # 5.2. コア変換原則 (入力JSONデータを先に展開)
        sheet_a_str = self.format_character_sheet(character_a_data)
        sheet_b_str = self.format_character_sheet(character_b_data)
        scene_sheet_str = self.format_scene_sheet(scene_info_data)
        
        prompt_part_5_2_core_principles = (
            "\n\n**5.2. コア変換原則：原作尊重と創造的洗練、読者体験の最適化**\n"
            "## 5.2.1. 原作核心尊重と創造的洗練:\n"
            f"### 提供シーン設定\n{scene_sheet_str}\n"
            "   - **忠実性:** 提供されたキャラクター設定（下記参照）、上記シーン設定の核心（性格、目的、状況、雰囲気、セリフのニュアンス等）を完全に反映してください。\n"
            "   - **創造的拡張:** 設定の行間にある未言の感情、シーンディテール、五感情報（下記【5.3. 五感表現と内的世界の深化】参照）、微細な反応、記憶の断片等を、物語の質を高める範囲で効果的に補強してください。\n"
            "   - **制約（雰囲気・経済性）：** 原作の雰囲気・トーンを尊重し、読者の想像力を自然に刺激する範囲の描写に留めてください。蛇足や冗長性を徹底的に排除し、常に「言葉の経済性」を意識してください。ただし、キャラクターの重要な内面描写や、シーンの雰囲気を深く伝えるための繊細な描写が求められる場合は、読者の理解と共感を優先し、必要な言葉を惜しまないでください。（詳細は5.2.3.1.参照）\n"
            "   - **物語的役割の意識:** シーンの物語全体での役割（例：上記シーン設定内の sceneInfo.purpose や sceneInfo.scene_role_in_episode 等を参照し、感情の起伏、テーマ性への貢献など）を深く理解し、描写の強弱や情報提示順序を最適化してください。\n"
            "   - **シーンタイプ別戦略:** （sceneInfo.target_novel_style や sceneInfo.scene_role_in_episode を参考に判断）\n"
            "       - コメディシーン： キャラクターの魅力と状況の面白さを、軽快なテンポ、生き生きとした表現、巧妙な伏線で追求してください。\n"
            "       - シリアスシーン： 抑制された緊張感の中で徐々に感情を深化させ、読者の期待感を高める質の高い「引き」を構築してください。\n\n"
            "## 5.2.2. 視点と語り口の一貫性と魅力:\n"
            f"### 提供キャラクター設定\n{sheet_a_str}\n\n{sheet_b_str}\n"
            f"   - **指定視点の厳格な維持:** キャラクター「{char_a_name}」の一人称視点、または三人称「{char_a_name}」限定視点を厳守してください。視点キャラクター以外の内面（思考や直接的な感情）は記述しないでください。\n"
            "   - **キャラクター固有の語り口:** 各キャラクターの性格、背景、教育レベル、状況に応じた自然な語り口、思考パターン、感情の機微をリアルに表現してください。（上記キャラクター設定内の characterInfo.personalityTraits, characterInfo.speechPatterns, characterInfo.nonverbalCues 等を最大限活用）\n"
            "   - 各キャラクターの性格特性、価値観、動機、内的葛藤、過去の経験、他キャラクターとの関係性、コミュニケーションスタイル、そして特に【禁止事項(NG行動)】や【タブー】を言動に矛盾なく、かつ深く反映させてください。\n"
            "   - キャラクターの「二面性」や「隠された側面」が設定されている場合、それらがシーンの中で効果的に、かつ自然な形で表出するように工夫してください。\n"
            "   - **読者がキャラクターの感情や状況を追体験できるような、具体的で感覚的な描写を重視してください。**"
        )
        
        # 5.2.3. 読者体験最適化
        prompt_part_5_2_3_reader_experience = (
            "\n\n## 5.2.3. 読者体験最適化：究極の自然な読みやすさと心地よい没入感の実現\n"
            "### 5.2.3.1. 描写バランスと表現制御 (Show/Tell & Expression Control):\n"
            "   - **基本バランス（Show優先）：** 「見せる描写(Show)」を約7割、「語る描写(Tell)」を約3割としつつ、シーン内容や入力JSON内の sceneInfo.show_tell_preference（指定があれば）に応じて柔軟に調整。\n"
            "   - **戦略的Tellの活用：** 複雑な内面、行動で伝達困難な機微、誤解防止、物語進行上不可欠な情報伝達の場合に限り、的確かつ簡潔に使用。\n"
            "   - **思考動詞ポリシー：** （入力JSON内の sceneInfo.thought_verb_allowance_level、デフォルト：minimal_for_essential_clarity を参照）基本は具体的感覚・行動・状況描写で思考感情を示唆。「～と思った」等は明確性・簡潔性が必要な場合に限定。\n"
            "   - **感情表現の強度制御：** （入力JSON内の sceneInfo.emotional_expression_intensity_target、デフォルト：moderate を参照）強度を調整。過度な誇張・大袈裟な表現・不自然な反復は排除。クライマックスやコメディの「オチ」等では、物語効果を高める範囲で「やや大胆な表現」（例：意図的な誇張、感情的なセリフ強調。ただしキャラ・トーンの一貫性を損なう唐突な感情爆発や過剰演出は厳禁）も許容。その際は感情の流れを丁寧に描写し、作品全体と調和させること。\n"
            "   - **簡潔性と密度、必要な言葉：** 安易な反復を避け、多様な語彙と表現技法で簡潔性と情報密度、「言葉の経済性」を最大化。ただし、**重要な内面描写やシーンの雰囲気を深く伝えるための繊細な描写が求められる場合は、必要な言葉を惜しまず、読者の理解と共感を優先してください。**\n"
            "   - **過剰説明の回避と行間の示唆：** 原作ト書き等の補完に留め、過剰説明や蛇足的感情敷衍は厳禁。読者の想像力を刺激する行間を意識。\n"
            "### 5.2.3.2. 文構造と明瞭性 (Sentence Structure & Clarity):\n"
            "   - **文長：** （入力JSON内の sceneInfo.avg_sentence_length_preference [デフォルト15-30字], sceneInfo.max_sentence_length_limit [デフォルト55字] を参照）読みやすく単調でない長さに調整。\n"
            "   - **複雑度：** 平易な文構造を最優先。一文一意。\n"
            "   - **情報密度：** 無駄を削ぎ落とし、各文が的確かつ印象的に情報を伝達。\n"
            "### 5.2.3.3. 視覚的テンポと可読フロー (Visual Pacing & Readability Flow):\n"
            "   - **段落区切り：** （入力JSON内の sceneInfo.paragraph_break_preference [デフォルト2-4文毎] を参照）感情・状況変化、話題転換に応じて区切り、視覚的リズムを創出。\n"
            "   - **効果的改行：** 会話が続く場面、印象的な描写を際立たせる場合に意識。\n"
            "   - **対話フロースタイル：** （入力JSON内の sceneInfo.dialogue_flow_style、デフォルト：flexible_insertion を参照）セリフと地の文の配置を最適化。状況に応じてnarration_after_dialogue_blockも検討。シーンに最も自然で効果的な表現を選択。\n"
            "### 5.2.3.4. 語彙選択と独自性 (Vocabulary Choice & Originality):\n"
            "   - **基本：** 現代標準日本語、明確かつ的確。\n"
            "   - **文脈適合性：** シーン雰囲気、キャラクター感情・性格・教育レベル等に完全合致した、最も表現力豊かで自然な語彙を選択。\n"
            "   - **独自性：** （入力JSON内の sceneInfo.vocabulary_originality_level、デフォルト：standard_natural を参照）陳腐表現を避け、新鮮かつ自然な範囲の言葉を選択。奇をてらった難解語・不自然な凝った表現は回避。\n"
            "### 5.2.3.5. 結びのインパクトと余韻 (Concluding Impact & Subtlety):\n"
            "   - **技法例：** 体言止め、短い問いかけ、示唆に富む感覚描写等を効果的に用い、静かで印象的な余韻や次への期待感を創出。"
        )

        # _build_subjective_instruction_v49 と _get_phase_tone_instruction_v49 の呼び出しと組み込み
        # これらは5.2.3.1 や 5.3 の具体化、または特定の位相・トーンの強調指示として機能する
        subjective_instruction_block_text = ""
        if subjective_focus_enabled:
            char_a_internal_style = str(character_a_data.get('internal_monologue_style', ''))
            char_a_perception_bias = str(character_a_data.get('perception_filter_bias', ''))
            char_a_emotion_method = str(character_a_data.get('emotion_expression_method', ''))
            raw_subj_text = self._build_subjective_instruction_v49(
                subj_intensity_enum, phase_obj_for_prompt, tone_obj_for_prompt, # type: ignore
                char_a_internal_style, char_a_perception_bias, char_a_emotion_method
            )
            # 見出しを調整して、5.2.3.1や5.3の補足指示として自然に繋がるようにする
            subjective_instruction_block_text = raw_subj_text.replace(
                 "**主観描写指示 (v4.9β):**",
                 "**【主観描写と感情表現の深化原則】(キャラクターの視点と内面を豊かにするために):**" # より包括的な見出しに
            )
        else:
            subjective_instruction_block_text = "**【主観描写と感情表現の深化原則】:** (この生成では主観描写の強化は無効になっています)"

        phase_tone_instruction_block_text = self._get_phase_tone_instruction_v49(phase_obj_for_prompt, tone_obj_for_prompt) # type: ignore
        if phase_tone_instruction_block_text.strip():
             # 見出しを調整
             phase_tone_instruction_block_text = phase_tone_instruction_block_text.replace(
                 " - **現在の目標位相/感情トーンの強調指示",
                 "**現在の目標位相/感情トーンの強調指示"
             )
             phase_tone_instruction_block_text = f"\n{phase_tone_instruction_block_text}" # 前に改行追加

        # 5.3. 五感表現と内的世界の深化
        prompt_part_5_3_sensory_palette = (
            "\n\n**5.3. 五感表現と内的世界の深化：読者の感覚に訴える描写**\n"
            "### 5.3.1. 感覚情報の戦略的かつ抑制された活用 (ESP1):\n"
            "   - **描写原則：** 五感情報はシーン核心、キャラ感情心理を効果的に伝え、読者の没入感を高めるために戦略的かつ抑制的に使用。情報羅列・説明過多は厳禁。常に主人公の感情フィルターを通して描写。\n"
            "   - **各感覚の焦点：** （入力JSON内の sceneInfo.visual_detail_level 等の各感覚詳細レベル設定、およびキャラクター設定内の appearance, nonverbalCues を参照）\n"
            "       - **視覚 (Visuals):** 微細な表情変化、決定的仕草、風景の光と影、空間感等、映像的イメージを喚起しトーンを設定する核心情報に集中。\n"
            "       - **聴覚 (Sounds):** 会話の声トーン/抑揚、心理を示唆する環境音、物語上意味を持つ効果音、意図された静寂の質等、臨場感と雰囲気を醸成する音に集中。\n"
            "       - **触覚 (Touch/Tactile):** 温度、湿度、物の手触り、風、身体的接触等、感情状況を伝え身体感覚に訴える情報に限定。\n"
            "       - **嗅覚 (Smells):** 場所特有の匂い、季節感、記憶感情を喚起する香り等を示唆する程度に、リアリティ・ノスタルジア・雰囲気醸成に効果的な場合に限定。\n"
            "   - **内的感覚・雰囲気・記憶トリガー：** 主人公のリアルな生理的反応（心拍、呼吸、胃の感覚等）、場の「空気感」、過去の記憶の断片（鮮明かつ短いイメージ）を、現在の感情状況と自然に結びつけ、読者の共感を深める効果的なタイミングで織り込む。長大な回想は避け、現在の流れを妨げないこと。\n"
            "### 5.3.2. 感覚表現のスタイル (ESP2):\n"
            "   - **感情フィルター：** 全感覚情報は視点キャラの現在感情・心理状態フィルターを通して描写。\n"
            "   - **筆致：** 読者が追体験するような、自然で抑制の効いた筆致。\n"
            "   - **比喩表現：** （入力JSON内の sceneInfo.metaphor_usage_level、デフォルト：subtle_and_evocative を参照）作品トーンから逸脱せず、洗練され共感を呼ぶ自然な比喩を効果的に使用。感情への影響を明確かつ押し付けがましくなく、心に染み入るように表現。"
        )

        # 5.4. 漫画的・映像的リズムの導入
        prompt_part_5_4_dynamic_rhythm = (
            "\n\n**5.4. 漫画的・映像的リズムの導入：小説的自然さを最優先とした選択的適用**\n"
            "   - **導入方針：** 以下の指示は、**小説としての自然な文体と物語のリアリティを絶対に損なわない範囲で**、読者の没入感を高めるために**選択的かつ効果的に**適用してください。シリアス場面では表現抑制を優先し、コミカル場面やキャラ個性強調に効果的な場合に検討。全技法の同時適用は不要。シーン全体の目的に応じ、最も効果的なものを選択・組み合わせてください。\n"
            "### 5.4.1. コマ割りを意識した緩急制御 (DVR1):\n"
            "   - **瞬間の切り取り (Sharp Impact Moments)：** 読者に強い印象を与えたい瞬間（表情アップ、決定的セリフ、重要アイテム、効果的オノマトペ等）は、ごく短い単文・フレーズ（1-2文ブロック、2-15字程度）で構成し、速度感とインパクトを演出。効果的改行で視線集中。ただし不自然な断続や細切れ感回避。\n"
            "   - **状況・感情の自然な展開 (Stable Development Flow)：** 会話応酬、状況説明、緩やかな感情推移等は、比較的短い文（2-4文ブロック、各15-30字程度）をリズミカルに繋ぎ、安定進行。\n"
            "   - **感情・状況の深掘り (Deep Dive / Climactic Scenes)：** 山場、深い内面吐露、重要記憶描写等、読者にじっくり味わわせたい箇所は、やや長めの文（3-5文ブロック、各30-50字程度、ただし文構造はシンプル維持）や複数文段落を用い、五感情報を効果的に盛り込み、読者感情に深く静かに確実に訴求。\n"
            "### 5.4.2. 柔軟なリズムパターンと感情同期 (DVR2):\n"
            "   - **基本リズム提案と感情同期：** [\"短い文\", \"短い文\", \"中程度の文\"] 等を意識しつつ、シーンの感情曲線（静→動、緊張→緩和）と文章リズムが完全にシンクロするように文長組み合わせを柔軟かつ意図的に変化。単調さを徹底排除し、読者を物語の感情の波に自然に乗せる。\n"
            "   - **ジャンル・シーン特性に応じたペース配分：** （入力JSON内の sceneInfo.target_novel_style, sceneInfo.scene_role_in_episode, sceneInfo.comedy_pacing_parameters, sceneInfo.serious_pacing_parameters 等を考慮し）ペース調整。コメディは軽快なテンポと「タメ・キレ」、シリアスは緩急と「静寂・空白」を意識。\n"
            "### 5.4.3. オノマトペの戦略的かつ自然な活用 (DVR3 - 最重要最終調整項目):\n"
            "   - **基本方針：** 使用は極めて慎重かつ戦略的に。**小説としての品位と自然さを絶対に損なわないことが絶対条件。**（入力JSON内の sceneInfo.onomatopoeia_frequency_level、デフォルト推奨：low_and_subtle を最優先参照）\n"
            "   - **効果音の「翻訳」ポリシー：** 入力情報内の効果音指示は、音の特性、心理的影響、文体トーンを深く考察し、以下を選択。\n"
            "       1.  シーンに真に必要かつ最も表現力豊かで自然に溶け込む洗練された日本語オノマトペに翻訳。（例：「どくん、と心臓が鳴った」「ぱたぱたと軽い足音が近づいてきた」）\n"
            "       2.  **オノマトペを使わず、地の文の優れた描写（動詞、副詞、比喩等）で表現。** (sceneInfo.onomatopoeia_frequency_levelがlow_and_subtleまたはminimal_or_noneの場合はこちらを積極検討)\n"
            "       **禁止例：** 「ドッカーン！」「ズドドドド！」等の漫画特有描き文字風表現は地の文で使用厳禁。\n"
            "   - **表現技法と抑制プロトコル：**\n"
            "       - 自然な統合最優先： 地の文やセリフに自然に溶け込ませ、文章の流れを不自然に途切れさせない。\n"
            "       - 独立行強調の厳格制限： 極めて限定的（決定的瞬間で代替不可能な効果が期待できる場合のみ）かつ頻度抑制。\n"
            "       - 心理的オノマトペ： （入力JSON内の sceneInfo.psychological_onomatopoeia_style、デフォルト推奨：subtle_and_integrated を参照）読者がキャラ心情をほぼ無意識的に感じ取れる極めてさりげない形で使用検討。\n"
            "       - 反復使用禁止（原則）： 陳腐化・冗長化を招くため回避。ただし象徴的意味を持つ場合は限定的に検討可。\n"
            "       - AIによる創造的追加の厳格指針： 原作指示なしの場合、必要性は最大限慎重に判断。蛇足排除。真に効果的で物語全体の質を明確に高めると確信できる場合に限定。**コメディシーンで sceneInfo.onomatopoeia_frequency_level が許容する場合（例：moderate_for_comedic_effect）、読者体験を豊かにする質の高いオノマトペ追加を、他指示との調和（キャラ一貫性、シーン雰囲気）を最大限考慮し、より積極的に検討。**\n"
            "       - 洗練と自然さの絶対基準： シーン雰囲気・世界観に完全調和し、読者が違和感なく受容できる、最も洗練された現代的語彙を選択。陳腐さ、**いかなる形の極端な強調（例：「あああああ！」のような単純長音連続や幼稚表現は完全回避し、現実的でキャラ性格状況に合致した抑制の効いた自然な範囲に必ず調整）**を徹底排除。全体の文体との調和を常に最優先。場合によってはオノマトペ不使用がより高度な文学的表現となることも積極考慮。\n"
            "### 5.4.4. 効果的な「間」と視覚的フロー制御 (DVR4):\n"
            "   - **戦略的句読点・空白活用：** 改行、意図的な1-2行空白、三点リーダー（点数・長さで感情度合・思考途切れを示唆）、ダッシュ（長さ・スペースで場面転換明確さ・言葉強調度合を示唆）を、漫画コマ間・映画カット割りのように戦略的に使用。緊張感、逡巡、余韻、場面転換の合図として機能させ、読書体験に自然な「呼吸」リズムと「奥行き」を付与。"
        )
        
        # スタイル指示 (DialogStyleManagerV49から取得)
        style_instruction_text = ""
        style_template_name = getattr(settings, 'style_template', "standard") # デフォルトは "standard"
        if self.style_manager and style_template_name:
            try:
                style_adds_text = self.style_manager.get_style_prompt_addition_text(style_template_name)
                if style_adds_text and style_adds_text.strip():
                    # Part 5.2 と 5.3 の間に挿入することを想定し、適切な見出しを付与
                    style_instruction_text = f"\n\n**5.2.4. 適用スタイル特有の指示 ({style_template_name}):**\n{style_adds_text.strip()}"
            except Exception as e_style:
                self.logger.warning(f"スタイル指示取得エラー (Style: '{style_template_name}'): {e_style}")

        # 5.5. 出力形式と品質保証 (既存のPart 4とPart 6の指示を包含・発展)
        prompt_part_5_5_quality_assurance = (
            "\n\n**5.5. 出力形式と品質保証**\n"
            "   - **出力形式の厳守:** 生成物は、セリフと地の文が混在した、そのまま小説やシナリオとして読める形式の**日本語テキストのみ**としてください。前置きや後書き、メタ的なコメント、AIとしての自己言及（例：「承知しました」「以下に生成します」）、HTML/Markdownタグ、JSONやXMLのような構造化データマーカーは一切含めないでください。\n"
            "   - **指示のオウム返し禁止:** 本プロンプトに含まれる指示（例：`**指示:**` や `- **視点:**` など）を、生成する小説本文中にオウム返しに記述しないでください。\n"
            "   - **創造性と適切性の両立:** 生成される対話や描写は、提供された設定や文脈に対して創造的かつ文学的に適切である必要があります。\n"
            "   - **高品質な生成のためのアドバイス（再確認）：**\n"
            "       - **具体性：** 抽象的な表現よりも具体的な描写や行動を心がけてください。\n"
            "       - **一貫性：** キャラクターの言動や性格、物語のトーンに一貫性を持たせてください。\n"
            "       - **独自性：** ありきたりな展開やセリフを避け、独自のアイデアや視点を盛り込んでください。\n"
            "       - **感情のリアリティ：** 感情の動きを自然かつ深く描写し、読者が共感できるようにしてください。\n"
            "       - **テンポとリズム：** 対話と描写のバランス、文の長短などを工夫し、読者を飽きさせないテンポとリズムを意識してください。\n"
            "       - **「見せる」描写：** 説明するのではなく、行動や描写を通してキャラクターの感情や状況を「見せて」ください。\n"
            "   - **生成後の自己検証の徹底（AI内部チェックリスト）：** 生成した小説シーンが、特に以下の主要品質項目について本プロンプトの意図と指示をどの程度達成できているか、批判的に自己評価し、改善点があれば修正する意識を持つこと。\n"
            "       1.  **読者体験：** 自然で魅力的か？（読みやすさ、面白さ、没入感）\n"
            "       2.  **キャラクター整合性：** 言動は設定と完全に一貫しているか？（性格、口調、目的）\n"
            "       3.  **シーン表現効果：** 雰囲気や目的は効果的に表現されているか？\n"
            "       4.  **技法適用精度：** 指示された表現技法（Show/Tell、五感、リズム、オノマトペ等）は適切かつ効果的に使用されているか？\n"
            "       5.  **簡潔性と密度：** 冗長表現や説明過多はないか？逆に行間を読ませるべき箇所で説明しすぎはないか？\n"
            "   - **最終確認：** 生成テキスト全体を本プロンプト全指示、特にこのPart 5指針に照らし最終確認。矛盾・不自然点がないか、読者にとって最高の物語体験を提供できる品質かを常に検証。"
        )
        
        # 最終的なプロンプトの組み立て
        final_prompt_parts = [
            prompt_part_5_1_philosophy_role,
            prompt_part_5_2_core_principles, # キャラクターシートとシーンシートを含む
            # style_instruction_text は 5.2 と 5.3 の間に挿入
            (style_instruction_text if style_instruction_text.strip() else ""),
            prompt_part_5_2_3_reader_experience, # 長さ指示を含む
            subjective_instruction_block_text, # 内部で見出し調整済み
            phase_tone_instruction_block_text, # 内部で見出し調整済み
            prompt_part_5_3_sensory_palette,
            prompt_part_5_4_dynamic_rhythm,
            prompt_part_5_5_quality_assurance,
            "\n\n以上の全ての指示と設定を注意深く読み込み、最高の創造性を発揮して、読者の心を揺さぶり、かつ後続処理に適した、独創的で質の高い対話シーンを生成してください。"
        ]

        final_prompt = "\n".join(p.strip() for p in final_prompt_parts if p and p.strip()) # 各パート間の改行は1つに
        
        self.logger.info(f"初期対話プロンプト作成完了 (v5.5 - 総文字数: {len(final_prompt)})")
        # デバッグ用にプロンプト全体を出力することも検討（ただし長大になるため注意）
        # self.logger.debug(f"生成された初期プロンプト (v5.5):\n{final_prompt}")
        if len(final_prompt) < 2000: # 比較的短い場合のみ全文ログ
             self.logger.debug(f"生成された初期プロンプト (v5.5):\n{final_prompt}")
        else:
             self.logger.debug(f"生成された初期プロンプト (v5.5 - 先頭500文字と末尾500文字):\n{final_prompt[:500]}...\n...{final_prompt[-500:]}")
        return final_prompt

    def create_evaluation_prompt(self, dialogue_text: str) -> str:
        # (v5.4から変更なし - マーカー指示強調済み)
        self.logger.debug(f"評価プロンプト作成開始 (対象テキスト長: {len(dialogue_text)})")
        tmpl = ("あなたは、日本語の小説・脚本の質を多角的に評価する、経験豊富で鋭い洞察力を持つプロの編集者AIです。\n"
                "以下の【評価対象の対話文】を慎重に読み込み、下記の【評価観点】に基づいて厳密に採点（各1.0～5.0点、0.1点刻み）し、その上で具体的なフィードバックを提供してください。\n"
                "評価の際は、表面的な体裁だけでなく、物語の深み、キャラクターのリアリティ、感情表現の巧みさ、読者への訴求力といった文学的・物語的な質を重視してください。\n\n"
                "【評価対象の対話文】\n```text\n{dialogue_to_evaluate}\n```\n\n"
                "**【採点セクション】**\n" # マーカーを太字に
                "**必ずこのマーカーの直後から採点を開始し、**以下の各評価観点について、1.0～5.0点の範囲で、0.1点刻みで厳密に採点し、**必ず下記の指定フォーマット通りに点数のみを記述してください。**\n" # 指示を強調・追加
                "フォーマット: `[項目名]: [数値スコア] 点` (例: `キャラクターの一貫性と魅力: 3.7 点`)\n"
                "**重要: 各項目の点数の後には、追加のコメントや括弧書き（例: `(詳細な理由)`など）を一切含めないでください。この【採点セクション】内では、各評価項目と点数のみを記述し、各項目は改行してください。**\n\n" # 指示を強調
                "1. **キャラクターの一貫性と魅力:** [1.0-5.0の数値] 点\n"
                "2. **対話の自然さとリズム・テンポ:** [1.0-5.0の数値] 点\n"
                "3. **感情表現と主観描写の深化:** [1.0-5.0の数値] 点\n"
                "4. **設定・制約条件の遵守と構成の巧みさ:** [1.0-5.0の数値] 点\n"
                "5. **対話の深みと複雑性（サブテキスト・伏線・テーマ性）:** [1.0-5.0の数値] 点\n"
                "6. **全体的な魅力と完成度（素材としての質を含む）:** [1.0-5.0の数値] 点\n\n"
                "**総合スコア:** [上記6項目の平均点、または編集者としての総合的な判断に基づく1.0-5.0の数値] 点\n"
                "\n"
                "**【フィードバックコメントセクション】**\n" # マーカーを太字に
                "上記採点の後、**必ずこのマーカーの直後に続けて**、良かった点（具体的にどの部分がどう良かったか）、改善した方が良いと思われる点（具体的にどの部分をどう改善すればよいか）、および具体的な提案（例：別の言い回し、描写の追加・削除、構成の変更案など）を、読解力のある編集者としての視点から詳細に記述してください。\n" # 指示を強調
                "（ここに具体的なフィードバックコメントを記述）\n\n"
                "上記指示に厳密に従って、評価結果とフィードバックを生成してください。")
        self.logger.debug("LLM評価プロンプト作成完了 (v5.4 - マーカー指示強調)。")
        return tmpl.format(dialogue_to_evaluate=dialogue_text.strip()).strip()

    def create_improvement_prompt(
        self,
        prev_dialogue: str,
        prev_evaluation_text: Union[str, Dict[str, Any]],
        feedback_context: 'FeedbackContextV49', # type: ignore
        settings: SettingsProtocol # type: ignore
    ) -> str:
        # (このメソッドはv5.2/v5.4から変更なし、extract_scores の改善が間接的に影響)
        if not all([self.SubjectiveIntensityLevel_cls, self.PsychologicalPhaseV49_cls, self.EmotionalToneV49_cls, self.LLMScoreKeys_cls, self.DFRSMetrics_cls, self.FeedbackContextV49_cls]):
            self.logger.critical("PromptBuilder初期化エラー(改善): 必須Enum/Modelクラス未ロード。改善プロンプト生成不可。")
            raise RuntimeError("PromptBuilder is not properly initialized for improvement prompt (missing essential classes).")

        loop_version = feedback_context.version + 1
        self.logger.info(f"改善プロンプト作成開始 (改善対象: v{feedback_context.version}, 生成目標: v{loop_version})")

        eval_summary_for_prompt: str
        extracted_llm_scores_str_value_key: Dict[str, float] = {}

        if isinstance(prev_evaluation_text, str) and prev_evaluation_text.strip():
            try:
                # extract_scores の改善により、ここでのスコア抽出精度向上が期待される
                extracted_llm_scores_enum_keys = PromptBuilderV49.extract_scores(prev_evaluation_text, self.LLMScoreKeys_cls) # type: ignore
                extracted_llm_scores_str_value_key = {
                    k.value: v for k, v in extracted_llm_scores_enum_keys.items() if hasattr(k, 'value') and isinstance(v, (float,int))
                }
            except Exception as e_extract_imp:
                self.logger.warning(f"改善プロンプト用の前評価スコア抽出中にエラーが発生しました: {e_extract_imp}", exc_info=True)

            feedback_comment_marker = "【フィードバックコメントセクション】"
            marker_pos = prev_evaluation_text.rfind(feedback_comment_marker)
            if marker_pos != -1:
                eval_summary_for_prompt = prev_evaluation_text[marker_pos + len(feedback_comment_marker):].strip()
                if not eval_summary_for_prompt:
                    eval_summary_for_prompt = "(具体的なフィードバックコメントはありませんでした。スコアと全体の印象から改善点を推測してください。)"
            else: # マーカーが見つからない場合のフォールバック (v5.2/v5.4のロジックを維持)
                score_section_end_marker = "```"
                score_section_end_pos = prev_evaluation_text.rfind(score_section_end_marker)
                if score_section_end_pos != -1:
                    second_marker_pos = prev_evaluation_text.rfind(score_section_end_marker, 0, score_section_end_pos)
                    if second_marker_pos != -1 :
                         # スコアセクションマーカーが複数ある場合、より精度の高いフィードバック抽出を試みる
                        potential_fb_start = prev_evaluation_text.find("```\n\n**【採点セクション】**", second_marker_pos) # このパターンは評価プロンプトに依存
                        if potential_fb_start != -1: # より具体的なマーカーの後から取得
                            fb_after_score_section_marker = prev_evaluation_text[potential_fb_start:]
                            actual_fb_start = fb_after_score_section_marker.find(feedback_comment_marker) # 再度フィードバックマーカーを探す
                            if actual_fb_start != -1:
                                eval_summary_for_prompt = fb_after_score_section_marker[actual_fb_start + len(feedback_comment_marker):].strip()
                            else: # フィードバックマーカーが見つからなければ、スコアセクションの```以降を暫定的に使用
                                eval_summary_for_prompt = prev_evaluation_text[score_section_end_pos + len(score_section_end_marker):].strip()
                        else: # スコアセクション特定できず
                            eval_summary_for_prompt = prev_evaluation_text[score_section_end_pos + len(score_section_end_marker):].strip()
                    else:
                        eval_summary_for_prompt = prev_evaluation_text[score_section_end_pos + len(score_section_end_marker):].strip()
                else:
                    eval_summary_for_prompt = prev_evaluation_text.strip()
            
            if extracted_llm_scores_str_value_key:
                score_feedback_parts = [f"{key_str}: {score_val:.1f}" for key_str, score_val in extracted_llm_scores_str_value_key.items()]
                eval_summary_for_prompt = f"前回のLLM評価スコア: {', '.join(score_feedback_parts)}\n\n{eval_summary_for_prompt}"
            else:
                eval_summary_for_prompt = f"前回のLLM評価スコア: (抽出失敗またはスコアなし - LLM応答を確認してください)\n\n{eval_summary_for_prompt}"

        elif isinstance(prev_evaluation_text, dict) and prev_evaluation_text.get("error_message"):
            eval_summary_for_prompt = f"(前回の評価処理でエラーが発生しました: Code={prev_evaluation_text.get('error_code', 'N/A')} - Message='{prev_evaluation_text.get('error_message')}')"
        else:
            eval_summary_for_prompt = "(前回の評価テキストが存在しないか、形式が不正です。一般的な改善を目指してください。)"

        additional_feedback_text = ""
        if self.feedback_strategy and hasattr(self.feedback_strategy, 'generate'):
            try:
                additional_feedback_text = self.feedback_strategy.generate(feedback_context) # type: ignore
            except Exception as e_feedback_gen:
                self.logger.error(f"FeedbackStrategyからの追加フィードバック生成中にエラー: {e_feedback_gen}", exc_info=True)
                additional_feedback_text = "(システムによる追加フィードバックの生成中にエラーが発生しました。)"

        focus_areas_prompt_str = "全体的な品質向上と、さらなる表現の洗練を目指してください。"
        current_llm_scores_from_ctx = feedback_context.llm_scores if isinstance(feedback_context.llm_scores, dict) else {}
        current_dfrs_scores_from_ctx = feedback_context.dfrs_scores if isinstance(feedback_context.dfrs_scores, dict) else {}
        combined_scores_for_focus_eval = {str(k): v for k, v in {**current_llm_scores_from_ctx, **current_dfrs_scores_from_ctx}.items()}

        score_thresholds_map_str_keys: Dict[str, float] = {}
        metric_display_names_map_str_keys: Dict[str, str] = {}
        if self.LLMScoreKeys_cls:
            for member in self.LLMScoreKeys_cls: # type: ignore
                if member != self.LLMScoreKeys_cls.UNKNOWN: # type: ignore
                    score_thresholds_map_str_keys[member.value] = 4.0
                    metric_display_names_map_str_keys[member.value] = f"LLM評価: {member.name.replace('_', ' ').title()}"
        if self.DFRSMetrics_cls: # type: ignore
            for member in self.DFRSMetrics_cls: # type: ignore
                if member != self.DFRSMetrics_cls.UNKNOWN and member != self.DFRSMetrics_cls.FINAL_EODF_V49: # type: ignore
                    score_thresholds_map_str_keys[member.value] = 3.5
                    metric_display_names_map_str_keys[member.value] = f"DFRS評価: {member.name.replace('_', ' ').title()}"
        
        low_scoring_items_list = []
        for score_key_str_val, score_num_val in combined_scores_for_focus_eval.items():
            threshold = score_thresholds_map_str_keys.get(score_key_str_val)
            if threshold is not None and isinstance(score_num_val, (int, float)) and score_num_val < threshold:
                display_name_str = metric_display_names_map_str_keys.get(score_key_str_val, score_key_str_val)
                low_scoring_items_list.append((display_name_str, score_num_val))
        
        sorted_low_scores_tuples = sorted(low_scoring_items_list, key=lambda item_tuple: item_tuple[1])
        focus_areas_names_list = [item_tuple[0] for item_tuple in sorted_low_scores_tuples[:3]]
        if focus_areas_names_list:
            focus_areas_prompt_str = "特に、以下の観点に注力して改善してください: " + ", ".join(focus_areas_names_list)

        preservation_instruction_text = ( "**【品質維持・強化のための重要原則】**\n" "1.  **前バージョンの長所を活かす:** 前回の対話文で特に評価の高かった点や、物語の核心に関わる重要な要素（キャラクターの根源的な動機、物語のテーマ性、重要な伏線など）は、可能な限り維持またはさらに強化してください。\n" "2.  **一貫性の担保:** キャラクターの性格、口調、価値観、目標、およびこれまでの行動との一貫性を常に意識してください。シーン内での感情や思考の変化は、そのキャラクターにとって自然で説得力のある動機付けに基づいている必要があります。\n" "3.  **物語全体の文脈との調和:** この対話シーンが、物語全体の大きな流れの中でどのような位置づけにあり、どのような役割を果たすべきかを常に念頭に置いてください。前後のシーンとの整合性や、伏線の効果的な配置・回収も重要です。" )
        intensity_guidance_text = ""
        if loop_version <= 2: intensity_guidance_text = "指示の強弱に注意し、特に「強く推奨」「必須」といった指示は最優先で対応してください。「～かもしれない」「～を検討」といった提案は、物語の流れやキャラクターの一貫性を損なわない範囲で柔軟に取り入れてください。"
        elif loop_version == 3: intensity_guidance_text = "これまでのフィードバックを踏まえ、より抜本的な改善も視野に入れてください。細かい修正だけでなく、シーンの構成や展開、キャラクターの行動選択肢など、より大きな視点での変更も恐れずに試みてください。ただし、常に物語とキャラクターの一貫性は保つこと。"
        else: intensity_guidance_text = "最終調整の段階です。これまでの改善で積み上げてきた品質を損なわないよう、細部の表現（言葉選び、句読点、リズムなど）の洗練に注力してください。特に、読者の感情に訴えかける描写の精度を高め、シーン全体の完成度を極限まで引き上げてください。"

        current_intended_phase_val = feedback_context.intended_phase
        current_intended_tone_val = feedback_context.intended_tone
        subj_intensity_setting_val = getattr(settings, 'subjective_intensity', self.SubjectiveIntensityLevel_cls.MEDIUM) # type: ignore
        subjective_intensity_enum_val = self._get_enum_member_from_value( self.SubjectiveIntensityLevel_cls, subj_intensity_setting_val, self.SubjectiveIntensityLevel_cls.MEDIUM )
        if not subjective_intensity_enum_val: subjective_intensity_enum_val = self.SubjectiveIntensityLevel_cls.MEDIUM # type: ignore

        char_a_data_from_state_obj = getattr(self.config, '_temp_char_a_for_improve', None) # type: ignore
        char_a_internal_style_str = str(char_a_data_from_state_obj.get('internal_monologue_style', '')) if char_a_data_from_state_obj and isinstance(char_a_data_from_state_obj, dict) else None
        char_a_perception_bias_str = str(char_a_data_from_state_obj.get('perception_filter_bias', '')) if char_a_data_from_state_obj and isinstance(char_a_data_from_state_obj, dict) else None
        char_a_emotion_method_str = str(char_a_data_from_state_obj.get('emotion_expression_method', '')) if char_a_data_from_state_obj and isinstance(char_a_data_from_state_obj, dict) else None

        subjective_instructions_for_prompt = self._build_subjective_instruction_v49( subjective_intensity_enum_val, current_intended_phase_val, current_intended_tone_val, char_internal_style=char_a_internal_style_str, char_perception_bias=char_a_perception_bias_str, char_emotion_method=char_a_emotion_method_str ) if getattr(settings, 'subjective_focus', True) else "**主観描写:** (この改善サイクルでは主観描写強化はオフです)" # type: ignore
        phase_tone_instructions_for_prompt = self._get_phase_tone_instruction_v49( current_intended_phase_val, current_intended_tone_val )
        
        style_instruction_for_prompt = ""
        style_template_name_for_improve = getattr(settings, 'style_template', None) # type: ignore
        if self.style_manager and style_template_name_for_improve:
            try:
                style_additions_text = self.style_manager.get_style_prompt_addition_text(style_template_name_for_improve)
                if style_additions_text and style_additions_text.strip():
                    style_instruction_for_prompt = f"**文体スタイル指示 ({style_template_name_for_improve}):**\n{style_additions_text.strip()}"
            except Exception as e_style_get_improve:
                self.logger.warning(f"改善プロンプト生成時のスタイル指示取得でエラー (Style: '{style_template_name_for_improve}'): {e_style_get_improve}")

        prompt_components_list: List[str] = [
            f"あなたは、以前生成した日本語の対話文（バージョン{feedback_context.version}）とその評価、および以下の改善指示に基づいて、より洗練された対話シーン（バージョン{loop_version}）を生成する、プロの小説編集AIです。",
            f"**【改善対象の対話文 (v{feedback_context.version})】**\n```text\n{prev_dialogue.strip()}\n```",
            f"**【前バージョンの評価サマリーと改善のためのフィードバック】**\n{eval_summary_for_prompt}",
            f"**【システムからの追加改善フィードバック (v{loop_version}目標)】**\n{additional_feedback_text.strip() if additional_feedback_text and additional_feedback_text.strip() else '(システムからの追加フィードバックはありません。上記評価と指示に基づいて改善してください。)'}",
            f"**【今回の改善における主な焦点】**\n{focus_areas_prompt_str}",
            f"**【目標とする心理的位相 (v{loop_version})】**\n - 現在目指すべき心理的位相は「{getattr(current_intended_phase_val, 'value', '指定なし')}」です。この位相が持つ特徴（例：緊張感、内省、行動の活発化など）を意識し、それがシーン全体に反映されるようにしてください。",
            f"**【目標とする感情トーン (v{loop_version})】**\n - 現在目指すべき感情トーンは「{getattr(current_intended_tone_val, 'value', '指定なし')}」です。この感情がキャラクターの言動や描写の細部に一貫して現れるように、表現のニュアンスを調整してください。",
            (phase_tone_instructions_for_prompt.strip() if phase_tone_instructions_for_prompt else ""),
            (subjective_instructions_for_prompt.strip() if subjective_instructions_for_prompt else ""),
            (style_instruction_for_prompt.strip() if style_instruction_for_prompt else ""),
            preservation_instruction_text,
            f"**【ループ回数に応じた指示 (現在v{loop_version})】**\n{intensity_guidance_text}",
            "**【具体的な改善・執筆指示】**\n1. 上記の全てのフィードバックと指示を総合的に考慮し、前回の対話文を全面的に書き直すか、部分的に修正・加筆してください。\n2. 特に指摘された問題点を解消し、提案された改善点を積極的に取り入れてください。\n3. 目標文字数は、前回と大きく逸脱しない範囲で、内容の充実度に応じて調整してください（目安：前回の文字数の0.9～1.2倍程度）。しかし、質の向上を最優先とし、文字数調整のために不自然な短縮や冗長な引き伸ばしは避けてください。",
            "**【出力形式】**\n改善された対話シーンの日本語テキストのみを出力してください。前置きや後書き、メタコメント（例：「承知しました」）、ハッシュタグ、その他の無関係な情報は一切含めないでください。"
        ]
        final_improvement_prompt_str = "\n\n".join(filter(None, [component.strip() for component in prompt_components_list])).strip()
        self.logger.info(f"改善プロンプト作成完了 (v{loop_version}目標, 総文字数: {len(final_improvement_prompt_str)})")
        return final_improvement_prompt_str

    @staticmethod
    def extract_scores(evaluation_text: str, llm_score_keys_enum: Type['ScoreKeysLLMEnumType_hint']) -> Dict['ScoreKeysLLMEnumType_hint', float]: # type: ignore
        """
        LLMからの評価テキストを解析し、指定されたスコアキーに基づいてスコアを抽出します。(v5.2ベースに強化)
        """
        logger_extract = logging.getLogger(f"{PromptBuilderV49.__module__}.PromptBuilderV49.extract_scores.v5.5") # バージョン更新
        extracted_scores_map_result: Dict[PromptBuilderV49.ScoreKeysLLMEnumType_hint, float] = {} # type: ignore

        if not llm_score_keys_enum or not isinstance(llm_score_keys_enum, enum.EnumMeta):
            logger_extract.critical(f"extract_scores: 有効なLLMスコアキーEnumクラス ({llm_score_keys_enum}) が提供されていません。スコア抽出はできません。")
            return {}

        # マッピング: 日本語項目名 -> Enumメンバー (v5.2と同様)
        metric_name_jp_to_enum_member_map: Dict[str, Optional[PromptBuilderV49.ScoreKeysLLMEnumType_hint]] = { # type: ignore
            "キャラクターの一貫性と魅力": getattr(llm_score_keys_enum, 'CONSISTENCY', None),
            "対話の自然さとリズム・テンポ": getattr(llm_score_keys_enum, 'NATURALNESS', None),
            "感情表現と主観描写の深化": getattr(llm_score_keys_enum, 'EMOTIONAL_DEPTH', None),
            "設定・制約条件の遵守と構成の巧みさ": getattr(llm_score_keys_enum, 'CONSTRAINTS', None),
            "対話の深みと複雑性（サブテキスト・伏線・テーマ性）": getattr(llm_score_keys_enum, 'COMPLEXITY', None),
            "全体的な魅力と完成度（素材としての質を含む）": getattr(llm_score_keys_enum, 'ATTRACTIVENESS', None),
            "総合スコア": getattr(llm_score_keys_enum, 'OVERALL', None),
        }
        valid_metric_name_to_enum_map: Dict[str, PromptBuilderV49.ScoreKeysLLMEnumType_hint] = { # type: ignore
            jp_name: enum_member for jp_name, enum_member in metric_name_jp_to_enum_member_map.items() if enum_member is not None
        }
        if not valid_metric_name_to_enum_map:
            logger_extract.error("extract_scores: 有効な日本語メトリック名とScoreKeys.LLM Enumメンバーのマッピングが構築できませんでした。")
            return {}

        # マーカーベースの検索範囲特定 (v5.2の改善ロジック + ログ強化)
        score_section_start_marker = "【採点セクション】"
        feedback_section_start_marker = "【フィードバックコメントセクション】"
        text_to_search_for_scores = evaluation_text
        
        score_section_start_index = evaluation_text.find(score_section_start_marker)
        
        if score_section_start_index != -1:
            search_start_offset = score_section_start_index + len(score_section_start_marker)
            # フィードバックマーカーが採点マーカーより後にある場合のみ、検索終了位置として使用
            feedback_section_start_index_after_score = evaluation_text.find(feedback_section_start_marker, search_start_offset)
            search_end_offset = feedback_section_start_index_after_score if feedback_section_start_index_after_score != -1 else len(evaluation_text)
            text_to_search_for_scores = evaluation_text[search_start_offset:search_end_offset]
            logger_extract.info(f"スコア抽出対象範囲をマーカー '{score_section_start_marker}' (位置: {search_start_offset}) から "
                                 f"'{feedback_section_start_marker}' (またはテキスト終端、位置: {search_end_offset}) の間で特定しました。"
                                 f"対象テキストプレビュー(100字): '{text_to_search_for_scores[:100].replace(os.linesep, ' ')}...'")
        else:
            logger_extract.warning(f"評価テキスト内にスコア開始マーカー '{score_section_start_marker}' が見つかりません。"
                                   f"評価テキスト全体 (長さ: {len(evaluation_text)}文字) を検索対象とします。プレビュー(100字): '{evaluation_text[:100].replace(os.linesep, ' ')}...'"
                                   "抽出精度が低下する可能性があります。LLMの応答が評価プロンプトの指示に従っているか確認してください。")
        
        processed_enum_members_set: Set[PromptBuilderV49.ScoreKeysLLMEnumType_hint] = set() # type: ignore

        # スコア抽出のための正規表現パターンリスト (v5.2ベースで強化)
        # 1. 基本形: "[項目名]: X.Y 点" または "項目名: X.Y 点" (行頭、項目名はキャプチャ)
        # 2. 項目名とEnumの.valueを直接比較するケースも考慮（英語キーなど）
        score_line_patterns = [
            # パターンA: 日本語項目名ベース (角括弧あり/なし、コロン全角/半角、数値全角/半角、"点"の有無)
            re.compile(
                r"^\s*(?:[\d①-⑦]+\.\s*|\*\*\s*)?" # 先頭の番号や**を許容
                r"(?P<item_name_jp>(?:\[.+?\]|[^:：\s]+(?:\s+[^:：\s]+)*))\s*[:：]\s*" # 日本語項目名
                r"(?P<score_value>[\d０-９]+\.?[\d０-９]*)(?:\s*点)?",
                re.MULTILINE
            ),
            # パターンB: Enumのvalue値 (英語キー等) ベース (上記と同様の柔軟性)
            re.compile(
                r"^\s*(?P<item_name_enum_val>[a-zA-Z_]+)\s*[:：]\s*" # Enumの.value (英字とアンダースコア)
                r"(?P<score_value>[\d０-９]+\.?[\d０-９]*)(?:\s*点)?",
                re.MULTILINE | re.IGNORECASE # Enum valueは大文字小文字を区別しないことが多い
            )
        ]

        for line_num, line_content in enumerate(text_to_search_for_scores.splitlines()):
            line_content_stripped = line_content.strip()
            if not line_content_stripped: continue

            match_found_in_line = False
            for pattern_idx, regex_pattern_compiled in enumerate(score_line_patterns):
                match_object = regex_pattern_compiled.match(line_content_stripped)
                if match_object:
                    logger_extract.debug(f"行 {line_num+1} (パターン {pattern_idx+1}) マッチ: '{line_content_stripped}'")
                    
                    item_name_candidate_str: Optional[str] = None
                    if "item_name_jp" in match_object.groupdict() and match_object.group("item_name_jp"):
                        item_name_candidate_str = match_object.group("item_name_jp").strip("[]").strip()
                    elif "item_name_enum_val" in match_object.groupdict() and match_object.group("item_name_enum_val"):
                        item_name_candidate_str = match_object.group("item_name_enum_val").strip()
                    
                    raw_score_value_str = match_object.group("score_value")

                    if item_name_candidate_str and raw_score_value_str:
                        normalized_score_value_str = raw_score_value_str.translate(str.maketrans('．０１２３４５６７８９', '.' + '0123456789'))
                        logger_extract.debug(f"  抽出試行: 項目名候補='{item_name_candidate_str}', スコア文字列='{raw_score_value_str}' (正規化後='{normalized_score_value_str}')")
                        
                        # Enumキーへのマッピング (日本語名 -> Enum、または Enum.value -> Enum)
                        target_enum_key_resolved: Optional[PromptBuilderV49.ScoreKeysLLMEnumType_hint] = None # type: ignore
                        if pattern_idx == 0: # パターンA (日本語項目名)
                            target_enum_key_resolved = valid_metric_name_to_enum_map.get(item_name_candidate_str)
                        elif pattern_idx == 1: # パターンB (Enum value)
                            for enum_member_iter in llm_score_keys_enum: # type: ignore
                                if hasattr(enum_member_iter, 'value') and enum_member_iter.value.lower() == item_name_candidate_str.lower():
                                    target_enum_key_resolved = enum_member_iter # type: ignore
                                    logger_extract.debug(f"  項目名候補'{item_name_candidate_str}'をEnumメンバー'{target_enum_key_resolved.name}' (.value一致) にマップ。") # type: ignore
                                    break
                        
                        if target_enum_key_resolved and target_enum_key_resolved not in processed_enum_members_set:
                            try:
                                score_as_float_val = float(normalized_score_value_str)
                                clamped_score_val = round(max(0.0, min(5.0, score_as_float_val)), 2)
                                extracted_scores_map_result[target_enum_key_resolved] = clamped_score_val
                                processed_enum_members_set.add(target_enum_key_resolved)
                                logger_extract.info(f"  スコア抽出成功: {target_enum_key_resolved.name if hasattr(target_enum_key_resolved, 'name') else 'N/A'} = {clamped_score_val} (元テキスト: '{raw_score_value_str}', 項目名: '{item_name_candidate_str}')")
                                match_found_in_line = True
                                break # この行については処理完了、次の行へ
                            except ValueError:
                                logger_extract.warning(f"  スコア文字列 '{normalized_score_value_str}' (元: '{raw_score_value_str}') をfloatに変換できませんでした ({item_name_candidate_str})。")
                        elif target_enum_key_resolved in processed_enum_members_set:
                            logger_extract.debug(f"  項目 '{item_name_candidate_str}' のスコアは既に処理済みです。スキップします。")
                        elif item_name_candidate_str : # マップできなかった場合
                            logger_extract.debug(f"  項目名候補 '{item_name_candidate_str}' は既知のメトリックにマップできませんでした。")
                    else:
                        logger_extract.debug(f"  行 '{line_content_stripped}' から項目名またはスコア文字列を正しく抽出できませんでした。")
                    if match_found_in_line: break # この行に対して最初のパターンがマッチしたらOK
        
        # 総合スコアの補完ロジック (v5.2と同様)
        overall_score_enum_member_ref = getattr(llm_score_keys_enum, 'OVERALL', None) # type: ignore
        if overall_score_enum_member_ref and overall_score_enum_member_ref not in extracted_scores_map_result and extracted_scores_map_result:
            scores_for_averaging_list = [
                s_val for enum_m_avg, s_val in extracted_scores_map_result.items()
                if enum_m_avg != overall_score_enum_member_ref and isinstance(s_val, (float, int))
            ]
            min_scores_needed_for_avg_calc = max(1, (len(valid_metric_name_to_enum_map) - 1) // 2) # 総合除く項目の半分以上
            
            if scores_for_averaging_list and len(scores_for_averaging_list) >= min_scores_needed_for_avg_calc:
                try:
                    calculated_average_score_val = round(statistics.mean(scores_for_averaging_list), 2)
                    extracted_scores_map_result[overall_score_enum_member_ref] = calculated_average_score_val # type: ignore
                    logger_extract.info(f"総合スコア (OVERALL) が直接抽出されなかったため、他の有効なスコア ({len(scores_for_averaging_list)}件) の平均値 ({calculated_average_score_val}) で補完しました。")
                except statistics.StatisticsError:
                    logger_extract.warning("総合スコアのフォールバック平均計算中に StatisticsError が発生しました（有効スコア不足の可能性）。")
                except Exception as e_avg_calc_err:
                    logger_extract.error(f"総合スコアのフォールバック平均計算中に予期せぬエラーが発生しました: {e_avg_calc_err}", exc_info=True)
            else:
                logger_extract.warning(f"総合スコア (OVERALL) が直接抽出されず、他の有効スコアも少ないため（{len(scores_for_averaging_list)}件、必要最低{min_scores_needed_for_avg_calc}件）、平均計算による補完は行いませんでした。")
        elif overall_score_enum_member_ref and overall_score_enum_member_ref not in extracted_scores_map_result:
             logger_extract.warning(f"総合スコア ({getattr(overall_score_enum_member_ref,'value','overall')}) が抽出できず、他のスコアも存在しないため補完もできませんでした。")
        
        if not extracted_scores_map_result:
            logger_extract.warning(
                f"評価テキストから有効なLLMスコアを一つも抽出できませんでした。(v5.5)\n"
                f"検索対象となったテキストのプレビュー (最初の500文字):\n'''\n{text_to_search_for_scores[:500].replace(chr(10),os.linesep)}\n'''"
            )
        else:
            logger_extract.info(f"スコア抽出処理完了 (v5.5)。抽出された有効スコア数: {len(extracted_scores_map_result)} / {len(valid_metric_name_to_enum_map)}.")
            for enum_member_key_final, score_value_num_final in extracted_scores_map_result.items():
                logger_extract.debug(f"  最終抽出スコア: {getattr(enum_member_key_final, 'name', 'UNKNOWN_ENUM_KEY')} ({getattr(enum_member_key_final, 'value', 'N/A')}) = {score_value_num_final}")
             
        return extracted_scores_map_result

# =============================================================================
# -- Part 5 終了点 (PromptBuilderV49 クラス定義終了)
# =============================================================================
# =============================================================================
# -- Part 6: Subjectivity & Fluctuation Scorer (v4.9α - 再検証・修正版)
# =============================================================================
# Part 6: 主観性・揺らぎスコアラー (再検証・修正版)
# 過去の実行確認済みバージョンをベースに、データロードとキーワードマッチング処理の
# 信頼性向上を目指した修正。ログ出力を強化。

from typing import TYPE_CHECKING, TypeVar, Set, List, Dict, Optional, Tuple, Union, Any, Type, Callable
from collections import defaultdict
import enum
import logging
import random
import re
import math

# --- グローバルスコープで利用可能であることを期待する変数 (Part 0 などで定義済み) ---
# _get_global_type は Part 0 で定義されている想定
_get_global_type_func: Optional[Callable[[str, Optional[type]], Optional[Type[Any]]]] = globals().get('_get_global_type')

if TYPE_CHECKING:
    # --- プロジェクト内部型 (Part 0, 1b, 2, 3, 4a で定義・インポート済みと仮定) ---
    # config: AppConfigV49 (ConfigProtocolを実装)
    # scorer: ScorerProtocol (このクラスが実装)
    # enums: SubjectivityCategoryV49, FluctuationCategoryV49
    # models: SubjectivityKeywordEntryV49, FluctuationPatternEntryV49
    from __main__ import ( # type: ignore[attr-defined]
        ConfigProtocol, ScorerProtocol,
        SubjectivityCategoryV49, FluctuationCategoryV49,
        SubjectivityKeywordEntryV49, FluctuationPatternEntryV49,
        AppConfigV49 # AppConfigV49 も ConfigProtocol を実装していると仮定
    )
    # 型エイリアス
    ConfigProto = ConfigProtocol
    SubjectivityCategoryEnumType = SubjectivityCategoryV49
    FluctuationCategoryEnumType = FluctuationCategoryV49
    CategoryEnumType = Union[SubjectivityCategoryEnumType, FluctuationCategoryEnumType]
    KeywordEntryType = SubjectivityKeywordEntryV49
    PatternEntryType = FluctuationPatternEntryV49
    EntryType = Union[KeywordEntryType, PatternEntryType]
else:
    # 実行時は文字列リテラル
    ConfigProto = 'ConfigProtocol'
    SubjectivityCategoryEnumType = 'SubjectivityCategoryV49'
    FluctuationCategoryEnumType = 'FluctuationCategoryV49'
    CategoryEnumType = Union['SubjectivityCategoryV49', 'FluctuationCategoryV49'] # 実行時用
    KeywordEntryType = 'SubjectivityKeywordEntryV49'
    PatternEntryType = 'FluctuationPatternEntryV49'
    EntryType = Union['SubjectivityKeywordEntryV49', 'FluctuationPatternEntryV49'] # 実行時用

class SubjectivityFluctuationScorerV49: # Implicitly implements ScorerProtocol
    """主観性スコアと揺らぎ強度スコアを計算 (v4.9α - 再検証・修正版)"""

    def __init__(self, config: ConfigProto): # type: ignore
        """Scorerを初期化"""
        self.config: AppConfigV49 = config # AppConfigV49 のインスタンスを期待
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        self.logger.info(f"SubjectivityFluctuationScorerV49 (System: {getattr(self.config, 'SYSTEM_VERSION', 'N/A')}) の初期化を開始します...")

        # --- 必須EnumクラスとPydanticモデルクラスのロード ---
        self.SubjectivityCategoryV49_cls: Optional[Type[SubjectivityCategoryEnumType]] = None # type: ignore
        self.FluctuationCategoryV49_cls: Optional[Type[FluctuationCategoryEnumType]] = None # type: ignore
        self.SubjectivityKeywordEntry_cls: Optional[Type[KeywordEntryType]] = None # type: ignore
        self.FluctuationPatternEntry_cls: Optional[Type[PatternEntryType]] = None # type: ignore
        
        load_success = True
        if _get_global_type_func:
            self.SubjectivityCategoryV49_cls = _get_global_type_func('SubjectivityCategoryV49', enum.EnumMeta) # type: ignore
            self.FluctuationCategoryV49_cls = _get_global_type_func('FluctuationCategoryV49', enum.EnumMeta) # type: ignore
            self.SubjectivityKeywordEntry_cls = _get_global_type_func('SubjectivityKeywordEntryV49') # type: ignore # Pydanticモデルはtype
            self.FluctuationPatternEntry_cls = _get_global_type_func('FluctuationPatternEntryV49') # type: ignore # Pydanticモデルはtype
        else:
            self.logger.error("CRITICAL: _get_global_type関数が見つかりません。クラスロード不可。")
            load_success = False

        if not (self.SubjectivityCategoryV49_cls and self.FluctuationCategoryV49_cls and
                self.SubjectivityKeywordEntry_cls and self.FluctuationPatternEntry_cls):
            missing_cls_names: List[str] = []
            if not self.SubjectivityCategoryV49_cls: missing_cls_names.append("SubjectivityCategoryV49")
            if not self.FluctuationCategoryV49_cls: missing_cls_names.append("FluctuationCategoryV49")
            if not self.SubjectivityKeywordEntry_cls: missing_cls_names.append("SubjectivityKeywordEntryV49")
            if not self.FluctuationPatternEntry_cls: missing_cls_names.append("FluctuationPatternEntryV49")
            self.logger.critical(f"CRITICAL: Scorer初期化に必要なEnum/Modelクラス({', '.join(missing_cls_names)})のロードに失敗しました。スコアリング機能は制限されます。")
            load_success = False

        # --- キーワード/パターンデータのロード ---
        # AppConfigV49 の subjectivity_data/fluctuation_data は Pydantic モデル (RootModel) インスタンスを期待
        # その .root 属性が実際のデータ辞書 (Dict[Enum, List[EntryModel]])
        
        self.subjectivity_keywords_data: Dict[SubjectivityCategoryEnumType, List[KeywordEntryType]] = {} # type: ignore
        self.fluctuation_patterns_data: Dict[FluctuationCategoryEnumType, List[PatternEntryType]] = {} # type: ignore

        if load_success: # 必須クラスがロードできた場合のみデータロード試行
            subjectivity_file_model_instance = getattr(self.config, 'subjectivity_data', None)
            if subjectivity_file_model_instance and hasattr(subjectivity_file_model_instance, 'root') and \
               isinstance(subjectivity_file_model_instance.root, dict):
                self.subjectivity_keywords_data = subjectivity_file_model_instance.root
                self.logger.debug(f"主観性キーワードデータを AppConfig.subjectivity_data.root からロードしました。"
                                  f"カテゴリ数: {len(self.subjectivity_keywords_data)}")
            else:
                self.logger.error(
                    f"AppConfigから主観性キーワードデータ(辞書型)を取得できませんでした。"
                    f" (subjectivity_data型: {type(subjectivity_file_model_instance)}, "
                    f" .root型: {type(getattr(subjectivity_file_model_instance, 'root', None)) if subjectivity_file_model_instance else 'N/A'})."
                    f" 空の辞書を使用します。"
                )

            fluctuation_file_model_instance = getattr(self.config, 'fluctuation_data', None)
            if fluctuation_file_model_instance and hasattr(fluctuation_file_model_instance, 'root') and \
               isinstance(fluctuation_file_model_instance.root, dict):
                self.fluctuation_patterns_data = fluctuation_file_model_instance.root
                self.logger.debug(f"揺らぎパターンデータを AppConfig.fluctuation_data.root からロードしました。"
                                  f"カテゴリ数: {len(self.fluctuation_patterns_data)}")
            else:
                self.logger.error(
                    f"AppConfigから揺らぎパターンデータ(辞書型)を取得できませんでした。"
                    f" (fluctuation_data型: {type(fluctuation_file_model_instance)}, "
                    f" .root型: {type(getattr(fluctuation_file_model_instance, 'root', None)) if fluctuation_file_model_instance else 'N/A'})."
                    f" 空の辞書を使用します。"
                )
        else: # クラスロード失敗時はデータロードもスキップ
            self.logger.error("必須クラスのロードに失敗したため、キーワード/パターンデータのロードをスキップします。")


        # --- Feature flags の取得 ---
        feature_flags_config = getattr(self.config, 'feature_flags', None)
        self.normalize_subj_by_length: bool = getattr(feature_flags_config, 'normalize_subj_by_length', True) if feature_flags_config else True
        self.normalize_fluc_by_length: bool = getattr(feature_flags_config, 'normalize_fluc_by_length', True) if feature_flags_config else True
        self.enable_subj_synonyms: bool = getattr(feature_flags_config, 'enable_subj_synonyms', True) if feature_flags_config else True
        self.enable_fluc_synonyms: bool = getattr(feature_flags_config, 'enable_fluc_synonyms', False) if feature_flags_config else False
        
        self._compiled_patterns: Dict[CategoryEnumType, List[Tuple[re.Pattern, float]]] = self._compile_all_patterns() # type: ignore
        
        subj_entry_count = sum(len(entries) for entries in self.subjectivity_keywords_data.values())
        fluc_entry_count = sum(len(entries) for entries in self.fluctuation_patterns_data.values())
        self.logger.info(f"SubjectivityFluctuationScorerV49 初期化完了。主観性エントリ総数: {subj_entry_count}, 揺らぎエントリ総数: {fluc_entry_count}")
        
    def _compile_all_patterns(self) -> Dict[CategoryEnumType, List[Tuple[re.Pattern, float]]]: # type: ignore
        """use_regex=True のパターンを事前コンパイル"""
        compiled_regex_map: Dict[CategoryEnumType, List[Tuple[re.Pattern, float]]] = defaultdict(list) # type: ignore
        all_data_sources_map = {
            "subjectivity": (self.subjectivity_keywords_data, self.SubjectivityKeywordEntry_cls, self.SubjectivityCategoryV49_cls),
            "fluctuation": (self.fluctuation_patterns_data, self.FluctuationPatternEntry_cls, self.FluctuationCategoryV49_cls)
        }

        for source_name, (data_map, entry_model_cls, category_enum_cls) in all_data_sources_map.items():
            if not (entry_model_cls and category_enum_cls): # 関連クラスがロードされていなければスキップ
                self.logger.warning(f"_compile_all_patterns: ソース '{source_name}' の必須クラス未ロード。コンパイルスキップ。")
                continue
            if not isinstance(data_map, dict):
                self.logger.warning(f"_compile_all_patterns: '{source_name}' のデータが辞書ではありません (型: {type(data_map)})。スキップします。")
                continue

            for category_enum_instance, entries_list_for_cat in data_map.items():
                if not isinstance(category_enum_instance, category_enum_cls):
                    self.logger.warning(f"_compile_all_patterns ({source_name}): 不正なカテゴリキー型 {type(category_enum_instance)} ('{getattr(category_enum_instance, 'value', category_enum_instance)}'). スキップ。")
                    continue
                if not isinstance(entries_list_for_cat, list):
                    self.logger.warning(f"_compile_all_patterns ({source_name}): カテゴリ '{getattr(category_enum_instance, 'value', '?')}' のエントリがリストではありません。スキップ。")
                    continue

                for entry_item in entries_list_for_cat:
                    if not isinstance(entry_item, entry_model_cls):
                        self.logger.warning(f"_compile_all_patterns ({source_name}): カテゴリ '{getattr(category_enum_instance, 'value', '?')}' 内の予期せぬエントリ型 {type(entry_item)} (期待: {entry_model_cls.__name__})。スキップ。")
                        continue
                    
                    pattern_str_from_entry = getattr(entry_item, 'pattern', None)
                    use_regex_from_entry = getattr(entry_item, 'use_regex', False)
                    intensity_from_entry = float(getattr(entry_item, 'intensity', 0.5))

                    if use_regex_from_entry and isinstance(pattern_str_from_entry, str) and pattern_str_from_entry:
                        try:
                            compiled_regex = re.compile(pattern_str_from_entry, re.IGNORECASE | re.UNICODE)
                            compiled_regex_map[category_enum_instance].append((compiled_regex, intensity_from_entry))
                        except re.error as e_re_compile_patterns:
                            self.logger.error(f"正規表現コンパイルエラー ({getattr(category_enum_instance, 'value', '?')}:{pattern_str_from_entry}): {e_re_compile_patterns}")
                        except Exception as e_compile_patterns_generic:
                             self.logger.error(f"正規表現コンパイル中に予期せぬエラー ({getattr(category_enum_instance, 'value', '?')}:{pattern_str_from_entry}): {e_compile_patterns_generic}", exc_info=True)
        return dict(compiled_regex_map)

    def _calculate_weighted_hits(
        self, text_content_input: str,
        structured_keyword_data: Dict[CategoryEnumType, List[EntryType]], # type: ignore
        normalize_by_length_flag: bool,
        enable_synonyms_flag: bool
    ) -> Tuple[float, int, Dict[CategoryEnumType, int]]: # type: ignore
        """テキスト内でキーワード/パターンの一致を重み付けしてカウント"""
        accumulated_score: float = 0.0
        total_hit_count: int = 0
        category_hit_counts: Dict[CategoryEnumType, int] = defaultdict(int) # type: ignore
        
        text_content_lower = text_content_input.lower()
        text_length_val = max(len(text_content_input), 1)

        if not isinstance(structured_keyword_data, dict):
            self.logger.warning(f"_calculate_weighted_hits: structured_keyword_dataが辞書ではありません (型: {type(structured_keyword_data)})。0スコアを返します。")
            return 0.0, 0, dict(category_hit_counts)

        valid_category_enum_classes = tuple(cls for cls in [self.SubjectivityCategoryV49_cls, self.FluctuationCategoryV49_cls] if cls is not None)
        if not valid_category_enum_classes:
            self.logger.error("_calculate_weighted_hits: カテゴリEnumクラスがロードされていません。処理を中断します。")
            return 0.0, 0, dict(category_hit_counts)

        for category_enum_key, entries_list_in_cat in structured_keyword_data.items():
            if not isinstance(category_enum_key, valid_category_enum_classes): # type: ignore
                self.logger.warning(f"_calculate_weighted_hits: 不正なカテゴリキー型 {type(category_enum_key)} ('{getattr(category_enum_key, 'value', category_enum_key)}'). スキップ。")
                continue
            if not isinstance(entries_list_in_cat, list):
                self.logger.warning(f"_calculate_weighted_hits: カテゴリ '{getattr(category_enum_key, 'value', '?')}' のエントリがリストではありません。スキップ。")
                continue

            compiled_patterns_for_current_cat = self._compiled_patterns.get(category_enum_key, [])

            for entry_model_instance in entries_list_in_cat:
                # entry_model_instance は SubjectivityKeywordEntryV49 または FluctuationPatternEntryV49 のインスタンスを期待
                # 関連するPydanticモデルクラスがロードされているか確認
                entry_expected_model_cls = None
                if self.SubjectivityCategoryV49_cls and isinstance(category_enum_key, self.SubjectivityCategoryV49_cls):
                    entry_expected_model_cls = self.SubjectivityKeywordEntry_cls
                elif self.FluctuationCategoryV49_cls and isinstance(category_enum_key, self.FluctuationCategoryV49_cls):
                    entry_expected_model_cls = self.FluctuationPatternEntry_cls
                
                if not entry_expected_model_cls: # 期待されるモデルクラスが不明な場合
                    self.logger.warning(f"_calculate_weighted_hits: カテゴリ '{getattr(category_enum_key, 'value', '?')}' に対応するエントリモデルクラスが不明。スキップ。")
                    continue
                if not isinstance(entry_model_instance, entry_expected_model_cls):
                    self.logger.warning(f"_calculate_weighted_hits: カテゴリ '{getattr(category_enum_key, 'value', '?')}' 内の予期せぬエントリ型 {type(entry_model_instance)} (期待: {entry_expected_model_cls.__name__})。スキップ。")
                    continue
                
                try:
                    keyword_from_entry = getattr(entry_model_instance, 'keyword', None)
                    pattern_from_entry = getattr(entry_model_instance, 'pattern', None)
                    use_regex_from_entry_flag = getattr(entry_model_instance, 'use_regex', False)
                    intensity_from_entry_val = float(getattr(entry_model_instance, 'intensity', 0.5))
                    synonyms_from_entry_list = getattr(entry_model_instance, 'synonyms', []) if enable_synonyms_flag else []
                    
                    targets_for_matching: List[Tuple[Union[str, re.Pattern], bool]] = [] # (target_string_or_regex, is_regex_flag)
                    processed_target_strings_set: Set[str] = set() # 重複処理を避けるため

                    # 1. 正規表現パターンの処理
                    if use_regex_from_entry_flag and isinstance(pattern_from_entry, str) and pattern_from_entry:
                        compiled_regex_found_flag = False
                        for pre_compiled_regex, _ in compiled_patterns_for_current_cat:
                            if pre_compiled_regex.pattern == pattern_from_entry:
                                if pre_compiled_regex.pattern not in processed_target_strings_set:
                                    targets_for_matching.append((pre_compiled_regex, True))
                                    processed_target_strings_set.add(pre_compiled_regex.pattern)
                                compiled_regex_found_flag = True; break
                        if not compiled_regex_found_flag: # 事前コンパイルで見つからなければ動的にコンパイル
                            try:
                                dynamic_compiled_regex_obj = re.compile(pattern_from_entry, re.IGNORECASE | re.UNICODE)
                                if dynamic_compiled_regex_obj.pattern not in processed_target_strings_set:
                                    targets_for_matching.append((dynamic_compiled_regex_obj, True))
                                    processed_target_strings_set.add(dynamic_compiled_regex_obj.pattern)
                            except re.error as e_re_dynamic_compile:
                                self.logger.error(f"動的正規表現コンパイル失敗 ({getattr(category_enum_key, 'value', '?')}:{pattern_from_entry}): {e_re_dynamic_compile}")
                    
                    # 2. キーワード文字列と非正規表現パターン文字列の処理
                    keywords_to_check_direct: List[str] = []
                    if isinstance(keyword_from_entry, str) and keyword_from_entry:
                        keywords_to_check_direct.append(keyword_from_entry)
                    if not use_regex_from_entry_flag and isinstance(pattern_from_entry, str) and pattern_from_entry:
                        keywords_to_check_direct.append(pattern_from_entry) # 正規表現でないパターンはキーワードとして扱う
                    
                    if isinstance(synonyms_from_entry_list, list):
                        keywords_to_check_direct.extend(s for s in synonyms_from_entry_list if isinstance(s, str) and s)

                    for kw_str_to_check in keywords_to_check_direct:
                        kw_lower_to_check = kw_str_to_check.lower()
                        if kw_lower_to_check not in processed_target_strings_set:
                            targets_for_matching.append((kw_lower_to_check, False)) # is_regex = False
                            processed_target_strings_set.add(kw_lower_to_check)
                    
                    # 3. マッチング実行とヒットカウント
                    hits_for_current_entry = 0
                    for target_object_match, is_regex_target_match in targets_for_matching:
                        try:
                            if is_regex_target_match and isinstance(target_object_match, re.Pattern):
                                matches_found_regex = target_object_match.findall(text_content_input) # text_content_input を使用
                                hits_for_current_entry += len(matches_found_regex)
                                if matches_found_regex: self.logger.debug(f"  Regex '{target_object_match.pattern}' found {len(matches_found_regex)} times in category '{getattr(category_enum_key,'value','?')}'")
                            elif not is_regex_target_match and isinstance(target_object_match, str) and target_object_match:
                                # 単純な部分文字列カウント (日本語の単語境界は考慮しない)
                                count_str_match = text_content_lower.count(target_object_match)
                                hits_for_current_entry += count_str_match
                                if count_str_match > 0: self.logger.debug(f"  Keyword '{target_object_match}' found {count_str_match} times (count method) in category '{getattr(category_enum_key,'value','?')}'")
                        except Exception as e_match_target:
                            self.logger.warning(f"キーワード/パターンマッチング中にエラー ({getattr(category_enum_key, 'value', '?')}: '{target_object_match}'): {e_match_target}")
                    
                    if hits_for_current_entry > 0:
                        accumulated_score += hits_for_current_entry * intensity_from_entry_val
                        total_hit_count += hits_for_current_entry
                        category_hit_counts[category_enum_key] += hits_for_current_entry
                        self.logger.debug(f"  Entry (kw='{keyword_from_entry}', pat='{pattern_from_entry}') contributed {hits_for_current_entry} hits with intensity {intensity_from_entry_val:.2f}")

                except Exception as e_proc_entry:
                    self.logger.error(f"スコアリングエントリ処理中に予期せぬエラー ({getattr(category_enum_key, 'value', '?')}): {e_proc_entry}", exc_info=True)

        # スコア正規化
        scale_factor_norm = 500.0
        max_score_guess_norm = 20.0
        
        normalized_score_val: float
        if normalize_by_length_flag:
            normalized_score_val = (accumulated_score / text_length_val) * scale_factor_norm
        else:
            normalized_score_val = (accumulated_score / max_score_guess_norm) if max_score_guess_norm > 0 else 0.0
            
        final_clipped_score_val = round(min(1.0, max(0.0, normalized_score_val)), 4)
        
        valid_category_hit_counts_map: Dict[CategoryEnumType, int] = { # type: ignore
            k_map: v_map for k_map, v_map in category_hit_counts.items() if isinstance(k_map, valid_category_enum_classes) # type: ignore
        }
        return final_clipped_score_val, total_hit_count, valid_category_hit_counts_map

    def calculate_subjectivity_score(self, text: str) -> Tuple[float, Dict[SubjectivityCategoryEnumType, int]]: # type: ignore
        """主観性スコア(0-1)とカテゴリ別ヒット数"""
        if not text or not text.strip():
            return 0.0, {}
        if not self.subjectivity_keywords_data:
            self.logger.warning("calculate_subjectivity_score: 主観性キーワードデータが空です。0スコアを返します。")
            return 0.0, {}
            
        final_score_val, total_hits_val, cat_hits_result_map = self._calculate_weighted_hits(
            text,
            self.subjectivity_keywords_data, # type: ignore
            self.normalize_subj_by_length,
            self.enable_subj_synonyms
        )
        self.logger.debug(
            f"主観性スコア計算結果: Hits={total_hits_val}, Score(0-1)={final_score_val:.3f}, "
            f"CategoryHits={ {getattr(k,'value','?'):v for k,v in cat_hits_result_map.items()} }"
        )
        
        SubjCat_cls_local_score = self.SubjectivityCategoryV49_cls
        if SubjCat_cls_local_score:
            return final_score_val, {k: v for k, v in cat_hits_result_map.items() if isinstance(k, SubjCat_cls_local_score)}
        return final_score_val, {}

    def calculate_fluctuation_intensity(self, text: str) -> Tuple[float, Dict[FluctuationCategoryEnumType, int]]: # type: ignore
        """揺らぎ強度スコア(0-1)とカテゴリ別ヒット数"""
        if not text or not text.strip():
            return 0.0, {}
        if not self.fluctuation_patterns_data:
            self.logger.warning("calculate_fluctuation_intensity: 揺らぎパターンデータが空です。0スコアを返します。")
            return 0.0, {}

        final_score_val, total_hits_val, cat_hits_result_map = self._calculate_weighted_hits(
            text,
            self.fluctuation_patterns_data, # type: ignore
            self.normalize_fluc_by_length,
            self.enable_fluc_synonyms
        )
        self.logger.debug(
            f"揺らぎ強度スコア計算結果: Hits={total_hits_val}, Score(0-1)={final_score_val:.3f}, "
            f"CategoryHits={ {getattr(k,'value','?'):v for k,v in cat_hits_result_map.items()} }"
        )
        
        FlucCat_cls_local_score = self.FluctuationCategoryV49_cls
        if FlucCat_cls_local_score:
            return final_score_val, {k: v for k, v in cat_hits_result_map.items() if isinstance(k, FlucCat_cls_local_score)}
        return final_score_val, {}

# =============================================================================
# -- Part 6 終了点
# =============================================================================
# =============================================================================
# -- Part 7: Enhanced DFRS Evaluation (v4.9β – 初期コードベース・堅牢性向上版 v5)
# =============================================================================
# - ユーザー提供の初期Part 7コードを厳密にベースとして修正
# - AttributeError (_safely_extract_list, DFRS計算メソッド) 解消のため、不足メソッドを復元・定義
# - NameError, UnboundLocalErrorを回避するための変数初期化とスコープ修正
# - Enum解決ロジックの改善とログ強化
# - v5: FileSettingsV49_clsのNameError修正、一部簡易処理メソッドの改善・ログ調整

from typing import TYPE_CHECKING, Counter as TypingCounter, Set, List, Dict, Optional, Tuple, Callable, Type, Union, Any
import enum
import pathlib
import hashlib
import json
import math
import statistics # DIT, PVS, ECS, PTN, Richness, Novelty で使用
import re # _analyze_structure, compute_symbolic_density, compute_expression_richness で使用
import logging

# グローバルスコープで利用可能であることを期待する変数 (Part 0 で定義)
# これらはスクリプトの先頭部分で適切にインポートまたは定義されている必要があります。

# ---------- type-checking-only imports ----------
if TYPE_CHECKING:
    from typing import TypeAlias # 標準ライブラリ
    # from numpy import ndarray as np_ndarray # 必要に応じて

    from __main__ import ( # type: ignore[attr-defined]
        ConfigProtocol, ScorerProtocol, DFRSMetricsV49, PsychologicalPhaseV49,
        EmotionalToneV49, PersistentCache, BaseModel, AppConfigV49,
        SubjectivityCategoryV49, FileSettingsV49,
        # TfidfVectorizer_cls as TfidfVectorizer_cls_hint, # Part 7では直接使用しない
        # cosine_similarity_func as cosine_similarity_func_hint, # Part 7では直接使用しない
        # np as np_hint, # Part 7では直接使用しない
        PYDANTIC_AVAILABLE
    )
    ConfigProtoType_eval_detail: TypeAlias = ConfigProtocol
    ScorerProtoType_eval_detail: TypeAlias = ScorerProtocol
    _PsychologicalPhaseV49EnumType_hint_eval: TypeAlias = PsychologicalPhaseV49
    _EmotionalToneV49EnumType_hint_eval: TypeAlias = EmotionalToneV49
    _DFRSMetricsV49EnumType_hint_eval: TypeAlias = DFRSMetricsV49
    _SubjectivityCategoryV49EnumType_hint_eval: TypeAlias = SubjectivityCategoryV49
    _PersistentCacheType_hint_eval: TypeAlias = PersistentCache[Dict[str, Optional[float]]]
    _SymbolicKeywordEntryType_hint: TypeAlias = Dict[str, Any]
else:
    ConfigProtoType_eval_detail = 'ConfigProtocol'
    ScorerProtoType_eval_detail = 'ScorerProtocol'
    _PsychologicalPhaseV49EnumType_hint_eval = 'PsychologicalPhaseV49'
    _EmotionalToneV49EnumType_hint_eval = 'EmotionalToneV49'
    _DFRSMetricsV49EnumType_hint_eval = 'DFRSMetricsV49'
    _SubjectivityCategoryV49EnumType_hint_eval = 'SubjectivityCategoryV49'
    _PersistentCacheType_hint_eval = 'PersistentCache'
    _SymbolicKeywordEntryType_hint = 'dict'

    # 実行時のグローバルクラス取得 (ユーザー提示の初期コードスタイル)
    # この部分はスクリプトの先頭 (Part 0 など) で一度だけ実行されるべきだが、
    # Part 7 のコードブロックとして提供されているため、ここにも記載する。
    # 実際の統合スクリプトでは、このような重複定義は避ける必要がある。
    try:
        DFRSMetricsV49 = globals()['DFRSMetricsV49']
        PsychologicalPhaseV49 = globals()['PsychologicalPhaseV49']
        EmotionalToneV49 = globals()['EmotionalToneV49']
        PersistentCache = globals()['PersistentCache']
        BaseModel = globals()['BaseModel']
        AppConfigV49 = globals()['AppConfigV49']
        SubjectivityCategoryV49 = globals()['SubjectivityCategoryV49']
        FileSettingsV49 = globals()['FileSettingsV49']
    except KeyError as e_key_part7_init_v5: # Version updated
        logging.getLogger(f"{__name__}.Part7Init.v5").critical(f"Part7初期化時: 必須グローバル定義 '{str(e_key_part7_init_v5)}' が見つかりません。")
        raise NameError(f"Missing global definition for Part 7 (Runtime): {e_key_part7_init_v5}") from e_key_part7_init_v5
    except Exception as e_glob_part7_init_other_v5: # Version updated
        logging.getLogger(f"{__name__}.Part7Init.v5").critical(f"Part7初期化時: グローバル定義取得中に予期せぬエラー: {e_glob_part7_init_other_v5}")
        raise


class DialogFlowEvaluatorBase:
    """基本的な対話構造分析機能を提供する基底クラス (v4.9β - 初期コードベース v5)"""
    def __init__(self, dialogue_text: Optional[str]):
        self.dialogue_text: str = dialogue_text.strip() if isinstance(dialogue_text, str) else ""
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        self.speech_block_texts: List[str] = []
        self.desc_block_texts: List[str] = []
        self.block_sequence_detailed: List[Tuple[str, str, int]] = [] # (type, text, length)
        self.block_sequence: List[Tuple[str, int]] = [] # (type, length)

        if self.dialogue_text:
            try:
                self._analyze_structure()
            except Exception as e:
                self.logger.error(f"対話テキストの構造解析中にエラーが発生しました: {e}", exc_info=True)
                self._clear_structure_data() # エラー時はデータをクリア
        else:
            self.logger.debug("対話テキストが空のため、構造解析はスキップされました。")
            self._clear_structure_data()

    def _clear_structure_data(self) -> None:
        self.speech_block_texts.clear()
        self.desc_block_texts.clear()
        self.block_sequence_detailed.clear()
        self.block_sequence.clear()
        self.logger.debug("構造解析関連の内部データをクリアしました。")

    def _analyze_structure(self) -> None:
        self._clear_structure_data()
        if not self.dialogue_text:
            self.logger.debug("構造解析: 対話テキストが空のため処理を中断します。")
            return

        # Regex to find speech blocks (e.g., 「...」 or 『...』)
        speech_pattern = re.compile(r'([「『](?:(?!「|』).)*?[」』])', re.DOTALL)
        current_pos = 0
        processed_blocks: List[Tuple[str, str]] = [] # Stores (type, text)

        try:
            for match in speech_pattern.finditer(self.dialogue_text):
                match_start, match_end = match.span()
                # Text before speech block is description
                if match_start > current_pos:
                    desc_text = self.dialogue_text[current_pos:match_start].strip()
                    if desc_text:
                        processed_blocks.append(("description", desc_text))
                # The speech block itself
                speech_text = match.group(0).strip() # group(0) is the entire match
                if speech_text:
                    processed_blocks.append(("speech", speech_text))
                current_pos = match_end
            
            # Text after the last speech block is description
            if current_pos < len(self.dialogue_text):
                tail_desc_text = self.dialogue_text[current_pos:].strip()
                if tail_desc_text:
                    processed_blocks.append(("description", tail_desc_text))

        except Exception as e_re: # Catch potential regex errors or other issues during finditer
            self.logger.error(f"構造解析中のテキスト処理でエラーが発生しました: {e_re}", exc_info=True)
            self._clear_structure_data() # Clear any partial data
            if self.dialogue_text: # Fallback: treat entire text as one description block
                self.block_sequence_detailed.append(("description", self.dialogue_text, len(self.dialogue_text)))
                self.block_sequence.append(("description", len(self.dialogue_text)))
                self.desc_block_texts.append(self.dialogue_text)
                self.logger.warning("構造解析エラーのため、テキスト全体を単一の描写ブロックとして処理しました。")
            return # Exit if error during iteration

        if not processed_blocks and self.dialogue_text:
            # No speech blocks found, treat the entire text as a single description block
            self.logger.info("構造解析: セリフ形式のブロックが見つかりませんでした。テキスト全体を単一の描写ブロックとして扱います。")
            self.block_sequence_detailed.append(("description", self.dialogue_text, len(self.dialogue_text)))
            self.block_sequence.append(("description", len(self.dialogue_text)))
            self.desc_block_texts.append(self.dialogue_text)
            return

        # Populate the final lists from processed_blocks
        for block_type, block_text in processed_blocks:
            block_length = len(block_text)
            self.block_sequence_detailed.append((block_type, block_text, block_length))
            self.block_sequence.append((block_type, block_length))
            if block_type == "speech":
                self.speech_block_texts.append(block_text)
            else: # description
                self.desc_block_texts.append(block_text)
        
        self.logger.info(f"構造解析完了: 総ブロック数={len(self.block_sequence_detailed)}, セリフブロック数={len(self.speech_block_texts)}, 描写ブロック数={len(self.desc_block_texts)}")

    def calculate_continuous_speech_ratio(self) -> float:
        if len(self.block_sequence) < 2: return 1.0
        transitions = len(self.block_sequence) - 1
        if transitions == 0: return 1.0 # Should be covered by the first line, but for safety.
        
        speech_speech_transitions = 0
        for i in range(transitions):
            if self.block_sequence[i][0] == 'speech' and self.block_sequence[i+1][0] == 'speech':
                speech_speech_transitions += 1
        
        # Score is 1.0 if there are no consecutive speech blocks (perfect alternation or single speech block).
        # Score decreases as the ratio of speech-speech transitions increases.
        score = 1.0 - (speech_speech_transitions / transitions)
        return round(max(0.0, min(1.0, score)), 4)

    def calculate_description_insertion_timing(self) -> float:
        desc_lengths = [length for type_val, length in self.block_sequence if type_val == 'description' and length > 0]
        if len(desc_lengths) < 2:
            self.logger.debug("DIT: 有効な描写ブロックが2未満のため、デフォルトスコア0.5を返します。")
            return 0.5
        try:
            mean_len = statistics.mean(desc_lengths)
            # Ensure at least 2 data points for stdev, otherwise stdev is 0
            std_dev = statistics.stdev(desc_lengths) if len(desc_lengths) >= 2 else 0.0
        except statistics.StatisticsError as e_stat:
            self.logger.warning(f"DIT統計計算エラー ({e_stat})。対象描写ブロック数: {len(desc_lengths)}。デフォルトスコア0.5を返します。")
            return 0.5
        except Exception as e_unexp:
            self.logger.error(f"DIT計算中に予期せぬエラー: {e_unexp}", exc_info=True)
            return 0.5 # Fallback for unexpected errors
            
        cv = (std_dev / mean_len) if mean_len > 1e-9 else 0.0 # Coefficient of Variation
        # tanh maps CV to 0-1 range. Higher CV (more varied lengths/timing) means higher score.
        # The multiplier 1.2 adjusts the sensitivity of tanh.
        dit_score = math.tanh(cv * 1.2)
        return round(max(0.0, min(1.0, dit_score)), 4)

    def calculate_pacing_variability(self) -> float:
        block_lengths = [length for _, length in self.block_sequence if length > 0] # Consider all block types
        if len(block_lengths) < 2:
            self.logger.debug("PVS: 有効なブロックが2未満のため、デフォルトスコア0.0を返します。") # Variability is low if few blocks
            return 0.0
        try:
            mean_len = statistics.mean(block_lengths)
            std_dev = statistics.stdev(block_lengths) if len(block_lengths) >= 2 else 0.0
        except statistics.StatisticsError as e_stat:
            self.logger.warning(f"PVS統計計算エラー ({e_stat})。対象ブロック数: {len(block_lengths)}。デフォルトスコア0.0を返します。")
            return 0.0
        except Exception as e_unexp:
            self.logger.error(f"PVS計算中に予期せぬエラー: {e_unexp}", exc_info=True)
            return 0.0
            
        cv = (std_dev / mean_len) if mean_len > 1e-9 else 0.0
        # Higher CV (more varied pacing) means higher score.
        # Multiplier 1.5 adjusts sensitivity.
        pvs_score = math.tanh(cv * 1.5)
        return round(max(0.0, min(1.0, pvs_score)), 4)


class EnhancedDialogFlowEvaluatorV49(DialogFlowEvaluatorBase):
    if TYPE_CHECKING:
        ConfigProtoType_eval_detail: TypeAlias = ConfigProtocol
        ScorerProtoType_eval_detail: TypeAlias = ScorerProtocol
        _PsychologicalPhaseV49EnumType_hint_eval: TypeAlias = PsychologicalPhaseV49
        _EmotionalToneV49EnumType_hint_eval: TypeAlias = EmotionalToneV49
        _DFRSMetricsV49EnumType_hint_eval: TypeAlias = DFRSMetricsV49
        _PersistentCacheType_hint_eval: TypeAlias = PersistentCache[Dict[str, Optional[float]]]
    else:
        ConfigProtoType_eval_detail = 'ConfigProtocol'
        ScorerProtoType_eval_detail = 'ScorerProtocol'
        _PsychologicalPhaseV49EnumType_hint_eval = 'PsychologicalPhaseV49'
        _EmotionalToneV49EnumType_hint_eval = 'EmotionalToneV49'
        _DFRSMetricsV49EnumType_hint_eval = 'DFRSMetricsV49'
        _PersistentCacheType_hint_eval = 'PersistentCache'

    def __init__(self, config: ConfigProtoType_eval_detail, scorer: ScorerProtoType_eval_detail): # type: ignore
        self.config = config
        self.scorer = scorer
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        system_version_init = getattr(self.config, 'SYSTEM_VERSION', 'N/A')
        self.logger.info(f"EnhancedDialogFlowEvaluatorV49 (System Version: {system_version_init}) 初期化開始...")

        # Load essential Enum classes from globals
        self.PsychologicalPhaseV49_cls: Optional[Type[PsychologicalPhaseV49]] = globals().get('PsychologicalPhaseV49') # type: ignore
        self.EmotionalToneV49_cls: Optional[Type[EmotionalToneV49]] = globals().get('EmotionalToneV49') # type: ignore
        self.DFRSMetrics_cls: Optional[Type[DFRSMetricsV49]] = globals().get('DFRSMetricsV49') # type: ignore
        
        enum_load_errors_list: List[str] = []
        if not (self.PsychologicalPhaseV49_cls and isinstance(self.PsychologicalPhaseV49_cls, enum.EnumMeta)):
            enum_load_errors_list.append("PsychologicalPhaseV49")
        if not (self.EmotionalToneV49_cls and isinstance(self.EmotionalToneV49_cls, enum.EnumMeta)):
            enum_load_errors_list.append("EmotionalToneV49")
        if not (self.DFRSMetrics_cls and isinstance(self.DFRSMetrics_cls, enum.EnumMeta)):
            enum_load_errors_list.append("DFRSMetricsV49")
        
        if enum_load_errors_list:
            self.logger.critical(f"CRITICAL ERROR: 必須Enumクラス ({', '.join(enum_load_errors_list)}) のロードに失敗したか、型が不正です。評価機能に重大な支障が出る可能性があります。")

        # Load DFRS weights from config
        self.weights: Dict[_DFRSMetricsV49EnumType_hint_eval, float] = {} # type: ignore
        if self.DFRSMetrics_cls and hasattr(self.config, 'dfrs_weights') and getattr(self.config, 'dfrs_weights') is not None:
            raw_weights_data = getattr(self.config, 'dfrs_weights', {})
            if isinstance(raw_weights_data, dict):
                for key_str, weight_val in raw_weights_data.items():
                    metric_enum_member: Optional['DFRSMetricsV49'] = None
                    if isinstance(key_str, self.DFRSMetrics_cls): # If key is already an Enum member
                        metric_enum_member = key_str
                    elif isinstance(key_str, str): # If key is a string, try to convert
                        try:
                            metric_enum_member = self.DFRSMetrics_cls(key_str.lower().strip().replace('-', '_')) # type: ignore
                        except ValueError: # If direct conversion fails, try by attribute name (all caps)
                            try:
                                metric_enum_member = getattr(self.DFRSMetrics_cls, key_str.upper().strip().replace('-', '_')) # type: ignore
                            except (AttributeError, ValueError):
                                self.logger.warning(f"DFRS重みキー '{key_str}' はDFRSMetricsV49 Enumのメンバーとして解決できませんでした。このキーは無視されます。")
                    else:
                        self.logger.warning(f"DFRS重みキー '{key_str}' (型: {type(key_str)}) が不正です。文字列またはDFRSMetricsV49 Enumメンバーである必要があります。")

                    if metric_enum_member and metric_enum_member not in [self.DFRSMetrics_cls.UNKNOWN, self.DFRSMetrics_cls.FINAL_EODF_V49]: # type: ignore
                        if isinstance(weight_val, (int, float)) and weight_val >= 0:
                            self.weights[metric_enum_member] = float(weight_val) # type: ignore
                        else:
                            self.logger.warning(f"DFRSメトリック '{getattr(metric_enum_member,'value',key_str)}' の重み値 '{weight_val}' が不正です (非数値または負数)。この重みは無視されます。")
            else:
                self.logger.error("設定 'config.dfrs_weights' が辞書形式ではありません。DFRS重みはロードされません。")
        else:
            self.logger.warning("DFRSMetrics_clsがロードされていないか、設定 'config.dfrs_weights' が存在しません。DFRS重みはロードされません。")
        
        if not self.weights:
            self.logger.error("有効なDFRS重みが一つもロードされませんでした。eODFスコア計算に影響します。")

        # Initialize persistent cache if enabled
        self.dfrs_cache: Optional[_PersistentCacheType_hint_eval] = None # type: ignore
        feature_flags = getattr(self.config, 'feature_flags', None)
        is_cache_enabled = getattr(feature_flags, 'persistent_cache_enabled', True) if feature_flags else True
        
        if is_cache_enabled:
            self.dfrs_cache = self._initialize_persistent_cache()
        else:
            self.logger.info("DFRSスコアの永続キャッシュは設定 (feature_flags.persistent_cache_enabled) により無効化されています。")
        
        self.logger.info(f"{self.__class__.__qualname__} の初期化が完了しました。ロードされたDFRS重みの数: {len(self.weights)}, DFRSキャッシュ利用可能か: {self.dfrs_cache is not None}")

    def _initialize_persistent_cache(self) -> Optional[_PersistentCacheType_hint_eval]: # type: ignore
        # Attempt to load necessary classes from globals
        PersistentCache_cls_local: Optional[Type['PersistentCache']] = globals().get('PersistentCache') # type: ignore
        FileSettingsV49_cls_local: Optional[Type['FileSettingsV49']] = globals().get('FileSettingsV49') # type: ignore
        BaseModel_local_for_cache: Type[Any] = globals().get('BaseModel', object)
        PYDANTIC_AVAILABLE_for_cache: bool = bool(globals().get('PYDANTIC_AVAILABLE', False))

        if not PersistentCache_cls_local:
            self.logger.warning("PersistentCacheクラスがグローバルスコープからロードできませんでした。DFRSキャッシュは無効になります。")
            return None
        
        try:
            file_settings_instance: Optional[Any] = None
            cache_config_dict: Dict[str, Any] = {}
            
            # Get settings from config object
            external_configs_obj = getattr(self.config, 'loaded_external_configs', None)
            if external_configs_obj:
                file_settings_instance = getattr(external_configs_obj, 'file_settings', None)
                raw_cache_config_value = getattr(external_configs_obj, 'cache_config', None)
                
                if isinstance(raw_cache_config_value, dict):
                    cache_config_dict = raw_cache_config_value
                elif PYDANTIC_AVAILABLE_for_cache and BaseModel_local_for_cache is not object and \
                     isinstance(raw_cache_config_value, BaseModel_local_for_cache) and hasattr(raw_cache_config_value, 'model_dump'):
                    try:
                        cache_config_dict = raw_cache_config_value.model_dump() # type: ignore
                    except Exception as e_dump_cache_cfg:
                        self.logger.warning(f"Pydanticモデル `cache_config` の model_dump() 呼び出しに失敗しました: {e_dump_cache_cfg}")
            
            # Determine cache base path
            default_cache_base_path = pathlib.Path("cache/dfrs_eval_cache_v49_final_v5_part7") # Updated version in path
            configured_cache_dir: Optional[pathlib.Path] = getattr(self.config, 'cache_dir', None) # type: ignore
            final_cache_base_path: pathlib.Path

            # *** NameError修正: FileSettingsV49_cls を FileSettingsV49_cls_local に変更 ***
            if file_settings_instance and FileSettingsV49_cls_local and \
               isinstance(file_settings_instance, FileSettingsV49_cls_local) and \
               hasattr(file_settings_instance, 'persistent_cache_dir') and \
               isinstance(file_settings_instance.persistent_cache_dir, pathlib.Path):
                final_cache_base_path = file_settings_instance.persistent_cache_dir / "dfrs_eval_cache_v49_final_v5_part7" # Consistent subfolder
            elif isinstance(configured_cache_dir, pathlib.Path):
                final_cache_base_path = configured_cache_dir / "dfrs_eval_cache_v49_final_v5_part7"
            else:
                self.logger.warning(f"キャッシュディレクトリが設定から特定できませんでした。デフォルトパス '{default_cache_base_path}' を使用します。")
                final_cache_base_path = default_cache_base_path
            
            # Define cache table name and DB file path
            cache_table_name = getattr(self.config, 'CACHE_DFRS_TABLE_NAME', 'dfrs_cache_v49_final_v5_part7') # Updated version
            cache_db_filename = f"{cache_table_name}.db"
            cache_db_path = final_cache_base_path / cache_db_filename
            
            try: # Ensure cache directory exists
                cache_db_path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"キャッシュディレクトリ '{cache_db_path.parent}' の存在を確認/作成しました。")
            except OSError as e_mkdir_cache:
                self.logger.error(f"DFRSキャッシュディレクトリ '{cache_db_path.parent}' の作成に失敗しました ({e_mkdir_cache})。キャッシュは無効になります。")
                return None

            # Instantiate PersistentCache
            cache_instance = PersistentCache_cls_local(
                db_path=cache_db_path,
                table_name=cache_table_name,
                max_size_mb=float(cache_config_dict.get('max_size_mb', 200.0)), # Default 200MB
                auto_cleanup_on_set_chance=float(cache_config_dict.get('auto_cleanup_on_set_chance', 0.1)), # Default 10% chance
                default_ttl_seconds=int(cache_config_dict.get('dfrs_cache_ttl_seconds', 24 * 60 * 60 * 7)), # Default 7 days TTL
                logger_instance=self.logger.getChild("Cache.DFRS_v49_Final_v5_Part7") # Updated version in logger name
            )
            
            if getattr(cache_instance, '_is_initialized_successfully', False):
                self.logger.info(f"DFRSスコア用PersistentCacheの初期化に成功しました (DB: '{cache_db_path}', Table: '{cache_table_name}')")
                return cache_instance
            else:
                self.logger.error(f"DFRSスコア用PersistentCacheの初期化に失敗しました (DB: '{cache_db_path}')。キャッシュは利用できません。")
        except Exception as e_init_cache:
            self.logger.error(f"DFRSスコア用PersistentCacheの初期化中に予期せぬエラーが発生しました: {e_init_cache}", exc_info=True)
        
        return None # Return None if any error occurs

    def _get_enum_member_from_value(
        self, enum_class: Optional[Type[enum.Enum]], value: Any,
        default_member_override: Optional[enum.Enum] = None
    ) -> Optional[enum.Enum]:
        # This method is crucial for converting string values (e.g., from config or data) to Enum members.
        logger_enum_conv = self.logger.getChild("_get_enum_member_from_value_eval_v5_part7") # Version updated
        if not enum_class or not isinstance(enum_class, enum.EnumMeta):
            logger_enum_conv.error(f"無効なEnumクラス '{enum_class}' (型: {type(enum_class)}) が指定されました。Noneを返します。")
            return None
        
        # Determine the default member to use if conversion fails or input is None.
        # Prefer UNKNOWN member if it exists in the enum_class.
        unknown_member_of_enum = getattr(enum_class, 'UNKNOWN', None)
        effective_default = default_member_override if default_member_override is not None else unknown_member_of_enum

        if value is None:
            logger_enum_conv.debug(f"入力値がNoneです。Enum '{enum_class.__name__}' のデフォルトメンバー '{getattr(effective_default,'name','N/A')}' を返します。")
            return effective_default
        if isinstance(value, enum_class): # Already an enum member
            return value

        val_str_to_convert = str(value).strip()
        if not val_str_to_convert: # Empty string after stripping
            logger_enum_conv.debug(f"入力値が空文字列です。Enum '{enum_class.__name__}' のデフォルトメンバー '{getattr(effective_default,'name','N/A')}' を返します。")
            return effective_default

        # Attempt to handle "EnumClassName.MEMBER_NAME" style strings
        if "." in val_str_to_convert:
            try:
                class_name_part, member_name_part_from_str = val_str_to_convert.rsplit(".", 1)
                if class_name_part.lower() == enum_class.__name__.lower(): # Check if class name part matches
                    normalized_member_name = member_name_part_from_str.upper().replace('-', '_') # Normalize for common variations
                    if hasattr(enum_class, normalized_member_name):
                        resolved_member = getattr(enum_class, normalized_member_name)
                        logger_enum_conv.debug(f"値 '{value}' はドット区切り形式 ('{enum_class.__name__}.{member_name_part_from_str}') でメンバー '{resolved_member.name}' に解決されました。")
                        return resolved_member
            except Exception: # If parsing fails, ignore and proceed to standard conversion
                pass

        # Standard conversion attempt (delegates to Enum's __call__ or _missing_)
        try:
            converted_member = enum_class(val_str_to_convert) # type: ignore
            member_name_for_logging = getattr(converted_member, 'name', str(converted_member))
            
            # Check if the input string was literally "unknown" vs. if a different string resolved to the UNKNOWN member
            was_input_str_unknown = val_str_to_convert.lower() == 'unknown'
            is_resolved_to_actual_unknown_member = unknown_member_of_enum and converted_member == unknown_member_of_enum
            
            # Log if a non-"unknown" input string unexpectedly resolved to the UNKNOWN member
            if is_resolved_to_actual_unknown_member and not was_input_str_unknown and \
               (not hasattr(unknown_member_of_enum, 'value') or val_str_to_convert.lower() != unknown_member_of_enum.value.lower()) and \
               (not hasattr(unknown_member_of_enum, 'name') or val_str_to_convert.upper() != unknown_member_of_enum.name.upper()): # type: ignore
                logger_enum_conv.info(f"値 '{value}' (文字列: '{val_str_to_convert}') は Enum '{enum_class.__name__}' の UNKNOWN メンバー ('{getattr(unknown_member_of_enum, 'value', 'N/A')}') にフォールバック（または_missing_により解決）されました。")
            else:
                logger_enum_conv.debug(f"値 '{value}' (文字列: '{val_str_to_convert}') は Enum '{enum_class.__name__}' のメンバー '{member_name_for_logging}' に正常に変換されました。")
            return converted_member
        except ValueError: # Standard conversion failed
            default_member_name_for_log = getattr(effective_default, 'name', 'N/A') if effective_default else 'None'
            logger_enum_conv.warning(f"値 '{value}' (文字列: '{val_str_to_convert}') をEnum '{enum_class.__name__}' の有効なメンバーに変換できませんでした。デフォルトの '{default_member_name_for_log}' を返します。")
            return effective_default
        except Exception as e_unexpected_conversion: # Catch any other unexpected errors during conversion
            default_member_name_for_exc_log = getattr(effective_default, 'name', 'N/A') if effective_default else 'None'
            logger_enum_conv.error(f"Enum '{enum_class.__name__}' の値 '{value}' 変換中に予期せぬエラーが発生: {e_unexpected_conversion}。デフォルトの '{default_member_name_for_exc_log}' を返します。", exc_info=True)
            return effective_default

    def _safely_extract_list(self, data_dict: Any, key: str) -> List[Dict[str, Any]]:
        # Safely extracts a list of dictionaries, handling Pydantic models if present.
        if not isinstance(data_dict, dict):
            self.logger.debug(f"_safely_extract_list: 入力データが辞書ではありません (キー: {key})。空リストを返します。")
            return []
        
        value_from_dict = data_dict.get(key)
        if not isinstance(value_from_dict, list):
            self.logger.debug(f"_safely_extract_list: キー '{key}' の値がリストではありません (型: {type(value_from_dict)})。空リストを返します。")
            return []
        
        processed_items_list: List[Dict[str, Any]] = []
        BaseModel_class_local = globals().get('BaseModel') # For Pydantic model check
        is_pydantic_available_locally = bool(globals().get('PYDANTIC_AVAILABLE', False))
        
        if not (BaseModel_class_local and is_pydantic_available_locally):
            self.logger.debug("_safely_extract_list: Pydanticが利用不可かBaseModelが未定義のため、Pydanticモデルのダンプは行いません。辞書型のみを処理します。")

        for index, item_in_list in enumerate(value_from_dict):
            if isinstance(item_in_list, dict):
                processed_items_list.append(item_in_list)
            elif is_pydantic_available_locally and BaseModel_class_local and isinstance(item_in_list, BaseModel_class_local): # type: ignore
                try: # Attempt Pydantic V2 style model_dump
                    processed_items_list.append(item_in_list.model_dump(mode='json', exclude_none=True, by_alias=True)) # type: ignore
                except AttributeError: # Fallback for Pydantic V1 style .dict() or if model_dump is missing
                    try:
                        processed_items_list.append(item_in_list.dict(exclude_none=True, by_alias=True)) # type: ignore
                    except Exception as e_v1_dump:
                        self.logger.warning(f"Pydantic V1形式の .dict() でのダンプに失敗しました (キー: {key}, インデックス: {index}, 型: {type(item_in_list)}): {e_v1_dump}")
                except Exception as e_model_dump:
                    self.logger.warning(f"Pydanticモデルのダンプに失敗しました (キー: {key}, インデックス: {index}, 型: {type(item_in_list)}): {e_model_dump}")
            else:
                self.logger.debug(f"_safely_extract_list: キー '{key}' 内の要素 (インデックス: {index}) は辞書でもPydanticモデルでもありません (型: {type(item_in_list)})。この要素はスキップされます。")
        return processed_items_list
        
    def compute_subjectivity_score(self) -> Optional[float]:
        # This method uses self.scorer, which should provide a detailed score.
        if not (hasattr(self, 'scorer') and self.scorer and hasattr(self.scorer, 'calculate_subjectivity_score')):
            self.logger.warning("Scorerまたはcalculate_subjectivity_scoreメソッドが未設定です。主観性スコアとして0.0を返します。")
            return 0.0
        try:
            if not hasattr(self, 'dialogue_text'): # dialogue_text should be set by super().__init__
                self.logger.error("compute_subjectivity_score: self.dialogue_textが未設定（DialogFlowEvaluatorBaseの初期化未実行の可能性）。0.0を返します。")
                return 0.0
            score_value, _ = self.scorer.calculate_subjectivity_score(self.dialogue_text) # type: ignore
            if isinstance(score_value, (int, float)):
                return round(max(0.0, min(1.0, float(score_value))), 4)
            else:
                self.logger.warning(f"Scorerから返された主観性スコアが数値ではありません (型: {type(score_value)})。0.0を返します。")
                return 0.0
        except Exception as e_subj_score:
            self.logger.error(f"compute_subjectivity_scoreの実行中にエラーが発生しました: {e_subj_score}", exc_info=True)
            return 0.0 # Fallback on error

    def compute_fluctuation_intensity(self) -> Optional[float]:
        # This method uses self.scorer.
        if not hasattr(self, 'dialogue_text') or not self.dialogue_text: # Check if dialogue_text is empty or not set
            self.logger.debug("compute_fluctuation_intensity: 対話テキストが空または未設定のため、0.0を返します。")
            return 0.0
        if not (hasattr(self, 'scorer') and self.scorer and hasattr(self.scorer, 'calculate_fluctuation_intensity')):
            self.logger.warning("Scorerまたはcalculate_fluctuation_intensityメソッドが未設定です。揺らぎ強度として0.0を返します。")
            return 0.0
        try:
            raw_score_value, category_hits_map = self.scorer.calculate_fluctuation_intensity(self.dialogue_text) # type: ignore
            if not isinstance(raw_score_value, (int, float)):
                self.logger.warning(f"Scorerから返された揺らぎ強度スコアが数値ではありません (型: {type(raw_score_value)})。0.0を返します。")
                return 0.0
            final_fluc_score = round(max(0.0, min(1.0, float(raw_score_value))), 4)
            # Log hits for debugging if needed
            num_hits = sum(category_hits_map.values()) if isinstance(category_hits_map, dict) else 'N/A'
            self.logger.info(f"Fluctuation Intensity: Score={final_fluc_score:.3f}, Total Category Hits={num_hits}")
            return final_fluc_score
        except Exception as e_fluc_intensity:
            self.logger.error(f"compute_fluctuation_intensityの実行中にエラーが発生しました: {e_fluc_intensity}", exc_info=True)
            return 0.0

    def calculate_emotion_curve_smoothness(self, emotion_curve_data: Optional[List[Dict[str, Any]]]) -> float:
        self.logger.debug(f"ECS (Emotion Curve Smoothness) 計算開始: 感情曲線データポイント数: {len(emotion_curve_data) if emotion_curve_data else 0}")
        if not emotion_curve_data or len(emotion_curve_data) < 2:
            self.logger.debug("ECS: データポイントが2未満のため、デフォルトスコア0.5を返します。")
            return 0.5
        
        ET_cls_local = getattr(self, 'EmotionalToneV49_cls', None)
        if not ET_cls_local:
            self.logger.error("ECS: EmotionalToneV49_clsがロードされていません。スコア0.3を返します（処理続行不可）。")
            return 0.3
            
        # Define tones to exclude from smoothness calculation (typically neutral or unknown)
        excluded_tones_from_calc = {getattr(ET_cls_local, 'UNKNOWN', None), getattr(ET_cls_local, 'NEUTRAL', None)}
        excluded_tones_from_calc.discard(None) # Remove None if UNKNOWN or NEUTRAL doesn't exist in Enum

        valid_transitions_count = 0
        category_transition_events = 0
        strength_difference_accumulator = 0.0
        last_valid_tone_enum: Optional[EmotionalToneV49] = None # type: ignore
        last_valid_strength_value: float = 0.0

        for i, point_dict_data in enumerate(emotion_curve_data):
            if not isinstance(point_dict_data, dict): continue # Skip malformed data points
            
            tone_str_value = point_dict_data.get("tone")
            current_tone_enum_member = self._get_enum_member_from_value(ET_cls_local, tone_str_value)
            
            if not current_tone_enum_member or current_tone_enum_member in excluded_tones_from_calc:
                self.logger.debug(f"ECS: ポイント {i+1} のトーン ('{getattr(current_tone_enum_member, 'value', tone_str_value)}') は除外対象です。")
                continue
            
            strength_from_data = point_dict_data.get("strength")
            current_strength_value = 0.5 # Default strength if invalid or missing
            if isinstance(strength_from_data, (int, float)):
                current_strength_value = round(max(0.0, min(1.0, float(strength_from_data))), 4)
            
            if last_valid_tone_enum is not None: # This means we have a previous valid point to compare with
                valid_transitions_count += 1
                if current_tone_enum_member != last_valid_tone_enum:
                    category_transition_events += 1
                strength_difference_accumulator += abs(current_strength_value - last_valid_strength_value)
            
            last_valid_tone_enum = current_tone_enum_member # type: ignore
            last_valid_strength_value = current_strength_value
            
        if valid_transitions_count == 0:
            self.logger.debug("ECS: 有効な感情遷移（比較可能なポイント）がなかったため、デフォルトスコア0.5を返します。")
            return 0.5
            
        category_stability_metric = 1.0 - (category_transition_events / valid_transitions_count)
        average_strength_difference_metric = strength_difference_accumulator / valid_transitions_count
        strength_stability_metric = 1.0 - average_strength_difference_metric # Higher difference means lower stability

        # Weighted average for the final ECS score
        final_ecs_score_value = round(max(0.0, min(1.0, (category_stability_metric * 0.6) + (strength_stability_metric * 0.4))), 4)
        self.logger.info(f"ECS計算完了: カテゴリ安定性={category_stability_metric:.3f}, 強度安定性={strength_stability_metric:.3f}, 最終スコア={final_ecs_score_value} ({valid_transitions_count} 有効遷移ポイント)")
        return final_ecs_score_value

    def calculate_dialogue_description_balance(self) -> float:
        self.logger.debug("DDB (Dialogue Description Balance) 計算呼び出し。")
        if not (hasattr(self,'speech_block_texts') and hasattr(self,'desc_block_texts')):
            self.logger.error("DDBエラー: speech_block_textsまたはdesc_block_textsがDialogFlowEvaluatorBaseで初期化されていません。デフォルトスコア0.5を返します。")
            return 0.5 # Indicate neutral balance if data is missing
            
        total_speech_char_length = sum(len(text_block) for text_block in self.speech_block_texts)
        total_description_char_length = sum(len(text_block) for text_block in self.desc_block_texts)
        total_char_length = total_speech_char_length + total_description_char_length
        
        if total_char_length == 0:
            self.logger.debug("DDB: 総テキスト長が0のため、バランススコア0.5（中立）を返します。")
            return 0.5
            
        speech_char_ratio = total_speech_char_length / total_char_length
        # Score is 1.0 if speech_ratio is 0.5 (perfect balance), decreases linearly to 0.0 as it approaches 0 or 1.
        balance_score_value = 1.0 - 2.0 * abs(speech_char_ratio - 0.5)
        return round(max(0.0, min(1.0, balance_score_value)), 4)

    def calculate_transition_smoothness(self) -> float:
        self.logger.debug("TS (Transition Smoothness) 計算呼び出し。")
        if not hasattr(self,'block_sequence') or len(self.block_sequence) < 2:
            self.logger.debug("TS: ブロックシーケンスが2未満（遷移なしまたは1つのみ）のため、デフォルトスコア0.7（ややスムーズ）を返します。")
            return 0.7
            
        num_total_transitions = len(self.block_sequence) - 1
        num_alternating_transitions = 0 # speech-desc or desc-speech
        for i in range(num_total_transitions):
            if self.block_sequence[i][0] != self.block_sequence[i+1][0]: # Check type of block ([0] is type)
                num_alternating_transitions += 1
        
        # Higher ratio of alternating transitions means smoother flow.
        smoothness_score_value = num_alternating_transitions / num_total_transitions if num_total_transitions > 0 else 0.7 # Default if no transitions
        return round(max(0.0, min(1.0, smoothness_score_value)), 4)

    def calculate_phase_transition_naturalness(self, phase_timeline_data: Optional[List[Dict[str, Any]]]) -> float:
        self.logger.debug(f"PTN (Phase Transition Naturalness) 計算開始: タイムライン長: {len(phase_timeline_data) if phase_timeline_data else 0}")
        if not phase_timeline_data or len(phase_timeline_data) < 2:
            self.logger.debug("PTN: データポイントが2未満のため、デフォルトスコア0.5を返します。")
            return 0.5

        PP_cls_local = getattr(self, 'PsychologicalPhaseV49_cls', None)
        if not PP_cls_local:
            self.logger.error("PTN: PsychologicalPhaseV49_clsがロードされていません。スコア0.3を返します（処理続行不可）。")
            return 0.3
        
        # Attempt to get phase_transition_matrix from config
        phase_transition_matrix_data: Optional[Dict[str, Dict[str, float]]] = None
        loaded_external_configs_obj = getattr(self.config, 'loaded_external_configs', None)
        if loaded_external_configs_obj:
            phase_transition_matrix_data = getattr(loaded_external_configs_obj, 'phase_transition_matrix', None)

        if not isinstance(phase_transition_matrix_data, dict) or not phase_transition_matrix_data:
            self.logger.warning("PTN: 位相遷移マトリックスが設定ファイルからロードできないか空です。デフォルトスコア0.4を返します。")
            return 0.4

        num_actual_transitions = 0
        total_naturalness_score_sum = 0.0
        last_valid_phase_enum: Optional[PsychologicalPhaseV49] = None # type: ignore
        unknown_phase_enum_instance = getattr(PP_cls_local, 'UNKNOWN', None)

        for point_dict_data in phase_timeline_data:
            if not isinstance(point_dict_data, dict): continue

            phase_str_value = point_dict_data.get("phase")
            current_phase_enum_member = self._get_enum_member_from_value(PP_cls_local, phase_str_value)

            if not current_phase_enum_member or current_phase_enum_member == unknown_phase_enum_instance:
                continue # Skip unknown or invalid phases for transition calculation

            if last_valid_phase_enum and last_valid_phase_enum != unknown_phase_enum_instance: # A valid previous phase exists
                num_actual_transitions += 1
                from_phase_key_str = last_valid_phase_enum.value # type: ignore
                to_phase_key_str = current_phase_enum_member.value # type: ignore
                
                current_transition_naturalness_score = 0.3 # Default for unlisted or one-way transitions
                if from_phase_key_str in phase_transition_matrix_data and \
                   isinstance(phase_transition_matrix_data[from_phase_key_str], dict):
                    current_transition_naturalness_score = phase_transition_matrix_data[from_phase_key_str].get(to_phase_key_str, 0.3)
                
                total_naturalness_score_sum += float(current_transition_naturalness_score)
            
            last_valid_phase_enum = current_phase_enum_member # type: ignore
            
        if num_actual_transitions == 0:
            self.logger.debug("PTN: 有効な位相遷移（比較可能なポイント間）がなかったため、デフォルトスコア0.7を返します。")
            return 0.7
            
        average_naturalness_value = total_naturalness_score_sum / num_actual_transitions
        final_ptn_score_value = round(max(0.0, min(1.0, average_naturalness_value)), 4)
        self.logger.info(f"PTN計算完了: 平均自然さ={average_naturalness_value:.3f}, 最終スコア={final_ptn_score_value} ({num_actual_transitions} 有効遷移)")
        return final_ptn_score_value

    def compute_internal_depth(self, analyzer_results: Optional[Dict[str, Any]]) -> float:
        # Calculation based on subjectivity_score. `analyzer_results` currently unused but kept for future.
        self.logger.debug("compute_internal_depth を計算します。")
        subjectivity_score_value = self.compute_subjectivity_score()
        
        # Heuristic: higher subjectivity generally implies more internal depth.
        # 0.1 base, 0.8 scaling factor for subjectivity.
        depth_score = 0.3 # Default if subjectivity_score is None or error
        if subjectivity_score_value is not None:
             depth_score = round((subjectivity_score_value * 0.8) + 0.1, 4)
        
        # TODO (Future Enhancement): Could leverage analyzer_results['nlp_analysis_summary']
        # e.g., count of abstract nouns, reflective phrases, etc., if available.
        # For now, relies solely on the subjectivity score.
        self.logger.info(f"Internal Depth Score (0-1): {depth_score:.4f} (主観性スコアに依存)")
        return depth_score

    def compute_emotion_complexity(self, emotion_curve_data: Optional[List[Dict[str, Any]]]) -> float:
        # Calculates complexity based on the variety of significant emotions detected.
        self.logger.debug("compute_emotion_complexity を計算します。")
        ET_cls_local = getattr(self, 'EmotionalToneV49_cls', None)
        if not emotion_curve_data or len(emotion_curve_data) < 2 or not ET_cls_local:
            self.logger.debug("Emotion Complexity: データ不足またはEmotionalToneV49_cls未ロード。デフォルトスコア0.3を返します。")
            return 0.3
            
        unique_significant_tones = set()
        # Tones to exclude when counting unique emotions for complexity
        excluded_tones_for_complexity_count = {getattr(ET_cls_local, 'UNKNOWN', None), getattr(ET_cls_local, 'NEUTRAL', None)}
        excluded_tones_for_complexity_count.discard(None) # Remove None if members don't exist

        for point_data_dict in emotion_curve_data:
            if isinstance(point_data_dict, dict):
                tone_enum_member = self._get_enum_member_from_value(ET_cls_local, point_data_dict.get("tone"))
                if tone_enum_member and tone_enum_member not in excluded_tones_for_complexity_count:
                    unique_significant_tones.add(tone_enum_member)
        
        # Score based on the number of unique significant emotions.
        # Heuristic: 5 unique significant emotions map to a score of 1.0.
        # This is a simple measure; could be enhanced by considering emotion intensity or transition patterns.
        num_unique_tones = len(unique_significant_tones)
        complexity_score_value = round(min(1.0, num_unique_tones / 5.0), 4)
        self.logger.info(f"Emotion Complexity: 有効なユニークトーン数 (Neutral/Unknown除く) = {num_unique_tones}, Score={complexity_score_value:.4f}")
        return complexity_score_value

    def compute_eti(self, emotion_curve_data: Optional[List[Dict[str, Any]]]) -> float: # Signature changed
        # Emotional Trajectory Inflexion: Measures significant changes in the emotional trajectory.
        self.logger.debug("compute_eti (Emotional Trajectory Inflexion) を計算します。")
        if not emotion_curve_data or len(emotion_curve_data) < 2:
            self.logger.debug("ETI: 感情曲線データが2ポイント未満のため、デフォルトスコア0.3を返します。")
            return 0.3

        ET_cls_local = getattr(self, 'EmotionalToneV49_cls', None)
        if not ET_cls_local:
            self.logger.error("ETI: EmotionalToneV49_clsがロードされていません。デフォルトスコア0.2を返します（処理続行不可）。")
            return 0.2

        inflexion_event_count = 0
        # Threshold for what's considered a "significant" change in emotion strength
        SIGNIFICANT_STRENGTH_CHANGE_THRESHOLD = 0.3
        
        last_valid_tone_enum: Optional[enum.Enum] = None
        last_valid_strength: float = 0.0
        num_comparable_transitions = 0

        # Tones to exclude from ETI calculation (typically non-emotional or undefined)
        excluded_tones_for_eti = {getattr(ET_cls_local, 'UNKNOWN', None), getattr(ET_cls_local, 'NEUTRAL', None)}
        excluded_tones_for_eti.discard(None)

        for i, point_data_dict in enumerate(emotion_curve_data):
            if not isinstance(point_data_dict, dict): continue
            
            current_tone_enum_member = self._get_enum_member_from_value(ET_cls_local, point_data_dict.get("tone"))
            strength_val_from_data = point_data_dict.get("strength")
            current_strength_val = 0.5 # Default if missing or invalid
            if isinstance(strength_val_from_data, (int, float)):
                current_strength_val = float(strength_val_from_data)

            if current_tone_enum_member and current_tone_enum_member not in excluded_tones_for_eti:
                if last_valid_tone_enum: # If there's a previous valid point to compare against
                    num_comparable_transitions += 1
                    is_tone_category_changed = (current_tone_enum_member != last_valid_tone_enum)
                    strength_difference = abs(current_strength_val - last_valid_strength)
                    
                    # An inflexion point is where either the tone category changes,
                    # or the strength 변화が顕著
                    if is_tone_category_changed or strength_difference >= SIGNIFICANT_STRENGTH_CHANGE_THRESHOLD:
                        inflexion_event_count += 1
                        self.logger.debug(f"  ETI: 転換点候補 (インデックス {i}): 前={last_valid_tone_enum.value if last_valid_tone_enum else 'N/A'}(強度:{last_valid_strength:.2f}) -> 現={current_tone_enum_member.value}(強度:{current_strength_val:.2f}). カテゴリ変化:{is_tone_category_changed}, 強度差:{strength_difference:.2f}")
                
                last_valid_tone_enum = current_tone_enum_member
                last_valid_strength = current_strength_val
        
        if num_comparable_transitions == 0:
            self.logger.debug("ETI: 有効な感情遷移（比較可能なポイント間）がなかったため、デフォルトスコア0.4を返します。")
            return 0.4

        # Score is the ratio of inflexion points to the number of possible transitions.
        eti_score_value = inflexion_event_count / num_comparable_transitions
        final_eti_score = round(max(0.0, min(1.0, eti_score_value)), 4)
        self.logger.info(f"ETI計算完了: 転換点イベント数={inflexion_event_count}, 比較可能遷移数={num_comparable_transitions}, Score={final_eti_score:.4f}")
        return final_eti_score

    def compute_symbolic_density(self, dialogue_text: str) -> float:
        # Calculates density based on configured keywords.
        self.logger.debug("compute_symbolic_density を計算します。")
        keyword_list: List[str] = []
        loaded_external_configs_obj = getattr(self.config, 'loaded_external_configs', None)
        if loaded_external_configs_obj:
            raw_keyword_list = getattr(loaded_external_configs_obj, 'symbolic_density_keywords', [])
            if isinstance(raw_keyword_list, list) and all(isinstance(k_word, str) for k_word in raw_keyword_list):
                keyword_list = raw_keyword_list
            else:
                self.logger.warning("Symbolic Density: 設定ファイル ('symbolic_density_keywords') からのキーワードリストが無効な形式（リストオブストリングス期待）。")

        if not dialogue_text or not dialogue_text.strip(): # Check for empty or whitespace-only text
             self.logger.debug("Symbolic Density: 対話テキストが空のため、スコア0.0を返します。")
             return 0.0
        if not keyword_list:
            self.logger.debug("Symbolic Density: 計算に使用するキーワードが設定されていません。スコア0.2（基本値）を返します。")
            return 0.2 # Default if no keywords are defined but text exists

        text_content_lower = dialogue_text.lower()
        total_keyword_occurrences = sum(text_content_lower.count(kw.lower()) for kw in keyword_list)
        
        # Normalize by text length (e.g., occurrences per 1000 characters) and then scale to 0-1.
        # The scaling factor (e.g., / 10.0) is heuristic and may need tuning based on typical keyword frequencies.
        density_score_value = 0.0
        text_length_chars = len(dialogue_text)
        if text_length_chars > 0:
            occurrences_per_1k_chars = (total_keyword_occurrences / text_length_chars) * 1000
            # Example scaling: 10 occurrences per 1k characters might correspond to a score of 1.0
            density_score_value = round(min(1.0, max(0.0, occurrences_per_1k_chars / 10.0)), 4)
        
        self.logger.info(f"Symbolic Density: キーワード総出現回数={total_keyword_occurrences}, Score={density_score_value:.4f}")
        return density_score_value

    def compute_content_novelty(self) -> float:
        # Content Novelty - Placeholder. True novelty requires comparison with a corpus or history.
        self.logger.debug("compute_content_novelty (簡易): 現在は固定値を返します。このメトリクスの詳細な実装には、過去の生成内容や外部コーパスとの比較など、より複雑なメカニズムが必要です。")
        return 0.65 # Placeholder fixed value, indicating moderate novelty by default.

    def compute_expression_richness(self) -> float:
        # Calculates richness based on Type-Token Ratio (TTR).
        self.logger.debug("compute_expression_richness を計算します (TTRベースアプローチ)。")
        # self.dialogue_text should be set by super().__init__() in get_dfrs_scores_v49
        if not hasattr(self, 'dialogue_text') or not self.dialogue_text or not self.dialogue_text.strip():
            self.logger.debug("Expression Richness: 対話テキストが空または未設定のため、スコア0.0を返します。")
            return 0.0

        # Simple word tokenization: find sequences of word characters.
        words_found = re.findall(r'\b\w+\b', self.dialogue_text.lower())
        
        if not words_found:
            self.logger.debug("Expression Richness: テキストから単語が抽出できませんでした。スコア0.0を返します。")
            return 0.0

        num_total_words = len(words_found)
        num_unique_words = len(set(words_found))
        
        # Calculate Type-Token Ratio
        ttr_value = num_unique_words / num_total_words if num_total_words > 0 else 0.0
        
        # Map TTR to a 0-1 score. This mapping is heuristic and can be refined.
        # TTR is sensitive to text length (shorter texts tend to have higher TTR).
        # A more sophisticated approach might normalize TTR based on text length.
        # For now, a piecewise linear mapping:
        if ttr_value >= 0.7: richness_score = 1.0
        elif ttr_value >= 0.6: richness_score = 0.8 + (ttr_value - 0.6) * 2.0 # Maps [0.6, 0.7) to [0.8, 1.0)
        elif ttr_value >= 0.5: richness_score = 0.6 + (ttr_value - 0.5) * 2.0 # Maps [0.5, 0.6) to [0.6, 0.8)
        elif ttr_value >= 0.4: richness_score = 0.4 + (ttr_value - 0.4) * 2.0 # Maps [0.4, 0.5) to [0.4, 0.6)
        elif ttr_value >= 0.3: richness_score = 0.2 + (ttr_value - 0.3) * 2.0 # Maps [0.3, 0.4) to [0.2, 0.4)
        else: richness_score = ttr_value * (0.2 / 0.3) # Maps [0.0, 0.3) to [0.0, 0.2)
        
        final_richness_score = round(max(0.0, min(1.0, richness_score)), 4)
        self.logger.info(f"Expression Richness: TTR={ttr_value:.4f} (総単語数: {num_total_words}, ユニーク単語数: {num_unique_words}), Score={final_richness_score:.4f}")
        return final_richness_score

    def analyze_dialogue_rhythm(self) -> Dict[str, Optional[float]]:
        self.logger.debug("analyze_dialogue_rhythm を計算します。")
        # Default scores (on a 0-1 scale before final 0-5 scaling)
        rhythm_consistency_score = 0.5
        rhythm_variability_score = 0.5

        if not hasattr(self,'block_sequence'):
            self.logger.error("リズム分析: block_sequenceがDialogFlowEvaluatorBaseで初期化されていません。デフォルト値を返します。")
            # Fallback to creating empty rhythm results if DFRSMetrics_cls is missing
            return { "rhythm_consistency_default": rhythm_consistency_score * 5.0,
                     "rhythm_variability_default": rhythm_variability_score * 5.0 }

        block_lengths_list = [length for _, length in self.block_sequence if length > 0]
        if len(block_lengths_list) >= 2:
            try:
                mean_block_length = statistics.mean(block_lengths_list)
                std_dev_block_length = statistics.stdev(block_lengths_list) # Requires at least 2 data points
                
                # Coefficient of Variation for block lengths
                cv_block_length = (std_dev_block_length / mean_block_length) if mean_block_length > 1e-9 else 0.0
                
                # Consistency: Lower CV (lengths are more similar) implies higher consistency.
                # Using exp(-cv * factor) to map CV to a 0-1 consistency score.
                # Factor (e.g., 0.8) adjusts sensitivity.
                rhythm_consistency_score = math.exp(-cv_block_length * 0.8)
                # Variability is calculated by calculate_pacing_variability (already 0-1)
                rhythm_variability_score = self.calculate_pacing_variability()
            except statistics.StatisticsError as e_stat_rhythm:
                self.logger.warning(f"リズム分析の統計計算中にエラー ({e_stat_rhythm})。デフォルト値を使用します。ブロック数: {len(block_lengths_list)}")
            except Exception as e_rhythm_calc:
                self.logger.error(f"リズム分析中に予期せぬエラーが発生しました: {e_rhythm_calc}", exc_info=True)
        
        rhythm_results_output_map: Dict[str, Optional[float]] = {}
        DFRS_cls_local_rhythm = getattr(self, 'DFRSMetrics_cls', None)
        if DFRS_cls_local_rhythm:
            # Use Enum member's .value for dictionary keys
            rc_key_str = getattr(DFRS_cls_local_rhythm.RHYTHM_CONSISTENCY, 'value', 'rhythm_consistency_fallback_key')
            rv_key_str = getattr(DFRS_cls_local_rhythm.RHYTHM_VARIABILITY, 'value', 'rhythm_variability_fallback_key')
            
            # Scores are calculated on a 0-1 scale, then scaled to 0-5 for final reporting
            rhythm_results_output_map[rc_key_str] = round(rhythm_consistency_score * 5.0, 3)
            rhythm_results_output_map[rv_key_str] = round(rhythm_variability_score * 5.0, 3)
            self.logger.info(f"リズム分析結果 (0-5スケール): Consistency Score ({rc_key_str}) = {rhythm_results_output_map.get(rc_key_str, 'N/A')}, Variability Score ({rv_key_str}) = {rhythm_results_output_map.get(rv_key_str, 'N/A')}")
        else:
            self.logger.error("リズム分析: DFRSMetrics_clsが未定義のため、結果キーがフォールバック名になります。")
            # Fallback keys if Enum is not available
            rhythm_results_output_map["rhythm_consistency_fallback_key"] = round(rhythm_consistency_score * 5.0, 3)
            rhythm_results_output_map["rhythm_variability_fallback_key"] = round(rhythm_variability_score * 5.0, 3)
            
        return rhythm_results_output_map

    def get_dfrs_scores_v49(
        self, dialogue_text: Optional[str]=None, analyzer_results: Optional[Dict[str, Any]]=None,
        intended_phase: Optional['PsychologicalPhaseV49']=None, intended_tone: Optional['EmotionalToneV49']=None # type: ignore
    ) -> Dict[str, Optional[float]]:
        
        text_to_be_evaluated: str
        if isinstance(dialogue_text, str) and dialogue_text.strip():
            text_to_be_evaluated = dialogue_text.strip()
        elif isinstance(analyzer_results, dict): # Try to reconstruct text from analyzer_results if dialogue_text is not provided
            output_dialogue_content_dict = analyzer_results.get("output_dialogue_content", {})
            dialogue_blocks_list_of_dicts = self._safely_extract_list(output_dialogue_content_dict, "blocks")
            text_to_be_evaluated = "\n".join(str(block_dict.get("text", "")) for block_dict in dialogue_blocks_list_of_dicts).strip()
            if not text_to_be_evaluated:
                 self.logger.warning("DFRS評価: dialogue_textが提供されず、analyzer_resultsからも有効なテキストを再構築できませんでした。")
        else:
            text_to_be_evaluated = ""

        if not text_to_be_evaluated:
            self.logger.error("DFRS評価: 有効な評価対象テキストがありません。空のスコア辞書を返します。")
            return {}
        
        # Initialize DialogFlowEvaluatorBase with the determined text for structure analysis
        super().__init__(text_to_be_evaluated)

        DFRSMetrics_cls_instance:Optional[Type['DFRSMetricsV49']]=getattr(self,'DFRSMetrics_cls',None) # type: ignore
        if not DFRSMetrics_cls_instance:
            self.logger.critical("DFRSMetricsV49 Enumクラスがロードされていません。DFRS評価は実行できません。空のスコア辞書を返します。")
            return {}
        
        # Extract emotion and phase data from analyzer_results if available
        current_analyzer_results_dict_local = analyzer_results if isinstance(analyzer_results, dict) else {}
        evaluation_metrics_partial_data = current_analyzer_results_dict_local.get("evaluation_metrics_partial", {})
        emotion_curve_data_list_local = self._safely_extract_list(evaluation_metrics_partial_data, "emotion_curve")
        phase_timeline_data_list_local = self._safely_extract_list(evaluation_metrics_partial_data, "phase_timeline")

        # Cache handling
        cache_key_identifier: Optional[str] = None
        should_use_cache = self.dfrs_cache is not None # self.dfrs_cache is already Optional PersistentCache type
        
        if should_use_cache and self.dfrs_cache:
            try:
                # Create a comprehensive cache key
                weights_map_for_key_gen = {key.value: val for key, val in self.weights.items() if hasattr(key, 'value')}
                weights_json_str = json.dumps(weights_map_for_key_gen, sort_keys=True)
                weights_component_hash = hashlib.sha256(weights_json_str.encode('utf-8')).hexdigest()[:8]
                
                text_component_hash = hashlib.sha256(self.dialogue_text.encode('utf-8')).hexdigest()[:12] # Use self.dialogue_text set by super()
                
                intended_phase_str_val = intended_phase.value if intended_phase and hasattr(intended_phase, 'value') else "NotSet"
                intended_tone_str_val = intended_tone.value if intended_tone and hasattr(intended_tone, 'value') else "NotSet"
                context_component_hash = f"phase_{intended_phase_str_val}_tone_{intended_tone_str_val}"
                
                # Include a version identifier in the cache key namespace
                cache_key_identifier = f"dfrs_scores_v49_final_v5_part7:{weights_component_hash}:{context_component_hash}:{text_component_hash}"
                
                cached_scores_from_db = self.dfrs_cache.get(cache_key_identifier)
                if isinstance(cached_scores_from_db, dict):
                    self.logger.info(f"DFRS Cache HIT: Key='{cache_key_identifier}' からスコアをロードしました。")
                    # Ensure all expected keys are present, or re-calculate if schema changed (optional)
                    return cached_scores_from_db
                else:
                    self.logger.info(f"DFRS Cache MISS: Key='{cache_key_identifier}'. 新規にスコアを計算します。")
            except Exception as e_cache_read:
                self.logger.warning(f"DFRSキャッシュからの読み込み中にエラーが発生しました (Key: {cache_key_identifier}): {e_cache_read}", exc_info=False) # exc_info=False to keep log concise

        # Initialize map for scores (on a 0-1 scale for internal calculation)
        calculated_scores_0_to_1_scale: Dict['DFRSMetricsV49', Optional[float]] = {} # type: ignore
        
        # Calculate rhythm scores once (analyze_dialogue_rhythm returns scores on 0-5 scale)
        rhythm_scores_0_to_5_scale_map = self.analyze_dialogue_rhythm()
        # Get .value for keys to match how rhythm_scores_0_to_5_scale_map keys are (potentially) stored
        rc_metric_key_str = getattr(DFRSMetrics_cls_instance.RHYTHM_CONSISTENCY, 'value', 'rhythm_consistency_fallback_key')
        rv_metric_key_str = getattr(DFRSMetrics_cls_instance.RHYTHM_VARIABILITY, 'value', 'rhythm_variability_fallback_key')

        # Define mapping from DFRSMetrics Enum members to their calculation functions
        # Note: Ensure all calculation functions return a float between 0.0 and 1.0, or None
        metric_calculation_function_map: Dict['DFRSMetricsV49', Callable[[], Optional[float]]] = { # type: ignore
            DFRSMetrics_cls_instance.CSR: self.calculate_continuous_speech_ratio,
            DFRSMetrics_cls_instance.DIT: self.calculate_description_insertion_timing,
            DFRSMetrics_cls_instance.PVS: self.calculate_pacing_variability,
            DFRSMetrics_cls_instance.DDB: self.calculate_dialogue_description_balance,
            DFRSMetrics_cls_instance.TS: self.calculate_transition_smoothness,
            DFRSMetrics_cls_instance.PHASE_ALIGNMENT: lambda: self._calculate_phase_tone_alignment(intended_phase, intended_tone, phase_timeline_data_list_local, None)[0],
            DFRSMetrics_cls_instance.TONE_ALIGNMENT: lambda: self._calculate_phase_tone_alignment(intended_phase, intended_tone, [], emotion_curve_data_list_local)[1],
            DFRSMetrics_cls_instance.PTN: lambda: self.calculate_phase_transition_naturalness(phase_timeline_data_list_local),
            DFRSMetrics_cls_instance.ECS: lambda: self.calculate_emotion_curve_smoothness(emotion_curve_data_list_local),
            # Rhythm scores need to be scaled back from 0-5 to 0-1 for internal calculation before weighting
            DFRSMetrics_cls_instance.RHYTHM_CONSISTENCY: lambda: (rhythm_scores_0_to_5_scale_map.get(rc_metric_key_str, 2.5) or 0.0) / 5.0 if rhythm_scores_0_to_5_scale_map else 0.5,
            DFRSMetrics_cls_instance.RHYTHM_VARIABILITY: lambda: (rhythm_scores_0_to_5_scale_map.get(rv_metric_key_str, 2.5) or 0.0) / 5.0 if rhythm_scores_0_to_5_scale_map else 0.5,
            DFRSMetrics_cls_instance.SUBJECTIVITY_SCORE: self.compute_subjectivity_score,
            DFRSMetrics_cls_instance.FLUCTUATION_INTENSITY: self.compute_fluctuation_intensity,
            DFRSMetrics_cls_instance.INTERNAL_DEPTH: lambda: self.compute_internal_depth(current_analyzer_results_dict_local),
            DFRSMetrics_cls_instance.EMOTION_COMPLEXITY: lambda: self.compute_emotion_complexity(emotion_curve_data_list_local),
            DFRSMetrics_cls_instance.ETI: lambda: self.compute_eti(emotion_curve_data_list_local), # Pass data
            DFRSMetrics_cls_instance.SYMBOLIC_DENSITY: lambda: self.compute_symbolic_density(self.dialogue_text), # Uses self.dialogue_text
            DFRSMetrics_cls_instance.CONTENT_NOVELTY: self.compute_content_novelty,
            DFRSMetrics_cls_instance.EXPRESSION_RICHNESS: self.compute_expression_richness, # Uses self.dialogue_text
        }

        # Calculate each DFRS metric
        for metric_enum in DFRSMetrics_cls_instance: # type: ignore
            if metric_enum in [DFRSMetrics_cls_instance.UNKNOWN, DFRSMetrics_cls_instance.FINAL_EODF_V49]: # type: ignore
                continue # These are not directly calculated or are aggregates
            
            calculation_method = metric_calculation_function_map.get(metric_enum)
            if calculation_method and callable(calculation_method):
                try:
                    raw_score_0_1 = calculation_method()
                    if isinstance(raw_score_0_1, (int, float)):
                        # Ensure score is clamped between 0.0 and 1.0
                        calculated_scores_0_to_1_scale[metric_enum] = round(max(0.0, min(1.0, float(raw_score_0_1))), 4)
                    else:
                        calculated_scores_0_to_1_scale[metric_enum] = None # Mark as None if calculation didn't return a number
                        self.logger.debug(f"DFRS指標 '{metric_enum.value}' の計算結果が数値ではありません (型: {type(raw_score_0_1)})。この指標はeODF計算から除外されます。")
                except Exception as e_metric_calc:
                    self.logger.error(f"DFRS指標 '{metric_enum.value}' の計算中にエラーが発生しました: {e_metric_calc}", exc_info=True)
                    calculated_scores_0_to_1_scale[metric_enum] = None # Mark as None on error
            else:
                calculated_scores_0_to_1_scale[metric_enum] = None # No calculation function defined
                self.logger.debug(f"DFRS指標 '{metric_enum.value}' の計算関数が定義されていません。この指標はeODF計算から除外されます。")
        
        # Calculate the final eODF score (weighted average, on 0-1 scale)
        eodf_weighted_sum = 0.0
        eodf_total_applied_weight = 0.0
        eodf_missing_score_details = [] # For logging which scores were missing for weighted sum

        for metric_for_eodf_calc, weight_value in self.weights.items():
            score_for_eodf_calc = calculated_scores_0_to_1_scale.get(metric_for_eodf_calc)
            if score_for_eodf_calc is not None and weight_value > 0: # Only consider valid scores and positive weights
                eodf_weighted_sum += max(0.0, min(1.0, score_for_eodf_calc)) * weight_value # Clamp score again just in case
                eodf_total_applied_weight += weight_value
            elif weight_value > 0: # Score is None but weight is positive, log this
                eodf_missing_score_details.append(f"{metric_for_eodf_calc.value} (weight:{weight_value:.2f})")
        
        final_eodf_score_0_1_scale = (eodf_weighted_sum / eodf_total_applied_weight) if eodf_total_applied_weight > 1e-9 else 0.0
        calculated_scores_0_to_1_scale[DFRSMetrics_cls_instance.FINAL_EODF_V49] = round(max(0.0, min(1.0, final_eodf_score_0_1_scale)), 4) # type: ignore
        
        if eodf_missing_score_details:
            self.logger.warning(f"eODF (0-1スケール) 計算: {len(eodf_missing_score_details)}個の指標のスコアが不足していたため、加重合計から除外されました: {', '.join(eodf_missing_score_details)}。")
        self.logger.info(f"eODF (0-1スケール) 計算詳細: 加重スコア合計={eodf_weighted_sum:.4f}, 適用された総重み={eodf_total_applied_weight:.4f}, 最終eODFスコア(0-1)={calculated_scores_0_to_1_scale.get(DFRSMetrics_cls_instance.FINAL_EODF_V49,0.0):.4f}") # type: ignore
        
        # Convert all scores to 0-5 scale for the final output dictionary, using Enum member's .value as key
        final_output_scores_0_to_5_scale = {
            metric_enum.value: round(score_0_1 * 5.0, 3) if score_0_1 is not None else None
            for metric_enum, score_0_1 in calculated_scores_0_to_1_scale.items() if hasattr(metric_enum, 'value')
        }
        
        # Cache the calculated scores (0-5 scale)
        if should_use_cache and cache_key_identifier and self.dfrs_cache:
            valid_scores_for_caching = {key_str: val_num for key_str, val_num in final_output_scores_0_to_5_scale.items() if val_num is not None}
            if valid_scores_for_caching: # Only cache if there are some valid scores to store
                try:
                    self.dfrs_cache.set(cache_key_identifier, valid_scores_for_caching)
                    self.logger.info(f"DFRSスコアをキャッシュに保存しました (Key: {cache_key_identifier})。")
                except Exception as e_cache_write:
                    self.logger.warning(f"DFRSスコアのキャッシュ保存中にエラーが発生しました (Key: {cache_key_identifier}): {e_cache_write}", exc_info=False)
            else:
                self.logger.debug(f"DFRS: 有効なスコアがなかったため、キャッシュへの保存はスキップされました (Key: {cache_key_identifier})。")

        final_eodf_metric_key_str = getattr(DFRSMetrics_cls_instance.FINAL_EODF_V49, 'value', 'final_eodf_v49_fallback_key')
        self.logger.info(f"DFRS評価完了 (0-5スケール)。eODFスコア ({final_eodf_metric_key_str}) = {final_output_scores_0_to_5_scale.get(final_eodf_metric_key_str, 'N/A')}")
        return final_output_scores_0_to_5_scale

    def _calculate_phase_tone_alignment(
        self, intended_phase_enum_param: Optional['PsychologicalPhaseV49'], intended_tone_enum_param: Optional['EmotionalToneV49'], # type: ignore
        phase_timeline_data_param: List[Dict[str, Any]], emotion_curve_data_param: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[float, float]:
        
        # Default alignment scores (0.0 to 1.0)
        phase_alignment_final_score, tone_alignment_final_score = 0.5, 0.5
        
        PP_cls_local_pta, ET_cls_local_pta = getattr(self,'PsychologicalPhaseV49_cls',None), getattr(self,'EmotionalToneV49_cls',None)

        if not (PP_cls_local_pta and ET_cls_local_pta):
            self.logger.error("Phase/Tone Alignment: 必須Enumクラス(PsychologicalPhaseV49 or EmotionalToneV49)が未ロードです。デフォルトスコア(0.3, 0.3)を返します。")
            return 0.3, 0.3
        
        logger_pta_calc = self.logger.getChild("_calculate_phase_tone_alignment_v5_part7") # Version updated

        # Helper to find the dominant Enum member from a timeline data list
        def get_dominant_enum_from_timeline_data(
            timeline_data_list: Optional[List[Dict[str, Any]]],
            target_enum_class_ref: Type[enum.Enum],
            data_key_in_dict: str
        ) -> Optional[enum.Enum]:
            if not timeline_data_list: return None # No data, no dominant member
            
            valid_enum_members_in_timeline: List[enum.Enum] = []
            # Get the UNKNOWN member of the specific Enum to exclude it from dominance calculation
            unknown_member_for_exclusion = getattr(target_enum_class_ref, 'UNKNOWN', None)

            for item_dictionary in timeline_data_list:
                if not isinstance(item_dictionary, dict): continue # Skip malformed entries
                
                enum_value_from_data = item_dictionary.get(data_key_in_dict)
                # Convert the string/value from data to an actual Enum member
                member_instance_from_data = self._get_enum_member_from_value(target_enum_class_ref, enum_value_from_data)
                
                # Add to list if it's a valid member and not the UNKNOWN member
                if member_instance_from_data and member_instance_from_data != unknown_member_for_exclusion:
                    valid_enum_members_in_timeline.append(member_instance_from_data)
            
            if not valid_enum_members_in_timeline: return None # No valid (non-UNKNOWN) members found
            
            # Find the most common valid Enum member
            dominant_candidate_list_result = TypingCounter(valid_enum_members_in_timeline).most_common(1)
            if dominant_candidate_list_result:
                dominant_member = dominant_candidate_list_result[0][0]
                logger_pta_calc.debug(f"Dominant {target_enum_class_ref.__name__} from timeline data (key: '{data_key_in_dict}'): '{getattr(dominant_member,'value','N/A')}'")
                return dominant_member
            return None # Should not happen if valid_enum_members_in_timeline is not empty

        # --- Phase Alignment Calculation ---
        dominant_actual_phase_enum = get_dominant_enum_from_timeline_data(phase_timeline_data_param, PP_cls_local_pta, "phase") # type: ignore
        unknown_pp_instance_val = getattr(PP_cls_local_pta, 'UNKNOWN', None)
        
        # Case 1: A specific phase is intended (and it's not UNKNOWN)
        if intended_phase_enum_param and intended_phase_enum_param != unknown_pp_instance_val:
            # Case 1a: A specific phase was also dominant in the actual timeline
            if dominant_actual_phase_enum and dominant_actual_phase_enum != unknown_pp_instance_val:
                phase_alignment_final_score = 1.0 if intended_phase_enum_param == dominant_actual_phase_enum else 0.0
            else: # Intended specific, but actual dominant phase was UNKNOWN or not found (low alignment)
                phase_alignment_final_score = 0.25
        # Case 2: No specific phase intended (or UNKNOWN was intended)
        elif not intended_phase_enum_param or intended_phase_enum_param == unknown_pp_instance_val:
            # Case 2a: Actual dominant phase was also UNKNOWN or not found (good alignment for non-specific intent)
            if not dominant_actual_phase_enum or dominant_actual_phase_enum == unknown_pp_instance_val:
                phase_alignment_final_score = 0.75
            else: # No specific intent, but a specific phase became dominant (moderate misalignment)
                phase_alignment_final_score = 0.4
        
        # --- Tone Alignment Calculation ---
        dominant_actual_tone_enum = get_dominant_enum_from_timeline_data(emotion_curve_data_param, ET_cls_local_pta, "tone") # type: ignore
        unknown_et_instance_val = getattr(ET_cls_local_pta, 'UNKNOWN', None)
        neutral_et_instance_val = getattr(ET_cls_local_pta, 'NEUTRAL', None)
        # Tones that are generally not considered strong emotional intents
        tones_excluded_from_strong_intent = {unknown_et_instance_val, neutral_et_instance_val}
        tones_excluded_from_strong_intent.discard(None) # Remove None if members don't exist

        # Case 1: A specific, non-neutral/unknown tone is intended
        if intended_tone_enum_param and intended_tone_enum_param not in tones_excluded_from_strong_intent:
            # Case 1a: A specific, non-neutral/unknown tone was also dominant
            if dominant_actual_tone_enum and dominant_actual_tone_enum not in tones_excluded_from_strong_intent:
                tone_alignment_final_score = 1.0 if intended_tone_enum_param == dominant_actual_tone_enum else 0.0
            else: # Specific emotional tone intended, but actual dominant was neutral/unknown or not found (low alignment)
                tone_alignment_final_score = 0.25
        # Case 2: Neutral/Unknown or no specific emotional tone intended
        elif not intended_tone_enum_param or intended_tone_enum_param in tones_excluded_from_strong_intent:
            # Case 2a: Actual dominant tone was also neutral/unknown/not found (good alignment for non-emotional intent)
            if not dominant_actual_tone_enum or dominant_actual_tone_enum in tones_excluded_from_strong_intent:
                tone_alignment_final_score = 0.75
            else: # Neutral/Unknown intent, but a specific emotional tone became dominant (moderate misalignment)
                tone_alignment_final_score = 0.4
            
        logger_pta_calc.info(f"Phase/Tone Alignment Scores (0-1): "
                             f"Phase (Intended: '{getattr(intended_phase_enum_param,'value','N/A')}', Dominant Actual: '{getattr(dominant_actual_phase_enum,'value','N/A')}') => {phase_alignment_final_score:.2f}. "
                             f"Tone (Intended: '{getattr(intended_tone_enum_param,'value','N/A')}', Dominant Actual: '{getattr(dominant_actual_tone_enum,'value','N/A')}') => {tone_alignment_final_score:.2f}")
        return round(phase_alignment_final_score, 4), round(tone_alignment_final_score, 4)

# =============================================================================
# -- Part 7 終了点 (EnhancedDialogFlowEvaluatorV49 クラス定義終了)
# =============================================================================
# =============================================================================
# -- Part 7b: Advanced Dialogue Analyzer (Optimized & Refined Version - v2.2)
# =============================================================================
# PsychologicalPhaseV49 Enum (11コアメンバー) への移行に伴う必要な修正は
# 既に適用済みであることを前提とします。
# 最も重要なのは、外部設定ファイル analyzer_keywords_v49.yaml の
# phase_keywords のキーが、最新Enumの .value 文字列に準拠していることです。
# v2.2変更点:
# - ロガー名を .v2.2 に更新。
# - analyze_and_get_results: 開始・終了ログを強化し、入力識別情報と結果サマリーを含むように。Docstringを更新。
# - __init__: キーワードマップロード失敗時のログを明確化。

from typing import TYPE_CHECKING, Counter as TypingCounter, Set, List, Dict, Optional, Tuple, Callable, Type, Union, Any, DefaultDict
import enum
import pathlib # Not directly used in this Part, but often part of a larger system
import hashlib # Not directly used in this Part (but could be for hashing text for logging)
import json    # Not directly used in this Part
import math    # Not directly used in this Part
import statistics # Not directly used in this Part
import re
import logging
import traceback # For detailed error logging
from collections import defaultdict # Ensure defaultdict is available

# グローバルスコープで利用可能であることを期待する変数 (Part 0 で定義)
# これらはスクリプトの先頭部分で適切にインポートまたは定義されている必要があります。

# ---------- type-checking-only imports ----------
if TYPE_CHECKING:
    from typing import TypeAlias
    # numpy は Part 0 で np としてインポートされていることを想定
    from numpy import ndarray as np_ndarray # type: ignore

    # --- プロジェクト内部型 (Part 0, 1b, 3, 4a で定義・インポート済みと仮定) ---
    from __main__ import ( # type: ignore[attr-defined]
        ConfigProtocol, ScorerProtocol, BaseModel, ValidationError, # Pydantic関連 (Part 0, 4a)
        PsychologicalPhaseV49, EmotionalToneV49, SubjectivityCategoryV49, FluctuationCategoryV49, # Enums (Part 1b)
        DialogueBlockV49, SpeechBlockV49, DescriptionBlockV49, # Pydanticモデル (Part 3c)
        BlockAnalysisTagsV49, EmotionCurvePointV49, PhaseTimelinePointV49, # Pydanticモデル (Part 3c)
        PYDANTIC_AVAILABLE, SPACY_AVAILABLE, spacy, # Part 0 で定義されるフラグとライブラリ
        AppConfigV49, FileSettingsV49 # Part 2, Part 3a で定義
    )
    # 型エイリアス
    DialogueBlockModelType_hint: TypeAlias = DialogueBlockV49
    SpeechBlockModelType_hint: TypeAlias = SpeechBlockV49
    DescriptionBlockModelType_hint: TypeAlias = DescriptionBlockV49
    BlockAnalysisTagsModelType_hint: TypeAlias = BlockAnalysisTagsV49
    EmotionCurvePointModelType_hint: TypeAlias = EmotionCurvePointV49
    PhaseTimelinePointModelType_hint: TypeAlias = PhaseTimelinePointV49
    PsychologicalPhaseV49EnumType_hint: TypeAlias = PsychologicalPhaseV49
    EmotionalToneV49EnumType_hint: TypeAlias = EmotionalToneV49
else:
    # 実行時は文字列リテラルとして定義 (前方参照のため)
    ConfigProtocol = 'ConfigProtocol'
    ScorerProtocol = 'ScorerProtocol'
    # PydanticモデルやEnumは __init__ で globals() から取得される想定を維持

class AdvancedDialogueAnalyzerV49: # Implicitly implements AnalyzerProtocol (defined in Part 4a)
    """
    対話テキストの詳細分析を行うクラス (v2.2)。
    構造解析、NLP特徴抽出、感情・心理的位相の推定などを行います。
    """
    if TYPE_CHECKING:
        ConfigProto_analyzer: TypeAlias = ConfigProtocol
        ScorerProto_analyzer: TypeAlias = ScorerProtocol
        # Enum クラスの型ヒント
        EmotionalTone_cls_type: Type[EmotionalToneV49]
        PsychologicalPhase_cls_type: Type[PsychologicalPhaseV49]
        # Pydantic モデルクラスの型ヒント
        DialogueBlock_cls_type: Type[DialogueBlockV49]
        SpeechBlock_cls_type: Type[SpeechBlockV49]
        DescriptionBlock_cls_type: Type[DescriptionBlockV49]
        BlockAnalysisTags_cls_type: Type[BlockAnalysisTagsV49]
        EmotionCurvePoint_cls_type: Type[EmotionCurvePointV49]
        PhaseTimelinePoint_cls_type: Type[PhaseTimelinePointV49]
        PydanticValidationError_cls_type: Optional[Type[ValidationError]]
    else:
        ConfigProto_analyzer = 'ConfigProtocol'
        ScorerProto_analyzer = 'ScorerProtocol'
        # Actual class references will be populated in __init__

    def __init__(self, config: ConfigProto_analyzer, scorer: ScorerProto_analyzer): # type: ignore
        self.config = config
        self.scorer = scorer # This is expected to be an instance of SubjectivityFluctuationScorerV49 or similar
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}.v2.2") # Version updated
        system_version_for_log = getattr(self.config, 'SYSTEM_VERSION', 'N/A_AdvAnalyzer_v2.2')
        self.logger.info(f"AdvancedDialogueAnalyzerV49 (System Version: {system_version_for_log}) の初期化を開始します...")

        # --- EnumクラスとPydanticモデルクラスのロード ---
        # This loading mechanism relies on these classes being in the global scope.
        required_classes_to_load_map = {
            'EmotionalToneV49': 'EmotionalTone_cls',
            'PsychologicalPhaseV49': 'PsychologicalPhase_cls',
            'DialogueBlockV49': 'DialogueBlock_cls',
            'SpeechBlockV49': 'SpeechBlock_cls',
            'DescriptionBlockV49': 'DescriptionBlock_cls',
            'BlockAnalysisTagsV49': 'BlockAnalysisTags_cls',
            'EmotionCurvePointV49': 'EmotionCurvePoint_cls',
            'PhaseTimelinePointV49': 'PhaseTimelinePoint_cls',
            'ValidationError': 'PydanticValidationError_cls' # Pydantic利用可能時のみ
        }
        missing_cls_names_init: List[str] = []

        for class_name_str, attr_name_to_set_str in required_classes_to_load_map.items():
            if class_name_str == 'ValidationError' and not globals().get('PYDANTIC_AVAILABLE'):
                setattr(self, attr_name_to_set_str, None)
                continue
            try:
                loaded_class_obj = globals()[class_name_str]
                setattr(self, attr_name_to_set_str, loaded_class_obj)
            except KeyError:
                missing_cls_names_init.append(class_name_str)
                setattr(self, attr_name_to_set_str, None) # 属性が存在しないエラーを防ぐためNoneを設定

        if missing_cls_names_init:
            critical_message = f"CRITICAL: AdvancedDialogueAnalyzer初期化エラー: 必須クラス {', '.join(missing_cls_names_init)} がグローバルスコープに見つかりません。スクリプトのロード順序や定義を確認してください。"
            self.logger.critical(critical_message)
            # ImportErrorを発生させることで、このコンポーネントが利用不可であることを上位に伝える
            raise ImportError(critical_message)
        else:
            self.logger.debug("必須のEnumおよびPydanticモデルクラスのグローバル参照に成功しました。")


        # Initialize keyword maps
        self.emotion_keywords_map: DefaultDict['EmotionalToneV49', List[str]] = defaultdict(list) # type: ignore
        self.phase_keywords_map: DefaultDict['PsychologicalPhaseV49', List[str]] = defaultdict(list) # type: ignore

        # Load keywords from config (e.g., from analyzer_keywords_v49.yaml via AppConfigV49)
        analyzer_keywords_data_dict = getattr(self.config, 'analyzer_keywords_data', None)

        if isinstance(analyzer_keywords_data_dict, dict):
            # --- Load emotion keywords ---
            raw_emotion_keywords_from_cfg = analyzer_keywords_data_dict.get("emotion_keywords")
            if isinstance(raw_emotion_keywords_from_cfg, dict) and self.EmotionalTone_cls:
                self.logger.debug(f"感情キーワードのロード処理開始 (emotion_keywords)。ロード元カテゴリ数: {len(raw_emotion_keywords_from_cfg)}")
                for tone_key_str_from_yaml, keywords_list_from_yaml in raw_emotion_keywords_from_cfg.items():
                    cleaned_tone_key_str = str(tone_key_str_from_yaml).strip()
                    try:
                        tone_enum_member = self.EmotionalTone_cls(cleaned_tone_key_str)
                        if isinstance(keywords_list_from_yaml, list) and all(isinstance(kw_item, str) for kw_item in keywords_list_from_yaml):
                            valid_keywords_for_tone = [kw.strip().lower() for kw in keywords_list_from_yaml if kw.strip()]
                            if valid_keywords_for_tone:
                                self.emotion_keywords_map[tone_enum_member].extend(valid_keywords_for_tone)
                                self.logger.debug(f"  感情カテゴリ '{tone_enum_member.value}': {len(valid_keywords_for_tone)}個のキーワードをロード ({', '.join(valid_keywords_for_tone[:3])}...)")
                        else:
                            self.logger.warning(f"感情キーワードデータで、キー '{cleaned_tone_key_str}' の値が不正な形式です (文字列のリストを期待しましたが、型は {type(keywords_list_from_yaml)})。このキーはスキップされます。")
                    except ValueError:
                        self.logger.warning(f"感情キーワードデータのキー '{cleaned_tone_key_str}' は EmotionalToneV49 Enum の有効な値ではありません。このキーはスキップされます。YAML設定を確認してください。")
                    except Exception as e_map_emotion_kw:
                        self.logger.error(f"感情キーワードマップの初期化中にエラーが発生しました (処理中のキー: '{tone_key_str_from_yaml}'): {e_map_emotion_kw}", exc_info=True)
            else:
                self.logger.warning("設定に 'emotion_keywords' が存在しないか、不正な形式（辞書型期待）です。感情キーワードマップは部分的に初期化されたか、空の可能性があります。")

            # --- Load phase keywords ---
            raw_phase_keywords_from_cfg = analyzer_keywords_data_dict.get("phase_keywords")
            if isinstance(raw_phase_keywords_from_cfg, dict) and self.PsychologicalPhase_cls:
                self.logger.debug(f"位相キーワードのロード処理開始 (phase_keywords)。ロード元カテゴリ数: {len(raw_phase_keywords_from_cfg)}")
                for phase_key_str_from_yaml, keywords_list_from_yaml_ph in raw_phase_keywords_from_cfg.items():
                    cleaned_phase_key_str = str(phase_key_str_from_yaml).strip()
                    try:
                        phase_enum_member = self.PsychologicalPhase_cls(cleaned_phase_key_str)
                        if isinstance(keywords_list_from_yaml_ph, list) and all(isinstance(kw_item_ph, str) for kw_item_ph in keywords_list_from_yaml_ph):
                            valid_keywords_for_phase = [kw_ph.strip().lower() for kw_ph in keywords_list_from_yaml_ph if kw_ph.strip()]
                            if valid_keywords_for_phase:
                                self.phase_keywords_map[phase_enum_member].extend(valid_keywords_for_phase)
                                self.logger.debug(f"  位相カテゴリ '{phase_enum_member.value}': {len(valid_keywords_for_phase)}個のキーワードをロード ({', '.join(valid_keywords_for_phase[:3])}...)")
                        else:
                            self.logger.warning(f"位相キーワードデータで、キー '{cleaned_phase_key_str}' の値が不正な形式です (文字列のリストを期待しましたが、型は {type(keywords_list_from_yaml_ph)})。このキーはスキップされます。")
                    except ValueError:
                        self.logger.warning(f"位相キーワードデータのキー '{cleaned_phase_key_str}' は PsychologicalPhaseV49 Enum の有効な値ではありません。このキーはスキップされます。YAML設定を確認してください（Enumの.valueと一致している必要があります）。")
                    except Exception as e_map_phase_kw:
                        self.logger.error(f"位相キーワードマップの初期化中にエラーが発生しました (処理中のキー: '{phase_key_str_from_yaml}'): {e_map_phase_kw}", exc_info=True)
            else:
                self.logger.warning("設定に 'phase_keywords' が存在しないか、不正な形式（辞書型期待）です。位相キーワードマップは部分的に初期化されたか、空の可能性があります。")
        else:
            self.logger.error("設定に 'analyzer_keywords_data' が存在しないか、不正な形式（辞書型期待）です。感情および位相のキーワードマップは空のままになります。")

        if not self.emotion_keywords_map: self.logger.warning("最終的な感情キーワードマップが空です。感情推定の精度に大きな影響が出る可能性があります。設定ファイル ('analyzer_keywords_v49.yaml' の emotion_keywords) を確認してください。")
        if not self.phase_keywords_map: self.logger.warning("最終的な位相キーワードマップが空です。位相推定の精度に大きな影響が出る可能性があります。設定ファイル ('analyzer_keywords_v49.yaml' の phase_keywords) を確認してください。")

        # --- spaCy NLP Model Loading ---
        self.nlp_model: Optional[Any] = None # spaCy model instance
        feature_flags_config = getattr(self.config, 'feature_flags', None)
        use_advanced_nlp = getattr(feature_flags_config, 'advanced_nlp_enabled', False) if feature_flags_config else False
        is_spacy_lib_available = globals().get('SPACY_AVAILABLE', False)
        spacy_module_ref = globals().get('spacy')

        if is_spacy_lib_available and spacy_module_ref and use_advanced_nlp:
            loaded_external_configs_obj = getattr(self.config, 'loaded_external_configs', None)
            nlp_model_name_to_load = getattr(loaded_external_configs_obj, 'nlp_model_name', 'ja_core_news_sm') if loaded_external_configs_obj else 'ja_core_news_sm'
            self.logger.info(f"高度なNLP機能が有効です。spaCyモデル '{nlp_model_name_to_load}' のロードを試みます...")
            try:
                self.nlp_model = spacy_module_ref.load(nlp_model_name_to_load)
                self.logger.info(f"spaCy NLPモデル '{nlp_model_name_to_load}' のロードに成功しました。")
            except OSError as e_spacy_os_err:
                self.logger.error(f"spaCy NLPモデル '{nlp_model_name_to_load}' のロードに失敗しました (OSError): {e_spacy_os_err}. "
                                  "モデルがダウンロードされているか、パスが正しいか確認してください。NLP機能は無効化されます。")
                self.nlp_model = None
            except Exception as e_spacy_generic_err:
                self.logger.error(f"spaCy NLPモデル '{nlp_model_name_to_load}' のロード中に予期せぬエラーが発生しました: {e_spacy_generic_err}", exc_info=True)
                self.nlp_model = None
        elif is_spacy_lib_available and not use_advanced_nlp:
            self.logger.info("spaCyライブラリは利用可能ですが、設定 (feature_flags.advanced_nlp_enabled=False) により高度なNLPモデルはロードされませんでした。")
        elif not is_spacy_lib_available:
            self.logger.info("spaCyライブラリが見つからないため、高度なNLP機能は利用できません。基本的なテキスト処理のみ行われます。")

        self.logger.info(f"{self.__class__.__qualname__} の初期化が正常に完了しました。")

    def _extract_speaker_from_prefix(self, text_before_speech: str) -> Optional[str]:
        # Attempts to extract a speaker name from text immediately preceding a quote.
        # This logic is heuristic and may need refinement for complex cases.
        # Increased max length of speaker name to 15 from original for more flexibility.
        # Patterns try to capture "Speaker「" or "Speaker："
        speaker_patterns_to_try = [
            r'([^\s「『（）「」『』【】［］｛｝：、。]{1,20})\s*([「『（【［｛])\s*$', # Speaker「... (max 20 chars for speaker)
            r'([^\s「『（）「」『』【】［］｛｝：、。]{1,20})\s*：\s*$',          # Speaker：... (max 20 chars for speaker)
        ]
        best_speaker_candidate: Optional[str] = None
        stripped_text_prefix = text_before_speech.strip()

        for regex_pattern_str in speaker_patterns_to_try:
            try:
                match_result = re.search(regex_pattern_str, stripped_text_prefix)
                if match_result:
                    potential_speaker = match_result.group(1).strip()
                    # Basic validation for a speaker name (non-empty, not purely numeric)
                    if potential_speaker and not potential_speaker.isdigit():
                        self.logger.debug(f"  話者推定ヒント: 候補='{potential_speaker}' (抽出元プレフィックス末尾30文字: '{stripped_text_prefix[-30:]}')")
                        best_speaker_candidate = potential_speaker
                        break # Use the first successful match
            except re.error as e_regex_speaker:
                self.logger.warning(f"話者推定のための正規表現処理中にエラー (パターン: '{regex_pattern_str}', 対象テキスト抜粋: '{stripped_text_prefix[-30:]}'): {e_regex_speaker}")
        return best_speaker_candidate

    def _parse_dialogue_structure(self, text_input: str) -> List['DialogueBlockV49']: # type: ignore
        # Parses dialogue text into a list of SpeechBlockV49 and DescriptionBlockV49.
        parsed_dialogue_blocks: List['DialogueBlockV49'] = [] # type: ignore

        # Ensure Pydantic models for blocks are loaded
        if not (self.SpeechBlock_cls and self.DescriptionBlock_cls):
            self.logger.critical("_parse_dialogue_structure: SpeechBlockV49またはDescriptionBlockV49クラスがロードされていません。構造解析は不可能です。空のリストを返します。")
            return []

        # Regex to find speech blocks (e.g., 「...」 or 『...』)
        # Using DOTALL to make `.` match newlines within quotes if any.
        # Non-greedy match `.*?` is crucial for finding distinct blocks.
        speech_block_finder_regex = re.compile(r'(([「『])(?:(?!「|』).)*?([」』]))', re.DOTALL)
        current_parse_position = 0
        total_text_length = len(text_input)

        try:
            for speech_match_obj in speech_block_finder_regex.finditer(text_input):
                match_start_index, match_end_index = speech_match_obj.span()

                # Part before the current speech block is a description block
                if match_start_index > current_parse_position:
                    description_text_segment = text_input[current_parse_position:match_start_index].strip()
                    if description_text_segment:
                        try:
                            parsed_dialogue_blocks.append(self.DescriptionBlock_cls.model_validate({ # type: ignore
                                "type": "description", "text": description_text_segment
                            }))
                        except (self.PydanticValidationError_cls if self.PydanticValidationError_cls else Exception) as e_validate_desc: # type: ignore
                            err_msg_desc = e_validate_desc.errors(include_url=False,include_input=False) if hasattr(e_validate_desc,'errors') else str(e_validate_desc)
                            self.logger.warning(f"描写ブロックのPydanticモデル検証エラー: {err_msg_desc}. 対象テキスト抜粋: '{description_text_segment[:70]}...'")

                # The matched speech block itself
                speech_text_segment = speech_match_obj.group(1).strip() # group(1) is the content including quotes
                if speech_text_segment:
                    # Attempt to extract speaker from text immediately preceding the speech block
                    # Look at up to 40 characters before the quote for speaker hints
                    prefix_text_for_speaker_extraction = text_input[max(0, match_start_index - 40):match_start_index]
                    extracted_speaker_name = self._extract_speaker_from_prefix(prefix_text_for_speaker_extraction) or "不明" # Default to "不明"
                    try:
                        parsed_dialogue_blocks.append(self.SpeechBlock_cls.model_validate({ # type: ignore
                            "type": "speech", "text": speech_text_segment, "speaker": extracted_speaker_name
                        }))
                    except (self.PydanticValidationError_cls if self.PydanticValidationError_cls else Exception) as e_validate_speech: # type: ignore
                        err_msg_speech = e_validate_speech.errors(include_url=False,include_input=False) if hasattr(e_validate_speech,'errors') else str(e_validate_speech)
                        self.logger.warning(f"セリフブロックのPydanticモデル検証エラー: {err_msg_speech}. 話者候補: '{extracted_speaker_name}', テキスト抜粋: '{speech_text_segment[:70]}...'")

                current_parse_position = match_end_index # Move parser position to end of current speech block

            # Any remaining text after the last speech block is a final description block
            if current_parse_position < total_text_length:
                final_description_text_segment = text_input[current_parse_position:].strip()
                if final_description_text_segment:
                    try:
                        parsed_dialogue_blocks.append(self.DescriptionBlock_cls.model_validate({ # type: ignore
                            "type": "description", "text": final_description_text_segment
                        }))
                    except (self.PydanticValidationError_cls if self.PydanticValidationError_cls else Exception) as e_validate_desc_final: # type: ignore
                        err_msg_desc_final = e_validate_desc_final.errors(include_url=False,include_input=False) if hasattr(e_validate_desc_final,'errors') else str(e_validate_desc_final)
                        self.logger.warning(f"末尾の描写ブロックのPydanticモデル検証エラー: {err_msg_desc_final}. テキスト抜粋: '{final_description_text_segment[:70]}...'")

            # If no blocks were parsed from a non-empty text (e.g., no speech quotes found),
            # treat the entire text as one single description block.
            if not parsed_dialogue_blocks and text_input.strip():
                self.logger.info("対話構造解析: テキスト内にセリフ形式のブロック（「」や『』）が見つかりませんでした。テキスト全体を単一の描写ブロックとして扱います。")
                try:
                    parsed_dialogue_blocks.append(self.DescriptionBlock_cls.model_validate({"type": "description", "text": text_input.strip()})) # type: ignore
                except (self.PydanticValidationError_cls if self.PydanticValidationError_cls else Exception) as e_validate_full_desc: # type: ignore
                    err_msg_full_desc = e_validate_full_desc.errors(include_url=False,include_input=False) if hasattr(e_validate_full_desc,'errors') else str(e_validate_full_desc)
                    self.logger.error(f"テキスト全体を単一描写ブロックとして処理する際のPydanticモデル検証エラー: {err_msg_full_desc}")

        except RecursionError as e_recursion_parsing: # Catch potential RecursionError with very complex/long texts
            self.logger.error(f"_parse_dialogue_structureで再帰エラーが発生しました: {e_recursion_parsing}. テキスト長: {len(text_input)}. "
                              f"テキストプレビュー: '{text_input[:100].replace(chr(10),'/')}...'", exc_info=True)
            self._clear_structure_data_from_analyzer() # Call internal cleanup if needed (currently placeholder)
            parsed_dialogue_blocks.clear() # Clear any partially parsed blocks to ensure clean state
            if text_input.strip(): # Fallback for RecursionError: treat entire text as one description block
                try:
                    self.logger.warning("再帰エラー発生後、テキスト全体を単一の描写ブロックとしてフォールバック処理を試みます。")
                    parsed_dialogue_blocks.append(self.DescriptionBlock_cls.model_validate({"type": "description", "text": text_input.strip()})) # type: ignore
                except Exception as e_fallback_recursion:
                    self.logger.error(f"再帰エラー後のフォールバック処理中にもエラーが発生しました: {e_fallback_recursion}", exc_info=True)
                    parsed_dialogue_blocks.clear() # Ensure list is empty if fallback also fails
        except Exception as e_general_parsing: # Catch any other unexpected errors during parsing
            self.logger.error(f"_parse_dialogue_structureで予期せぬエラーが発生しました: {e_general_parsing}", exc_info=True)
            self._clear_structure_data_from_analyzer() # Placeholder
            parsed_dialogue_blocks.clear()
            if text_input.strip(): # Fallback for other exceptions
                try:
                    self.logger.warning("予期せぬエラー発生後、テキスト全体を単一の描写ブロックとしてフォールバック処理を試みます。")
                    parsed_dialogue_blocks.append(self.DescriptionBlock_cls.model_validate({"type": "description", "text": text_input.strip()})) # type: ignore
                except Exception as e_fallback_general:
                    self.logger.error(f"予期せぬエラー後のフォールバック処理中にもエラーが発生しました: {e_fallback_general}", exc_info=True)
                    parsed_dialogue_blocks.clear()

        self.logger.debug(f"対話構造解析完了。抽出されたブロック総数: {len(parsed_dialogue_blocks)}")
        return parsed_dialogue_blocks

    def _clear_structure_data_from_analyzer(self) -> None:
        """このアナライザ内部で使用する構造データをクリア（将来的な拡張用）。現状はプレースホルダー。"""
        # This method is a placeholder. If AdvancedDialogueAnalyzerV49 held internal state
        # related to the parsed structure that needed clearing, it would be done here.
        # As _parse_dialogue_structure returns the list directly, this may not be strictly needed
        # unless some internal caching or state is introduced in this class.
        self.logger.debug("AdvancedDialogueAnalyzerV49 内部の構造データクリア処理が呼び出されました（現在は何もしません）。")
        pass


    def _analyze_nlp_features(self, text_content_to_analyze: str, dialogue_blocks_list_param: Optional[List['DialogueBlockV49']] = None) -> Dict[str, Any]: # type: ignore
        # Analyzes NLP features using spaCy if available and enabled, otherwise uses basic tokenization.
        # dialogue_blocks_list_param is currently unused but kept for potential future use (e.g., block-level NLP).
        nlp_analysis_output: Dict[str, Any] = {
            "tokens_count": 0,
            "unique_lemmas_count": 0, # Or unique words if spaCy not used
            "named_entities": [], # List of (text, label) tuples
            "noun_chunks_count": 0,
            "processed_with_spacy": False # Flag indicating if spaCy was used
        }
        if not text_content_to_analyze or not text_content_to_analyze.strip():
            self.logger.debug("NLP特徴抽出: 入力テキストが空または空白のみのため、処理をスキップします。")
            return nlp_analysis_output

        if self.nlp_model and globals().get('SPACY_AVAILABLE'): # Check if spaCy model is loaded and library available
            try:
                spacy_doc = self.nlp_model(text_content_to_analyze)
                # Count non-space tokens
                nlp_analysis_output["tokens_count"] = len([token for token in spacy_doc if not token.is_space])

                # Collect unique lemmas (lowercase, excluding punctuation, spaces, stop words, and empty lemmas)
                unique_lemmas_set = set()
                for token_obj in spacy_doc:
                    if not token_obj.is_punct and not token_obj.is_space and \
                       token_obj.lemma_ and token_obj.lemma_.strip() and not token_obj.is_stop:
                        unique_lemmas_set.add(token_obj.lemma_.lower())
                nlp_analysis_output["unique_lemmas_count"] = len(unique_lemmas_set)

                # Collect unique named entities (text, label tuples)
                unique_named_entities_set = set()
                for entity_obj in spacy_doc.ents:
                    if entity_obj.text and entity_obj.text.strip():
                        unique_named_entities_set.add((entity_obj.text.strip(), entity_obj.label_))
                nlp_analysis_output["named_entities"] = list(unique_named_entities_set)

                nlp_analysis_output["noun_chunks_count"] = len(list(spacy_doc.noun_chunks))
                nlp_analysis_output["processed_with_spacy"] = True
                self.logger.debug(f"spaCyを使用したNLP特徴抽出が完了しました: "
                                  f"Tokens={nlp_analysis_output['tokens_count']}, "
                                  f"UniqueLemmas={nlp_analysis_output['unique_lemmas_count']}, "
                                  f"NamedEntities={len(nlp_analysis_output['named_entities'])}, "
                                  f"NounChunks={nlp_analysis_output['noun_chunks_count']}")
            except Exception as e_spacy_analysis_error:
                self.logger.error(f"spaCy NLP処理中にエラーが発生しました: {e_spacy_analysis_error}", exc_info=True)
                # Fallback to basic tokenization if spaCy processing fails mid-way
                basic_tokens_list = [word_tok for word_tok in text_content_to_analyze.split() if word_tok.strip()]
                nlp_analysis_output["tokens_count"] = len(basic_tokens_list)
                nlp_analysis_output["unique_lemmas_count"] = len(set(word.lower() for word in basic_tokens_list)) # Uses unique words as fallback
                nlp_analysis_output["processed_with_spacy"] = False # Mark as not successfully processed by spaCy
        else: # spaCy not available or not enabled
            basic_tokens_list_fallback = [word_tok_fb for word_tok_fb in text_content_to_analyze.split() if word_tok_fb.strip()]
            nlp_analysis_output["tokens_count"] = len(basic_tokens_list_fallback)
            nlp_analysis_output["unique_lemmas_count"] = len(set(word_fb.lower() for word_fb in basic_tokens_list_fallback))
            if self.nlp_model is None and globals().get('SPACY_AVAILABLE'): # spaCy available but model not loaded (e.g. due to config or load error)
                 self.logger.debug("簡易NLP特徴抽出完了 (spaCyモデル未ロードまたは無効化のため): "
                                   f"Tokens={nlp_analysis_output['tokens_count']}, UniqueWords={nlp_analysis_output['unique_lemmas_count']}")
            else: # spaCy library itself not available
                 self.logger.debug("簡易NLP特徴抽出完了 (spaCyライブラリ非利用のため): "
                                   f"Tokens={nlp_analysis_output['tokens_count']}, UniqueWords={nlp_analysis_output['unique_lemmas_count']}")
        return nlp_analysis_output

    def _analyze_emotions_and_phases_per_block(
        self, dialogue_blocks_input_list: List['DialogueBlockV49'] # type: ignore
    ) -> Tuple[List['EmotionCurvePointV49'], List['PhaseTimelinePointV49']]: # type: ignore
        # Analyzes emotions and psychological phases for each dialogue block.
        emotion_curve_points_generated: List['EmotionCurvePointV49'] = [] # type: ignore
        phase_timeline_points_generated: List['PhaseTimelinePointV49'] = [] # type: ignore

        # Verify that necessary Pydantic model and Enum classes are loaded
        required_classes_for_analysis_map = {
            "EmotionCurvePoint_cls": self.EmotionCurvePoint_cls,
            "EmotionalTone_cls": self.EmotionalTone_cls,
            "PhaseTimelinePoint_cls": self.PhaseTimelinePoint_cls,
            "PsychologicalPhase_cls": self.PsychologicalPhase_cls,
        }
        missing_required_classes_list = [
            cls_name for cls_name, cls_ref in required_classes_for_analysis_map.items() if not cls_ref
        ]
        if missing_required_classes_list:
            self.logger.critical(f"ブロックごとの感情・位相分析に必要なクラスがロードされていません: {', '.join(missing_required_classes_list)}。この分析処理をスキップします。")
            return [], []

        for block_index, current_dialogue_block in enumerate(dialogue_blocks_input_list):
            # Ensure block has valid text content
            block_text_content_str = getattr(current_dialogue_block, 'text', None)
            if not isinstance(block_text_content_str, str) or not block_text_content_str.strip():
                self.logger.debug(f"ブロック {block_index + 1}: テキストが空か無効なため、感情・位相分析をスキップします。")
                continue

            block_text_lower_case = block_text_content_str.lower() # For case-insensitive keyword matching

            # --- Emotion Analysis for the current block ---
            detected_emotion_tone_scores: DefaultDict['EmotionalToneV49', float] = defaultdict(float) # type: ignore
            if self.emotion_keywords_map: # Check if emotion keywords are loaded
                for tone_enum_member, keywords_for_tone in self.emotion_keywords_map.items():
                    for keyword_str in keywords_for_tone:
                        if keyword_str in block_text_lower_case:
                            detected_emotion_tone_scores[tone_enum_member] += 1.0 # Simple hit count

            # Determine dominant tone and its strength
            current_block_estimated_tone: 'EmotionalToneV49' = self.EmotionalTone_cls.UNKNOWN # type: ignore
            calculated_tone_strength = 0.3 # Default strength for UNKNOWN or weakly indicated tones

            if detected_emotion_tone_scores:
                current_block_estimated_tone = max(detected_emotion_tone_scores, key=lambda k_tone: detected_emotion_tone_scores[k_tone]) # type: ignore
                calculated_tone_strength = min(1.0, max(0.1, 0.4 + detected_emotion_tone_scores[current_block_estimated_tone] * 0.1))
                self.logger.debug(f"ブロック {block_index + 1}: 感情キーワードヒット。主要トーン: '{current_block_estimated_tone.value}', "
                                  f"計算された強度: {calculated_tone_strength:.3f}, 全ヒットスコア: {dict(detected_emotion_tone_scores)}")
            else:
                self.logger.debug(f"ブロック {block_index + 1}: 感情キーワードのヒットなし。デフォルトトーンを '{current_block_estimated_tone.value}' (強度: {calculated_tone_strength:.3f}) として使用します。")

            try:
                emotion_curve_points_generated.append(self.EmotionCurvePoint_cls.model_validate({ # type: ignore
                    "block_index": block_index,
                    "tone": current_block_estimated_tone,
                    "strength": round(calculated_tone_strength, 3)
                }))
            except (self.PydanticValidationError_cls if self.PydanticValidationError_cls else Exception) as e_validate_emotion_point: # type: ignore
                err_msg_emo_pt = e_validate_emotion_point.errors(include_url=False,include_input=False) if hasattr(e_validate_emotion_point,'errors') else str(e_validate_emotion_point)
                self.logger.warning(f"EmotionCurvePointモデルの作成中にPydantic検証エラー (ブロック {block_index+1}): {err_msg_emo_pt}")

            # --- Phase Analysis for the current block ---
            detected_phase_scores: DefaultDict['PsychologicalPhaseV49', float] = defaultdict(float) # type: ignore
            if self.phase_keywords_map: # Check if phase keywords are loaded
                for phase_enum_member, keywords_for_phase in self.phase_keywords_map.items():
                    for keyword_str_ph in keywords_for_phase:
                        if keyword_str_ph in block_text_lower_case:
                            detected_phase_scores[phase_enum_member] += 1.0

            current_block_estimated_phase: 'PsychologicalPhaseV49' = self.PsychologicalPhase_cls.UNKNOWN # type: ignore
            calculated_phase_confidence = 0.3 # Default confidence for UNKNOWN or weakly indicated phases

            if detected_phase_scores:
                current_block_estimated_phase = max(detected_phase_scores, key=lambda k_phase: detected_phase_scores[k_phase]) # type: ignore
                calculated_phase_confidence = min(1.0, max(0.1, 0.5 + detected_phase_scores[current_block_estimated_phase] * 0.1))
                self.logger.debug(f"ブロック {block_index + 1}: 位相キーワードヒット。主要位相: '{current_block_estimated_phase.value}', "
                                  f"計算された確信度: {calculated_phase_confidence:.3f}, 全ヒットスコア: {dict(detected_phase_scores)}")
            elif block_index == 0 and dialogue_blocks_input_list: # First block defaults to INTRODUCTION
                current_block_estimated_phase = self.PsychologicalPhase_cls.INTRODUCTION # type: ignore
                calculated_phase_confidence = 0.7 # Higher confidence for default first phase
                self.logger.debug(f"ブロック {block_index + 1}: 位相キーワードのヒットなし。最初のブロックのため、デフォルト位相を '{current_block_estimated_phase.value}' (確信度: {calculated_phase_confidence:.3f}) として使用します。")
            else:
                 self.logger.debug(f"ブロック {block_index + 1}: 位相キーワードのヒットなし。デフォルト位相を '{current_block_estimated_phase.value}' (確信度: {calculated_phase_confidence:.3f}) として使用します。")

            try:
                phase_timeline_points_generated.append(self.PhaseTimelinePoint_cls.model_validate({ # type: ignore
                    "block_index": block_index,
                    "phase": current_block_estimated_phase,
                    "confidence": round(calculated_phase_confidence, 3)
                }))
            except (self.PydanticValidationError_cls if self.PydanticValidationError_cls else Exception) as e_validate_phase_point: # type: ignore
                err_msg_phase_pt = e_validate_phase_point.errors(include_url=False,include_input=False) if hasattr(e_validate_phase_point,'errors') else str(e_validate_phase_point)
                self.logger.warning(f"PhaseTimelinePointモデルの作成中にPydantic検証エラー (ブロック {block_index+1}): {err_msg_phase_pt}")

        self.logger.debug(f"キーワードベースの感情・位相推定が完了しました。"
                          f"生成された感情曲線ポイント数: {len(emotion_curve_points_generated)}, "
                          f"生成された位相タイムラインポイント数: {len(phase_timeline_points_generated)}")
        return emotion_curve_points_generated, phase_timeline_points_generated

    def analyze_and_get_results(self, text_to_analyze: str) -> Dict[str, Any]:
        """
        対話テキストの包括的な分析を実行し、結果を構造化された辞書として返します。
        この辞書は、呼び出し元で `analyzer_results` として利用されることを想定しています。

        返り値の辞書構造:
        {
            "input_text_length": int,
            "analysis_summary_partial": {"dominant_phase": Optional[str], "dominant_tone": Optional[str]},
            "output_dialogue_content": {"blocks": List[Dict[str, Any]]}, # DialogueBlockV49モデルの辞書表現
            "evaluation_metrics_partial": {
                "emotion_curve": List[Dict[str, Any]],  # EmotionCurvePointV49モデルの辞書表現
                "phase_timeline": List[Dict[str, Any]] # PhaseTimelinePointV49モデルの辞書表現
            },
            "nlp_analysis_summary": Dict[str, Any],
            "error": Optional[str],
            "error_traceback": Optional[str]
        }
        """
        text_preview = text_to_analyze[:60].replace(chr(10),'/') + ('...' if len(text_to_analyze) > 60 else '')
        text_hash_for_log = hashlib.sha1(text_to_analyze.encode('utf-8', errors='ignore')).hexdigest()[:8]
        self.logger.info(f"詳細テキスト分析処理を開始 (v2.2): テキスト長={len(text_to_analyze)}文字, "
                         f"Hash={text_hash_for_log}, 先頭抜粋='{text_preview}'")

        analysis_output_dict: Dict[str, Any] = {
            "input_text_length": len(text_to_analyze),
            "analysis_summary_partial": {"dominant_phase": None, "dominant_tone": None},
            "output_dialogue_content": {"blocks": []},
            "evaluation_metrics_partial": {"emotion_curve": [], "phase_timeline": []},
            "nlp_analysis_summary": {},
            "error": None,
            "error_traceback": None
        }

        if not text_to_analyze or not text_to_analyze.strip():
            self.logger.warning(f"分析対象のテキストが空または空白のみ (Hash={text_hash_for_log})。空の結果を返却します。")
            analysis_output_dict["error"] = "Input text is empty or contains only whitespace."
            return analysis_output_dict

        if not self.BlockAnalysisTags_cls:
            critical_error_message="CRITICAL ERROR: BlockAnalysisTagsV49クラスがロードされていません。ブロックごとのタグ付け処理が実行できないため、分析を中止します。"
            self.logger.critical(critical_error_message)
            analysis_output_dict["error"] = critical_error_message
            return analysis_output_dict

        try:
            dialogue_block_model_instances_list = self._parse_dialogue_structure(text_to_analyze)
            if not dialogue_block_model_instances_list and text_to_analyze.strip():
                self.logger.warning(f"対話構造解析処理で有効なブロックが抽出できませんでした (Hash={text_hash_for_log})。分析結果が不完全になる可能性があります。")

            analysis_output_dict["nlp_analysis_summary"] = self._analyze_nlp_features(text_to_analyze, dialogue_block_model_instances_list)

            emotion_curve_model_instances_list, phase_timeline_model_instances_list = \
                self._analyze_emotions_and_phases_per_block(dialogue_block_model_instances_list)

            updated_dialogue_blocks_with_tags: List['DialogueBlockV49'] = [] # type: ignore
            for block_idx, block_model_obj in enumerate(dialogue_block_model_instances_list):
                current_block_analysis_tags_obj = self.BlockAnalysisTags_cls()
                if self.scorer:
                    block_text_content_for_scorer = getattr(block_model_obj, 'text', '')
                    if block_text_content_for_scorer:
                        try:
                            _, subjectivity_category_hits = self.scorer.calculate_subjectivity_score(block_text_content_for_scorer) # type: ignore
                            if isinstance(subjectivity_category_hits, dict) and subjectivity_category_hits:
                                current_block_analysis_tags_obj.matched_subjectivity_categories = list(subjectivity_category_hits.keys())
                        except Exception as e_subj_score_call:
                            self.logger.error(f"ブロック {block_idx+1} の主観性スコア計算呼び出し中にエラー: {e_subj_score_call}", exc_info=False)
                        try:
                            _, fluctuation_category_hits = self.scorer.calculate_fluctuation_intensity(block_text_content_for_scorer) # type: ignore
                            if isinstance(fluctuation_category_hits, dict) and fluctuation_category_hits:
                                current_block_analysis_tags_obj.matched_fluctuation_categories = list(fluctuation_category_hits.keys())
                        except Exception as e_fluc_score_call:
                            self.logger.error(f"ブロック {block_idx+1} の揺らぎ強度スコア計算呼び出し中にエラー: {e_fluc_score_call}", exc_info=False)
                if block_idx < len(phase_timeline_model_instances_list) and hasattr(phase_timeline_model_instances_list[block_idx], 'phase'):
                    current_block_analysis_tags_obj.estimated_phase = phase_timeline_model_instances_list[block_idx].phase
                if block_idx < len(emotion_curve_model_instances_list) and hasattr(emotion_curve_model_instances_list[block_idx], 'tone'):
                    current_block_analysis_tags_obj.estimated_tone = emotion_curve_model_instances_list[block_idx].tone
                if hasattr(block_model_obj, 'analysis_tags'):
                    block_model_obj.analysis_tags = current_block_analysis_tags_obj # type: ignore
                updated_dialogue_blocks_with_tags.append(block_model_obj)

            analysis_output_dict["output_dialogue_content"]["blocks"] = [
                block_obj.model_dump(exclude_none=True, by_alias=True) for block_obj in updated_dialogue_blocks_with_tags
            ]
            analysis_output_dict["evaluation_metrics_partial"]["emotion_curve"] = [
                point_obj.model_dump(exclude_none=True, by_alias=True) for point_obj in emotion_curve_model_instances_list
            ]
            analysis_output_dict["evaluation_metrics_partial"]["phase_timeline"] = [
                point_obj.model_dump(exclude_none=True, by_alias=True) for point_obj in phase_timeline_model_instances_list
            ]

            unknown_phase_enum_instance = getattr(self.PsychologicalPhase_cls, 'UNKNOWN', None)
            all_phases_from_timeline = [
                phase_point.phase for phase_point in phase_timeline_model_instances_list
                if hasattr(phase_point,'phase') and phase_point.phase and phase_point.phase != unknown_phase_enum_instance
            ]
            if all_phases_from_timeline:
                dominant_phase_enum_member = TypingCounter(all_phases_from_timeline).most_common(1)[0][0]
                analysis_output_dict["analysis_summary_partial"]["dominant_phase"] = dominant_phase_enum_member.value

            unknown_tone_enum_instance = getattr(self.EmotionalTone_cls, 'UNKNOWN', None)
            neutral_tone_enum_instance = getattr(self.EmotionalTone_cls, 'NEUTRAL', None)
            all_tones_from_curve = [
                emotion_point.tone for emotion_point in emotion_curve_model_instances_list
                if hasattr(emotion_point,'tone') and emotion_point.tone and \
                   emotion_point.tone != unknown_tone_enum_instance and emotion_point.tone != neutral_tone_enum_instance
            ]
            if all_tones_from_curve:
                dominant_tone_enum_member = TypingCounter(all_tones_from_curve).most_common(1)[0][0]
                analysis_output_dict["analysis_summary_partial"]["dominant_tone"] = dominant_tone_enum_member.value

            num_blocks_extracted = len(analysis_output_dict["output_dialogue_content"]["blocks"])
            dom_phase_log = analysis_output_dict['analysis_summary_partial']['dominant_phase'] or 'N/A'
            dom_tone_log = analysis_output_dict['analysis_summary_partial']['dominant_tone'] or 'N/A'
            self.logger.info(f"テキスト分析処理が正常に完了 (v2.2)。Hash={text_hash_for_log}, "
                             f"抽出ブロック数: {num_blocks_extracted}, "
                             f"優勢位相(推定): {dom_phase_log}, 優勢トーン(推定): {dom_tone_log}")

        except RecursionError as e_recursion_main_analysis:
            self.logger.error(f"テキスト分析プロセス全体で再帰エラーが発生しました (Hash={text_hash_for_log}): {e_recursion_main_analysis}. ", exc_info=True)
            analysis_output_dict["error"] = f"RecursionError during analysis: {e_recursion_main_analysis}"
            analysis_output_dict["error_traceback"] = traceback.format_exc()
        except Exception as e_main_analysis_unexpected:
            self.logger.error(f"テキスト分析プロセス全体で予期せぬエラーが発生しました (Hash={text_hash_for_log}): {e_main_analysis_unexpected}", exc_info=True)
            analysis_output_dict["error"] = str(e_main_analysis_unexpected)
            try:
                analysis_output_dict["error_traceback"] = traceback.format_exc()
            except Exception:
                analysis_output_dict["error_traceback"] = "Traceback formatting also failed."

        return analysis_output_dict
# =============================================================================
# -- Part 7b 終了点 (AdvancedDialogueAnalyzerV49 クラス定義終了)
# =============================================================================
# =============================================================================
# -- Part 8: Adaptation Strategy (v4.9α - 最適化・Enum移行対応版)
# =============================================================================
# PsychologicalPhaseV49 Enum (11コアメンバー) への移行を考慮し、
# 特に履歴データのロード/セーブ時のEnum変換の堅牢性を向上させます。
# _missing_ メソッドの役割を前提としつつ、ログ出力を強化します。

# from typing import TYPE_CHECKING, TypeVar, Deque # Part 0/4でインポート済み想定
# import enum, logging, random, pathlib, json, time # Part 0でインポート済み想定
# from collections import deque # Part 0でインポート済み想定
# from datetime import datetime, timezone # Part 0でインポート済み想定

if TYPE_CHECKING:
    # --- 型チェック用の前方参照 ---
    from __main__ import ( # 単一ファイル実行を想定
        ConfigProtocol, ExceptionManagerV49,
        PsychologicalPhaseV49, EmotionalToneV49,
        PhaseTransitionRecordV49, GeneratorStateV49,
        AdaptationStrategyConfigV49, # Part 3aで定義
        # AppConfigV49 # _setup_history_file で config 経由で参照
    )
    # 型エイリアス
    AdaptationConfigType: TypeAlias = AdaptationStrategyConfigV49
    PsychologicalPhaseV49EnumType: TypeAlias = PsychologicalPhaseV49
    EmotionalToneV49EnumType: TypeAlias = EmotionalToneV49
    PhaseTransitionRecordV49Type: TypeAlias = PhaseTransitionRecordV49
    GeneratorStateV49Type: TypeAlias = Optional[GeneratorStateV49] # Optional許容
else:
    # 実行時は文字列リテラルやglobals().get()で対応
    ConfigProtocol = 'ConfigProtocol'
    ExceptionManagerV49 = 'ExceptionManagerV49'
    AdaptationConfigType = 'AdaptationStrategyConfigV49'
    PsychologicalPhaseV49EnumType = globals().get('PsychologicalPhaseV49', enum.Enum)
    EmotionalToneV49EnumType = globals().get('EmotionalToneV49', enum.Enum)
    PhaseTransitionRecordV49Type = globals().get('PhaseTransitionRecordV49', type(BaseModel)) if PYDANTIC_AVAILABLE else dict # type: ignore
    GeneratorStateV49Type = Optional[globals().get('GeneratorStateV49', type(BaseModel)) if PYDANTIC_AVAILABLE else dict] # type: ignore

class PhaseToneAdaptationStrategyV49: # Implicitly implements AdaptationStrategyProtocol
    if TYPE_CHECKING:
        ConfigProtoType = ConfigProtocol
        ExceptionManagerProtoType = ExceptionManagerV49
    else:
        ConfigProtoType = 'ConfigProtocol'
        ExceptionManagerProtoType = 'ExceptionManagerV49'

    def __init__(self, config: ConfigProtoType, exception_manager: ExceptionManagerProtoType): # type: ignore
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        self.exception_manager = exception_manager

        # --- Enumクラスのロード ---
        self.PsychologicalPhase_cls = PsychologicalPhaseV49 # 直接参照
        self.EmotionalTone_cls = EmotionalToneV49 # 直接参照
        self.PhaseTransitionRecord_cls = PhaseTransitionRecordV49 # 直接参照

        # 戦略パラメータのデフォルト値 (フォールバック用)
        default_exploration_rate_init = 0.15
        default_exploration_decay = 0.995
        default_min_exploration_rate = 0.1
        default_alignment_threshold = 0.6
        default_max_history_size = 1000
        default_history_save_interval = 10

        if not (self.PsychologicalPhase_cls and self.EmotionalTone_cls and self.PhaseTransitionRecord_cls):
            self.logger.critical("CRITICAL: AdaptationStrategyに必要なEnumまたはPydanticモデルクラスがロードできませんでした。(直接参照失敗の可能性)")
            self.enabled = False
            self.log_transitions_enabled = False
            # 戦略が無効でも、属性が存在しないことによる AttributeError を避けるためフォールバック値を設定
            self.exploration_rate_init = default_exploration_rate_init
            self.exploration_decay = default_exploration_decay
            self.min_exploration_rate = default_min_exploration_rate
            self.alignment_threshold = default_alignment_threshold
            self.max_history_size = default_max_history_size
            self.history_save_interval = default_history_save_interval
            self.strategy_type = "disabled" # 無効であることを示す
        else:
            adapt_cfg: Optional[AdaptationConfigType] = self.config.adaptation_config # type: ignore
            
            if adapt_cfg is None:
                self.logger.warning("AdaptationConfigがAppConfigから取得できませんでした。デフォルト値で戦略を初期化します。")
                self.enabled = True # デフォルトでは有効と仮定
                self.strategy_type = "probabilistic_history"
                self.log_transitions_enabled = True
                self.alignment_threshold = default_alignment_threshold
                ph_params = None # 下の分岐でデフォルト値が使われるように
            else:
                self.enabled = getattr(adapt_cfg, 'enabled', True)
                self.strategy_type = getattr(adapt_cfg, 'strategy_type', "probabilistic_history")
                self.log_transitions_enabled = getattr(adapt_cfg, 'log_transitions', True)
                self.alignment_threshold = getattr(adapt_cfg, 'alignment_threshold', default_alignment_threshold)
                ph_params = getattr(adapt_cfg, 'probabilistic_history_params', None)
            
            if ph_params:
                # Pydanticモデルの populate_by_name=True を考慮し、aliasも参照できるgetattrを使用
                self.exploration_rate_init = getattr(ph_params, 'exploration_rate_init', default_exploration_rate_init)
                self.exploration_decay = getattr(ph_params, 'exploration_decay', default_exploration_decay)
                self.min_exploration_rate = getattr(ph_params, 'min_exploration_rate', default_min_exploration_rate)
                self.max_history_size = getattr(ph_params, 'max_history_size', default_max_history_size)
                self.history_save_interval = getattr(ph_params, 'history_save_interval', default_history_save_interval)
                # alignment_threshold は adapt_cfg 直下から取得済みなので、ここでは上書きしない
            else:
                self.logger.warning("probabilistic_history_params が AdaptationConfig に見つかりません。デフォルト値を使用します。")
                self.exploration_rate_init = default_exploration_rate_init
                self.exploration_decay = default_exploration_decay
                self.min_exploration_rate = default_min_exploration_rate
                self.max_history_size = default_max_history_size
                self.history_save_interval = default_history_save_interval
        
        # current_exploration_rate の初期化を、関連属性が設定された後に移動
        self.current_exploration_rate: float = self.exploration_rate_init
        
        self.history_file_path: Optional[pathlib.Path] = self._setup_history_file()
        self.transition_history: Dict[str, Dict[str, Dict[str, Union[int, float]]]] = {}
        self.recent_transitions: deque[PhaseTransitionRecordV49Type] = deque(maxlen=self.max_history_size) # type: ignore

        if self.enabled and self.log_transitions_enabled:
            self._load_history()
        
        self.logger.info(f"PhaseToneAdaptationStrategyV49 (Type:'{self.strategy_type}', Enabled:{self.enabled}) 初期化完了。")
        if self.enabled:
            self.logger.info(f"  戦略パラメータ: ε_init={self.exploration_rate_init:.3f}, ε_decay={self.exploration_decay:.3f}, ε_min={self.min_exploration_rate:.3f}, align_thresh={self.alignment_threshold:.2f}")
            if self.history_file_path: self.logger.info(f"  履歴ファイル: {self.history_file_path}")

    def _setup_history_file(self) -> Optional[pathlib.Path]:
        """履歴ファイルのパスを準備し、必要ならディレクトリを作成します。"""
        if not self.log_transitions_enabled:
            self.logger.debug("遷移ログが無効のため、履歴ファイルパスの設定をスキップします。")
            return None
        try:
            # AppConfigV49からキャッシュディレクトリとRLモデルディレクトリ名を取得
            # self.config は ConfigProtocol を実装した AppConfigV49 インスタンスを期待
            cache_dir_base: pathlib.Path = self.config.cache_dir # AppConfigV49.cache_dir (Pathオブジェクト)
            
            rl_model_dir_name_str: str = "rl_models_history" # デフォルト
            if self.config.loaded_external_configs and self.config.loaded_external_configs.file_settings: # type: ignore
                rl_model_dir_name_str = self.config.loaded_external_configs.file_settings.rl_model_dir # type: ignore
            
            history_dir = cache_dir_base / rl_model_dir_name_str
            history_dir.mkdir(parents=True, exist_ok=True) # ディレクトリ作成
            
            history_filename = "phase_tone_adaptation_history_v49.json"
            full_history_path = history_dir / history_filename
            self.logger.info(f"適応戦略の履歴ファイルパスが '{full_history_path}' に設定されました。")
            return full_history_path
        except AttributeError as e_attr:
            self.logger.error(f"履歴ファイルパス設定エラー: configオブジェクトに必要な属性がありません ({e_attr})。", exc_info=True)
        except Exception as e_path_setup:
            self.logger.error(f"履歴ファイルパス設定中に予期せぬエラー: {e_path_setup}", exc_info=True)
        return None

    def _get_state_key(self, phase: Optional[PsychologicalPhaseV49EnumType], tone: Optional[EmotionalToneV49EnumType]) -> str: # type: ignore
        """与えられた位相とトーンのEnumメンバーから状態キー文字列を生成します。"""
        # PsychologicalPhaseV49Enum.value と EmotionalToneV49Enum.value を使用
        phase_value_str = phase.value if phase and hasattr(phase, 'value') else "NONE_PHASE" # Noneや不正な型の場合の代替文字列
        tone_value_str = tone.value if tone and hasattr(tone, 'value') else "NONE_TONE"
        return f"{phase_value_str}:{tone_value_str}"

    def _parse_state_key(self, key_str: str) -> Tuple[Optional[PsychologicalPhaseV49EnumType], Optional[EmotionalToneV49EnumType]]: # type: ignore
        """
        状態キー文字列（例: "introduction:happy"）を解析し、
        対応するPsychologicalPhaseV49とEmotionalToneV49のEnumメンバーのタプルを返します。
        変換できない場合は (None, None) または (UNKNOWN, UNKNOWN) を返します。
        Enumの _missing_ メソッドの動作に依存します。
        """
        if not (self.PsychologicalPhase_cls and self.EmotionalTone_cls):
            self.logger.error("_parse_state_key: PsychologicalPhaseV49 または EmotionalToneV49 Enumクラスが未ロードです。")
            return None, None
            
        try:
            phase_str_part, tone_str_part = key_str.split(":", 1)
            
            parsed_phase: Optional[PsychologicalPhaseV49EnumType] = None # type: ignore
            if phase_str_part != "NONE_PHASE":
                try:
                    parsed_phase = self.PsychologicalPhase_cls(phase_str_part) # type: ignore # _missing_呼び出し
                    if parsed_phase == self.PsychologicalPhase_cls.UNKNOWN and phase_str_part.lower() != self.PsychologicalPhase_cls.UNKNOWN.value:
                         self.logger.info(f"状態キー解析: 位相部分 '{phase_str_part}' は UNKNOWN にマップされました。")
                except ValueError: # _missing_がNoneを返すか、変換失敗
                    self.logger.warning(f"状態キー '{key_str}' の位相部分 '{phase_str_part}' をPsychologicalPhaseV49に変換できませんでした。UNKNOWN扱い。")
                    parsed_phase = self.PsychologicalPhase_cls.UNKNOWN # type: ignore
            
            parsed_tone: Optional[EmotionalToneV49EnumType] = None # type: ignore
            if tone_str_part != "NONE_TONE":
                try:
                    parsed_tone = self.EmotionalTone_cls(tone_str_part) # type: ignore # _missing_呼び出し
                    if parsed_tone == self.EmotionalTone_cls.UNKNOWN and tone_str_part.lower() != self.EmotionalTone_cls.UNKNOWN.value:
                         self.logger.info(f"状態キー解析: トーン部分 '{tone_str_part}' は UNKNOWN にマップされました。")
                except ValueError:
                    self.logger.warning(f"状態キー '{key_str}' のトーン部分 '{tone_str_part}' をEmotionalToneV49に変換できませんでした。UNKNOWN扱い。")
                    parsed_tone = self.EmotionalTone_cls.UNKNOWN # type: ignore
            
            return parsed_phase, parsed_tone
        except ValueError: # キー文字列が ":" を含まないなど、形式不正
            self.logger.error(f"状態キー '{key_str}' の形式が不正です ('phase_value:tone_value' 形式を期待)。")
            return None, None # または (UNKNOWN, UNKNOWN)
        except Exception as e_parse:
            self.logger.error(f"状態キー '{key_str}' の解析中に予期せぬエラー: {e_parse}", exc_info=True)
            return None, None

    def _load_history(self) -> None:
        """遷移履歴をJSONファイルからロードします。堅牢性とログ出力を強化。"""
        if not self.history_file_path:
            self.logger.info("遷移履歴ファイルパスが未設定のため、ロードをスキップします。")
            return
        if not self.history_file_path.is_file():
            self.logger.info(f"遷移履歴ファイル '{self.history_file_path}' が見つからないため、ロードをスキップします。")
            return

        self.logger.info(f"遷移履歴ファイル '{self.history_file_path}' のロードを開始します...")

        # ExceptionManagerV49.safe_file_operation を使用してファイルロード
        # ★★★ 修正箇所: キーワード引数をメソッド定義に合わせて 'name' と 'func' に変更 ★★★
        success_load, loaded_data, error_details_obj = self.exception_manager.safe_file_operation( # type: ignore[attr-defined]
            name="適応戦略履歴ロード",        # 第1引数 (name)
            func=load_json,               # 第2引数 (func)
            args=(self.history_file_path,)
        )
        # ★★★ 修正完了 ★★★

        if not success_load:
            error_msg = f"履歴ファイル '{self.history_file_path}' のロードに失敗しました。"
            if error_details_obj:
                error_msg += f" エラー: {error_details_obj}" # error_details_obj は StructuredErrorV49 インスタンスの可能性
            self.logger.error(error_msg)
            self.transition_history = {} # 失敗時は空にする
            self.recent_transitions.clear()
            return

        if not isinstance(loaded_data, dict):
            self.logger.error(
                f"履歴ファイル '{self.history_file_path}' の内容が期待される辞書形式ではありません (型: {type(loaded_data)})。履歴を初期化します。"
            )
            self.transition_history = {}
            self.recent_transitions.clear()
            return

        # --- transition_history のロードと検証 ---
        raw_transition_counts = loaded_data.get("transition_counts_rewards")
        parsed_history: Dict[str, Dict[str, Dict[str, Union[int, float]]]] = {}
        if isinstance(raw_transition_counts, dict):
            for from_key, to_map_raw in raw_transition_counts.items():
                if not isinstance(from_key, str):
                    self.logger.warning(f"履歴データ内の不正な遷移元キー型: {type(from_key)} ({from_key})。スキップします。")
                    continue
                if not isinstance(to_map_raw, dict):
                    self.logger.warning(f"履歴データ内の遷移元キー '{from_key}' に対応する遷移先データが辞書形式ではありません。スキップします。")
                    continue

                if not (":" in from_key and len(from_key.split(':')) == 2):
                    self.logger.warning(f"履歴データ内の不正な形式の遷移元キー: '{from_key}'。スキップします。")
                    continue

                valid_to_map: Dict[str, Dict[str, Union[int, float]]] = {}
                for to_key, stats_dict_raw in to_map_raw.items():
                    if not isinstance(to_key, str):
                        self.logger.warning(f"履歴データ(遷移元: {from_key})内の不正な遷移先キー型: {type(to_key)} ({to_key})。スキップします。")
                        continue
                    if not (":" in to_key and len(to_key.split(':')) == 2):
                         self.logger.warning(f"履歴データ(遷移元: {from_key})内の不正な形式の遷移先キー: '{to_key}'。スキップします。")
                         continue

                    if isinstance(stats_dict_raw, dict):
                        count_val = stats_dict_raw.get("count")
                        avg_reward_val = stats_dict_raw.get("avg_reward")
                        if isinstance(count_val, int) and isinstance(avg_reward_val, (float, int)):
                            valid_to_map[to_key] = {"count": count_val, "avg_reward": float(avg_reward_val)}
                        else:
                            self.logger.warning(f"履歴データ(遷移元: {from_key}, 遷移先: {to_key})の統計情報 (count/avg_reward) が不正です。スキップします。")
                    else:
                        self.logger.warning(f"履歴データ(遷移元: {from_key}, 遷移先: {to_key})の統計情報が辞書形式ではありません。スキップします。")

                if valid_to_map:
                    parsed_history[from_key] = valid_to_map
        else:
            self.logger.warning("'transition_counts_rewards' が履歴データ内にないか、辞書形式ではありません。")

        self.transition_history = parsed_history
        self.logger.info(f"ロードおよび検証された遷移履歴の状態キー数: {len(self.transition_history)}")

        # --- recent_transitions のロードと検証 (PhaseTransitionRecordV49モデルを使用) ---
        raw_recent_transitions = loaded_data.get("recent_transitions")
        loaded_recent_records: List[PhaseTransitionRecordV49Type] = [] # type: ignore[name-defined]

        if isinstance(raw_recent_transitions, list):
            if self.PhaseTransitionRecord_cls: # Pydanticモデルクラスがロードされているか確認
                for r_idx, record_dict_raw in enumerate(raw_recent_transitions):
                    if isinstance(record_dict_raw, dict):
                        try:
                            validated_record = self.PhaseTransitionRecord_cls.model_validate(record_dict_raw)
                            loaded_recent_records.append(validated_record)
                        except ValidationError as ve_recent: # Pydanticの検証エラー
                            self.logger.warning(
                                f"最近の遷移記録(エントリIDX: {r_idx})のPydantic検証エラー。エラー: {ve_recent.errors(include_input=False, include_url=False)}. "
                                f"データプレビュー: {str(record_dict_raw)[:200]}"
                            )
                        except Exception as e_val_recent_other:
                            self.logger.error(
                                f"最近の遷移記録(エントリIDX: {r_idx})のPydanticモデル変換中に予期せぬエラー: {e_val_recent_other}. "
                                f"データプレビュー: {str(record_dict_raw)[:200]}",
                                exc_info=True
                            )
                    else:
                        self.logger.warning(f"最近の遷移記録(エントリIDX: {r_idx})が辞書形式ではありません (型: {type(record_dict_raw)})。スキップします。")
            else:
                self.logger.error("PhaseTransitionRecordV49 モデルクラスが未ロードのため、最近の遷移記録を検証できません。")
        else:
             self.logger.warning("'recent_transitions' が履歴データ内にないか、リスト形式ではありません。")

        self.recent_transitions.clear()
        start_index = max(0, len(loaded_recent_records) - self.max_history_size)
        self.recent_transitions.extend(loaded_recent_records[start_index:])

        self.logger.info(
            f"遷移履歴ロード完了: transition_historyキー数={len(self.transition_history)}, "
            f"recent_transitions件数={len(self.recent_transitions)} (最大保持数: {self.max_history_size})"
        )


    def save_history(self) -> None:
        """現在の遷移履歴をJSONファイルに保存します。"""
        if not self.history_file_path or not self.log_transitions_enabled:
            self.logger.debug("遷移ログが無効か履歴ファイルパス未設定のため、保存をスキップします。")
            return
        
        # self.transition_history のキーは既に _get_state_key で生成された
        # 最新のEnum .value 文字列に基づいているはず。
        data_to_save = {
            "transition_counts_rewards": self.transition_history,
            # self.recent_transitions は PhaseTransitionRecordV49 インスタンスのリスト
            # Pydanticモデルの model_dump でシリアライズ可能な辞書に変換
            "recent_transitions": [record.model_dump(mode='json', by_alias=True) for record in self.recent_transitions],
            "last_updated_utc": datetime.now(timezone.utc).isoformat() + 'Z'
        }
        
        # create_backup, save_json はPart 2で定義されたユーティリティ関数
        ok_backup, backup_path, err_backup = self.exception_manager.safe_file_operation( # type: ignore
            "適応戦略履歴バックアップ作成", create_backup, args=(self.history_file_path,)
        )
        if not ok_backup:
            self.logger.warning(f"履歴ファイルのバックアップ作成に失敗: {err_backup}")

        ok_save, _, err_save_main = self.exception_manager.safe_file_operation( # type: ignore
            "適応戦略履歴保存", save_json, args=(data_to_save, self.history_file_path)
        )
        if ok_save:
            self.logger.info(f"遷移履歴 ({len(self.recent_transitions)}件の最近の記録) を '{self.history_file_path}' に保存しました。")
        else:
            self.logger.error(f"遷移履歴の保存に失敗しました: {err_save_main}")


    def record_transition(self, record: PhaseTransitionRecordV49Type) -> None: # type: ignore
        """遷移情報を記録し、履歴を更新します。"""
        if not self.log_transitions_enabled: return
        if not self.PhaseTransitionRecord_cls or not isinstance(record, self.PhaseTransitionRecord_cls): # type: ignore
            self.logger.warning(f"記録試行: 不正な遷移記録データ型 ({type(record)})。無視します。")
            return
        
        # record.from_phase, record.to_phase は既に PsychologicalPhaseV49 Enumメンバーであることを期待。
        # record.from_tone, record.to_tone も同様。
        if not (record.from_phase and record.from_tone and record.to_phase and record.to_tone):
            self.logger.debug("記録試行: 遷移記録に必要な位相/トーン情報が不足。詳細履歴更新をスキップ。")
            # この場合でも recent_transitions には追加するかもしれない
            self.recent_transitions.append(record)
            return

        record.reward = round(max(0.0, min(1.0, record.reward)), 4) # 報酬を正規化
        self.recent_transitions.append(record) # deque に追加 (maxlen で自動的に古いものが削除)

        from_state_key_str = self._get_state_key(record.from_phase, record.from_tone)
        to_state_key_str = self._get_state_key(record.to_phase, record.to_tone)
        
        if from_state_key_str not in self.transition_history:
            self.transition_history[from_state_key_str] = {}
        if to_state_key_str not in self.transition_history[from_state_key_str]:
            self.transition_history[from_state_key_str][to_state_key_str] = {"count": 0, "avg_reward": 0.0}
        
        current_entry = self.transition_history[from_state_key_str][to_state_key_str]
        old_total_reward = current_entry["avg_reward"] * current_entry["count"]
        new_count = current_entry["count"] + 1
        new_avg_reward = (old_total_reward + record.reward) / new_count
        
        current_entry["count"] = new_count
        current_entry["avg_reward"] = round(new_avg_reward, 4)
        
        self.logger.debug(f"遷移履歴更新: {from_state_key_str} -> {to_state_key_str}, Count={new_count}, AvgReward={new_avg_reward:.3f}")

        # 一定間隔で履歴をファイルに保存
        loop_num_for_save = getattr(record, 'loop_number', 0) # PhaseTransitionRecordV49にloop_numberがある想定
        if loop_num_for_save > 0 and self.history_save_interval > 0 and \
           loop_num_for_save % self.history_save_interval == 0:
            self.save_history()

    def suggest_next_state(
        self,
        current_intended_phase: Optional[PsychologicalPhaseV49EnumType], # type: ignore
        current_intended_tone: Optional[EmotionalToneV49EnumType], # type: ignore
        last_inferred_phase: Optional[PsychologicalPhaseV49EnumType], # type: ignore
        last_inferred_tone: Optional[EmotionalToneV49EnumType], # type: ignore
        last_alignment_scores: Tuple[Optional[float], Optional[float]],
        current_generator_state: GeneratorStateV49Type = None # type: ignore
    ) -> Tuple[Optional[PsychologicalPhaseV49EnumType], Optional[EmotionalToneV49EnumType]]: # type: ignore
        """次の意図すべき位相とトーンを提案します。"""
        if not self.enabled:
            self.logger.debug("適応戦略が無効のため、現在の意図状態をそのまま返します。")
            return current_intended_phase, current_intended_tone
        
        # PsychologicalPhaseV49 Enumの変更に伴い、_suggest_by_threshold や
        # _suggest_by_probabilistic_history 内での位相比較やランダム選択が、
        # 新しい11コアメンバーを正しく扱えるか確認済み（既存のロジックで対応可能）。
        if self.strategy_type == "probabilistic_history":
            return self._suggest_by_probabilistic_history(current_intended_phase, current_intended_tone)
        elif self.strategy_type == "simple_threshold":
            return self._suggest_by_threshold(
                current_intended_phase, current_intended_tone,
                last_inferred_phase, last_inferred_tone,
                last_alignment_scores
            )
        else: # 未知の戦略タイプの場合
            self.logger.warning(f"未知の適応戦略タイプ '{self.strategy_type}' が指定されました。simple_threshold戦略をフォールバックとして使用します。")
            return self._suggest_by_threshold(
                current_intended_phase, current_intended_tone,
                last_inferred_phase, last_inferred_tone,
                last_alignment_scores
            )

    def _suggest_by_threshold(
        self,
        current_phase: Optional[PsychologicalPhaseV49EnumType], # type: ignore
        current_tone: Optional[EmotionalToneV49EnumType], # type: ignore
        last_inferred_phase: Optional[PsychologicalPhaseV49EnumType], # type: ignore
        last_inferred_tone: Optional[EmotionalToneV49EnumType], # type: ignore
        last_alignment_scores: Tuple[Optional[float], Optional[float]]
    ) -> Tuple[Optional[PsychologicalPhaseV49EnumType], Optional[EmotionalToneV49EnumType]]: # type: ignore
        # このメソッド内のロジックは、Enumメンバー間の直接比較（==）に依存しており、
        # Enumの具体的な値や順序には依存しないため、PsychologicalPhaseV49の変更による影響は少ない。
        # last_inferred_phase が UNKNOWN でないことを確認するロジックは有効。
        next_suggested_phase = current_phase
        next_suggested_tone = current_tone
        phase_align_score, tone_align_score = last_alignment_scores

        if self.PsychologicalPhase_cls and phase_align_score is not None and phase_align_score < self.alignment_threshold:
            if last_inferred_phase and last_inferred_phase != self.PsychologicalPhase_cls.UNKNOWN:
                next_suggested_phase = last_inferred_phase
                self.logger.info(
                    f"ThresholdStrategy: 位相整合性低 ({phase_align_score:.2f} < {self.alignment_threshold:.2f})。"
                    f" 次の意図位相を推定された '{last_inferred_phase.value}' に変更提案。"
                )
            else:
                self.logger.info(
                    f"ThresholdStrategy: 位相整合性低 ({phase_align_score:.2f} < {self.alignment_threshold:.2f}) だが、有効な推定位相なし。"
                    f" 意図位相 '{getattr(current_phase, 'value', 'N/A')}' を維持。"
                )
        
        if self.EmotionalTone_cls and tone_align_score is not None and tone_align_score < self.alignment_threshold:
            if last_inferred_tone and last_inferred_tone != self.EmotionalTone_cls.UNKNOWN:
                next_suggested_tone = last_inferred_tone
                self.logger.info(
                    f"ThresholdStrategy: トーン整合性低 ({tone_align_score:.2f} < {self.alignment_threshold:.2f})。"
                    f" 次の意図トーンを推定された '{last_inferred_tone.value}' に変更提案。"
                )
            else:
                self.logger.info(
                    f"ThresholdStrategy: トーン整合性低 ({tone_align_score:.2f} < {self.alignment_threshold:.2f}) だが、有効な推定トーンなし。"
                    f" 意図トーン '{getattr(current_tone, 'value', 'N/A')}' を維持。"
                )
                
        return next_suggested_phase, next_suggested_tone

    def _suggest_by_probabilistic_history(
        self,
        current_phase: Optional[PsychologicalPhaseV49EnumType], # type: ignore
        current_tone: Optional[EmotionalToneV49EnumType] # type: ignore
    ) -> Tuple[Optional[PsychologicalPhaseV49EnumType], Optional[EmotionalToneV49EnumType]]: # type: ignore
        """ε-greedy戦略に基づいて次の状態を提案します。"""
        # 探索率の減衰
        self.current_exploration_rate = max(self.min_exploration_rate, self.current_exploration_rate * self.exploration_decay)
        self.logger.debug(f"適応戦略(ProbabilisticHistory): 現在の探索率 ε = {self.current_exploration_rate:.4f}")

        if random.random() < self.current_exploration_rate:
            self.logger.info("ProbabilisticHistory (ε-greedy): [探索] ランダムな次状態を選択します。")
            return self._get_random_phase_tone() # ランダム選択
        else:
            self.logger.info("ProbabilisticHistory (ε-greedy): [活用] 履歴から最良の次状態を選択します。")
            best_next_phase, best_next_tone = self._get_best_action_from_history(current_phase, current_tone)
            
            if best_next_phase is not None or best_next_tone is not None: # 有効な提案があった場合
                return best_next_phase, best_next_tone
            else: # 履歴に情報がない場合もランダム選択にフォールバック
                self.logger.info("ProbabilisticHistory: 履歴に有効な情報なし。ランダムな次状態を選択します（活用のフォールバック）。")
                return self._get_random_phase_tone()

    def _get_random_phase_tone(self) -> Tuple[Optional[PsychologicalPhaseV49EnumType], Optional[EmotionalToneV49EnumType]]: # type: ignore
        """ランダムに次の位相とトーンを選択します。UNKNOWNは除外します。"""
        if not (self.PsychologicalPhase_cls and self.EmotionalTone_cls):
            self.logger.error("_get_random_phase_tone: PsychologicalPhaseV49 または EmotionalToneV49 Enumクラスが未ロードです。")
            return None, None

        # UNKNOWNメンバーを除外してランダム選択
        unknown_phase_member = getattr(self.PsychologicalPhase_cls, 'UNKNOWN', None)
        available_phases = [p for p in self.PsychologicalPhase_cls if p != unknown_phase_member] # type: ignore
        
        unknown_tone_member = getattr(self.EmotionalTone_cls, 'UNKNOWN', None)
        available_tones = [t for t in self.EmotionalTone_cls if t != unknown_tone_member] # type: ignore
        
        next_random_phase: Optional[PsychologicalPhaseV49EnumType] = random.choice(available_phases) if available_phases else None # type: ignore
        next_random_tone: Optional[EmotionalToneV49EnumType] = random.choice(available_tones) if available_tones else None # type: ignore
        
        self.logger.debug(f"ランダム選択結果: Phase={getattr(next_random_phase, 'value', 'N/A')}, Tone={getattr(next_random_tone, 'value', 'N/A')}")
        return next_random_phase, next_random_tone

    def _get_best_action_from_history(
        self,
        current_phase: Optional[PsychologicalPhaseV49EnumType], # type: ignore
        current_tone: Optional[EmotionalToneV49EnumType] # type: ignore
    ) -> Tuple[Optional[PsychologicalPhaseV49EnumType], Optional[EmotionalToneV49EnumType]]: # type: ignore
        """現在の状態キーに基づいて、履歴から最も平均報酬の高い次の状態を返します。"""
        current_state_key = self._get_state_key(current_phase, current_tone)
        best_next_state_key: Optional[str] = None
        highest_average_reward: float = -1.0 # 報酬は0以上なので、-1.0で初期化

        if current_state_key in self.transition_history:
            possible_next_state_transitions = self.transition_history[current_state_key]
            if isinstance(possible_next_state_transitions, dict) and possible_next_state_transitions:
                for next_state_key_str, stats_data_dict in possible_next_state_transitions.items():
                    if isinstance(stats_data_dict, dict):
                        avg_reward_val = stats_data_dict.get("avg_reward")
                        count_val = stats_data_dict.get("count")
                        
                        if isinstance(avg_reward_val, (float, int)) and isinstance(count_val, int):
                            if avg_reward_val > highest_average_reward:
                                highest_average_reward = avg_reward_val
                                best_next_state_key = next_state_key_str
                            elif avg_reward_val == highest_average_reward:
                                # 報酬が同じ場合は、試行回数が多い方を優先（より信頼性が高いと仮定）
                                current_best_count = possible_next_state_transitions.get(str(best_next_state_key), {}).get('count', 0) if best_next_state_key else 0
                                if count_val > current_best_count:
                                    best_next_state_key = next_state_key_str
                                    self.logger.debug(f"  履歴最良選択: 同報酬 ({avg_reward_val:.3f}) だが、'{next_state_key_str}' の試行回数 ({count_val}) が現在の最良候補の試行回数 ({current_best_count}) より多いため更新。")
                
                if best_next_state_key:
                    self.logger.info(f"履歴からの最良遷移先: '{current_state_key}' -> '{best_next_state_key}' (平均報酬: {highest_average_reward:.3f})")
                else:
                    self.logger.info(f"状態 '{current_state_key}' からの有効な遷移履歴（報酬付き）が見つかりませんでした。")
            else:
                self.logger.info(f"状態 '{current_state_key}' からの遷移履歴が空または不正な形式です。")
        else:
            self.logger.info(f"状態 '{current_state_key}' の遷移履歴がありません。")

        if best_next_state_key:
            return self._parse_state_key(best_next_state_key)
        
        # 有効な履歴がない場合はNoneを返す（呼び出し元でランダム選択などにフォールバック）
        return None, None

# =============================================================================
# -- Part 8 終了点
# =============================================================================
# =============================================================================
# -- Part 9: Feedback Strategy (v4.9α - 最適化・Enum移行対応版)
# =============================================================================
# フィードバック生成戦略パターン。
# PsychologicalPhaseV49 Enum (11コアメンバー) への対応を確実にする。
# 設定参照の堅牢性、型ヒントの正確性、エラーハンドリング、ログ出力を改善。

from typing import TYPE_CHECKING, TypeVar, List, Dict, Optional, Tuple, Union, Any, Type # Part 0でインポート想定
import enum # Part 0でインポート想定
import logging # Part 0でインポート想定
import re # Part 0でインポート想定
# from collections import defaultdict # 必要に応じて (SbjFbStrategyなどで使用の可能性)

# fmtユーティリティ関数 (Part 2で定義済みと仮定)
if TYPE_CHECKING:
    from __main__ import fmt # type: ignore
else:
    fmt = globals().get('fmt', lambda val, prec=2, na="N/A": str(val))


# --- 型チェック用の前方参照 ---
if TYPE_CHECKING:
    from __main__ import ( # 単一ファイル実行を想定
        ConfigProtocol, FeedbackContextV49, FeedbackStrategyProtocol,
        PsychologicalPhaseV49, EmotionalToneV49, DFRSMetricsV49, ScoreKeys,
        AppConfigV49, # 設定取得のため
        TwoDimensionalTemperatureParams, # SubjectivityFeedbackStrategyV49で使用
        AdaptationStrategyConfigV49, FeedbackStrategyConfigV49, PhaseToneFeedbackParams # 設定モデル
    )
    # 型エイリアス
    FeedbackContextV49Type: TypeAlias = FeedbackContextV49
    ConfigProtoType: TypeAlias = ConfigProtocol
    PsychologicalPhaseV49EnumType: TypeAlias = PsychologicalPhaseV49
    EmotionalToneV49EnumType: TypeAlias = EmotionalToneV49
    DFRSMetricsV49EnumType: TypeAlias = DFRSMetricsV49
    ScoreKeysLLMEnumType: TypeAlias = ScoreKeys.LLM
    # 設定モデルの型エイリアス
    AdaptationConfigType: TypeAlias = AdaptationStrategyConfigV49
    FeedbackConfigType: TypeAlias = FeedbackStrategyConfigV49
    PhaseToneParamsType: TypeAlias = PhaseToneFeedbackParams
    TwoDimTempParamsType: TypeAlias = TwoDimensionalTemperatureParams

else:
    # 実行時は文字列リテラルやglobals().get()で対応
    FeedbackContextV49Type = 'FeedbackContextV49'
    ConfigProtoType = 'ConfigProtocol'
    FeedbackStrategyProtocol = 'FeedbackStrategyProtocol'
    PsychologicalPhaseV49EnumType = globals().get('PsychologicalPhaseV49', enum.Enum)
    EmotionalToneV49EnumType = globals().get('EmotionalToneV49', enum.Enum)
    DFRSMetricsV49EnumType = globals().get('DFRSMetricsV49', enum.Enum)
    _ScoreKeys_cls = globals().get('ScoreKeys')
    ScoreKeysLLMEnumType = getattr(_ScoreKeys_cls, 'LLM', enum.Enum) if _ScoreKeys_cls else enum.Enum
    AdaptationConfigType = globals().get('AdaptationStrategyConfigV49', dict)
    FeedbackConfigType = globals().get('FeedbackStrategyConfigV49', dict)
    PhaseToneParamsType = globals().get('PhaseToneFeedbackParams', dict)
    TwoDimTempParamsType = globals().get('TwoDimensionalTemperatureParams', dict)


# --- Concrete Feedback Strategies ---

class PhaseToneFeedbackStrategyV49: # Implicitly implements FeedbackStrategyProtocol
    """位相・感情の整合性に関するフィードバック生成戦略 (v4.9α 最適化版)"""
    if TYPE_CHECKING:
        ConfigRefType = ConfigProtoType
        FeedbackContextRefType = FeedbackContextV49Type
    else:
        ConfigRefType = 'ConfigProtocol'
        FeedbackContextRefType = 'FeedbackContextV49'

    def __init__(self, config: ConfigRefType): # type: ignore
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        
        # 設定値のロード (AppConfigV49のプロパティ経由を想定)
        adapt_config_model: Optional[AdaptationConfigType] = self.config.adaptation_config # type: ignore
        self.alignment_threshold_01: float = getattr(adapt_config_model, 'alignment_threshold', 0.6) \
            if adapt_config_model else 0.6

        feedback_config_model: Optional[FeedbackConfigType] = self.config.feedback_config # type: ignore
        phase_tone_params_model: Optional[PhaseToneParamsType] = None # type: ignore
        if feedback_config_model:
            phase_tone_params_model = getattr(feedback_config_model, 'phase_tone_params', None)
        
        self.ecs_threshold_low: float = getattr(phase_tone_params_model, 'ecs_low_threshold', 2.5) \
            if phase_tone_params_model else 2.5
        self.ecs_threshold_high: float = getattr(phase_tone_params_model, 'ecs_high_threshold', 4.5) \
            if phase_tone_params_model else 4.5
        self.ptn_threshold_low: float = getattr(phase_tone_params_model, 'ptn_low_threshold', 3.0) \
            if phase_tone_params_model else 3.0
        
        self.DFRSMetrics_cls_ref: Optional[Type[DFRSMetricsV49EnumType]] = _get_global_type('DFRSMetricsV49', enum.EnumMeta) # type: ignore
        if not self.DFRSMetrics_cls_ref:
            self.logger.error("DFRSMetricsV49 Enumクラスがロードできませんでした。PhaseToneFeedbackStrategyが正しく機能しない可能性があります。")

        self.logger.debug(
            f"PhaseToneFeedbackStrategy初期化: align_thresh={self.alignment_threshold_01:.2f}, "
            f"ecs_low={self.ecs_threshold_low:.2f}, ecs_high={self.ecs_threshold_high:.2f}, ptn_low={self.ptn_threshold_low:.2f}"
        )

    def generate(self, context: FeedbackContextRefType) -> str: # type: ignore
        """
        文脈に基づき、位相/感情整合性に関する自然なフィードバックを生成します。
        context.intended_phase と context.inferred_phase は PsychologicalPhaseV49 Enumメンバーを期待。
        """
        feedback_parts: List[str] = []
        if not context: return ""
        
        if not self.DFRSMetrics_cls_ref: # DFRSMetrics Enumがロードできていない場合
            self.logger.error("DFRSMetricsV49 Enumが利用不可のため、フィードバックを生成できません。")
            return "(DFRS指標参照エラーにより位相/トーンフィードバック生成不可)"

        try:
            # アライメント閾値 (0-5スケールに変換)
            alignment_threshold_for_5_scale = self.alignment_threshold_01 * 5.0
            
            # intended_phase/tone は FeedbackContextV49 内で PsychologicalPhaseV49/EmotionalToneV49 Enumメンバーになっている想定
            intended_phase_str = context.intended_phase.value if context.intended_phase else '指定なし'
            inferred_phase_str = context.inferred_phase.value if context.inferred_phase else '不明'
            intended_tone_str = context.intended_tone.value if context.intended_tone else '指定なし'
            inferred_tone_str = context.inferred_tone.value if context.inferred_tone else '不明'
            
            phase_align_score = context.dfrs_scores.get(self.DFRSMetrics_cls_ref.PHASE_ALIGNMENT.value)
            tone_align_score = context.dfrs_scores.get(self.DFRSMetrics_cls_ref.TONE_ALIGNMENT.value)

            if phase_align_score is not None:
                if phase_align_score < alignment_threshold_for_5_scale * 0.8: # 閾値の80%未満で強い警告
                    feedback_parts.append(
                        f"⚠️ **位相のズレ大:** 意図された位相「{intended_phase_str}」に対し、文章から強く推定される位相は「{inferred_phase_str}」です (整合スコア: {fmt(phase_align_score)})。"
                        f"意図に沿うよう関連キーワードや描写を強化するか、意図自体を推定された位相に近づけることを検討してください。"
                    )
                elif phase_align_score < alignment_threshold_for_5_scale:
                    feedback_parts.append(
                        f"⚪️ **位相のズレ中:** 意図位相「{intended_phase_str}」と推定位相「{inferred_phase_str}」にやや乖離が見られます (整合スコア: {fmt(phase_align_score)})。"
                        f"意図する位相の雰囲気がより明確に伝わるような表現を意識してください。"
                    )

            if tone_align_score is not None:
                if tone_align_score < alignment_threshold_for_5_scale * 0.8:
                    feedback_parts.append(
                        f"⚠️ **感情トーンのズレ大:** 意図された感情「{intended_tone_str}」に対し、文章から強く推定される感情は「{inferred_tone_str}」です (整合スコア: {fmt(tone_align_score)})。"
                        f"「{intended_tone_str}」の感情がセリフ、行動、描写、内的反応などを通して一貫して表現されるよう見直してください。"
                    )
                elif tone_align_score < alignment_threshold_for_5_scale:
                     feedback_parts.append(
                        f"⚪️ **感情トーンのズレ中:** 意図感情「{intended_tone_str}」と推定感情「{inferred_tone_str}」にやや乖離が見られます (整合スコア: {fmt(tone_align_score)})。"
                        f"表現のニュアンスを調整し、意図する感情トーンをより明確にしましょう。"
                    )
            
            ecs_score = context.dfrs_scores.get(self.DFRSMetrics_cls_ref.ECS.value)
            if ecs_score is not None:
                if ecs_score < self.ecs_threshold_low:
                    feedback_parts.append(f"📉 **感情変化の単調さ:** 感情の起伏が少なく、シーンが平坦に感じられる可能性があります (ECSスコア: {fmt(ecs_score)})。感情のコントラストや変化を意識的に加えると、よりドラマチックな展開が期待できます。")
                elif ecs_score > self.ecs_threshold_high:
                    feedback_parts.append(f"📈 **感情変化の急激さ:** 感情の変化が急すぎたり、多すぎたりする印象です (ECSスコア: {fmt(ecs_score)})。感情の移行をより段階的に描写するか、描写の量を調整して自然な流れを目指しましょう。")

            ptn_score = context.dfrs_scores.get(self.DFRSMetrics_cls_ref.PTN.value)
            if ptn_score is not None and ptn_score < self.ptn_threshold_low:
                feedback_parts.append(f"🧭 **位相遷移の自然さ:** 物語の位相の移り変わりがやや唐突に感じられる部分があるかもしれません (PTNスコア: {fmt(ptn_score)})。次の位相への移行がより滑らかになるような「つなぎ」の描写や予兆を意識してください。")

        except AttributeError as e_attr: # Enumメンバーアクセスエラーなど
             self.logger.error(f"PhaseToneFeedback生成中に属性エラー: {e_attr}", exc_info=True)
             feedback_parts.append(f"(位相/感情フィードバック生成中に内部エラーが発生しました: 属性参照エラー)")
        except Exception as e_gen:
            self.logger.warning(f"PhaseToneFeedback生成中に予期せぬエラー: {e_gen}", exc_info=True)
            feedback_parts.append(f"(位相/感情フィードバック生成中に不明なエラー)")
            
        return "\n".join(f"- {part}" for part in feedback_parts if part) if feedback_parts else ""


class SubjectivityFeedbackStrategyV49: # Implicitly implements FeedbackStrategyProtocol
    """主観性に関するフィードバック生成戦略 (v4.9α 最適化版)"""
    if TYPE_CHECKING:
        ConfigRefType = ConfigProtoType
        FeedbackContextRefType = FeedbackContextV49Type
        TwoDimTempParamsRefType = TwoDimTempParamsType
    else:
        ConfigRefType = 'ConfigProtocol'
        FeedbackContextRefType = 'FeedbackContextV49'
        TwoDimTempParamsRefType = 'TwoDimensionalTemperatureParams'

    def __init__(self, config: ConfigRefType): # type: ignore
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        
        # 設定から閾値を取得 (TemperatureStrategyConfigV49.two_dimensional_params を参照)
        two_dim_params: Optional[TwoDimTempParamsRefType] = None # type: ignore
        if self.config.temperature_config and hasattr(self.config.temperature_config, 'two_dimensional_params'): # type: ignore
            two_dim_params = self.config.temperature_config.two_dimensional_params # type: ignore
        
        if two_dim_params:
            self.subj_low_thresh: float = getattr(two_dim_params, 'low_subjectivity_threshold', 3.0)
            self.subj_high_thresh: float = getattr(two_dim_params, 'high_subjectivity_threshold', 4.5)
            self.depth_low_thresh: float = getattr(two_dim_params, 'low_internal_depth_threshold', 2.8)
            self.complex_low_thresh: float = getattr(two_dim_params, 'low_emotion_complexity_threshold', 2.8)
            self.eti_low_thresh: float = getattr(two_dim_params, 'low_eti_threshold', 2.5)
            self.symbolic_low_thresh: float = getattr(two_dim_params, 'low_symbolic_threshold', 2.0)
        else:
            self.logger.warning("SubjectivityFeedback: TemperatureConfigのtwo_dimensional_paramsが見つからないため、デフォルト閾値を使用します。")
            self.subj_low_thresh, self.subj_high_thresh = 3.0, 4.5
            self.depth_low_thresh, self.complex_low_thresh, self.eti_low_thresh, self.symbolic_low_thresh = 2.8, 2.8, 2.5, 2.0
            
        self.DFRSMetrics_cls_ref: Optional[Type[DFRSMetricsV49EnumType]] = _get_global_type('DFRSMetricsV49', enum.EnumMeta) # type: ignore
        if not self.DFRSMetrics_cls_ref:
             self.logger.error("DFRSMetricsV49 Enumクラスがロードできませんでした。SubjectivityFeedbackStrategyが正しく機能しない可能性があります。")

        self.logger.debug(f"SubjectivityFeedback閾値: Subj(L/H)=({self.subj_low_thresh}/{self.subj_high_thresh}), Depth(L)={self.depth_low_thresh}, Complex(L)={self.complex_low_thresh}, ETI(L)={self.eti_low_thresh}, Symbolic(L)={self.symbolic_low_thresh}")

    def generate(self, context: FeedbackContextRefType) -> str: # type: ignore
        feedback_items: Dict[str, str] = {} # カテゴリごとの指摘を管理
        if not context or not context.dfrs_scores: return ""
        if not self.DFRSMetrics_cls_ref: return "(主観性FBエラー: DFRS指標定義なし)"

        dfrs = context.dfrs_scores
        try:
            # 主観描写の度合い (SUBJECTIVITY_SCORE)
            s_subj = dfrs.get(self.DFRSMetrics_cls_ref.SUBJECTIVITY_SCORE.value)
            if isinstance(s_subj, (int, float)):
                if s_subj < self.subj_low_thresh * 0.6: # 閾値の60%未満 (特に低い)
                    feedback_items["subjectivity_very_low"] = f"⚠️ **主観描写が著しく不足(スコア:{fmt(s_subj)})**: キャラクターの思考、感情、知覚、記憶などを積極的に描写し、内面世界を深く表現してください。"
                elif s_subj < self.subj_low_thresh:
                    feedback_items["subjectivity_low"] = f"🤔 **主観描写深化の余地(スコア:{fmt(s_subj)})**: 内的モノローグや感覚描写を増やし、キャラクターの視点を読者に伝えましょう。"
                elif s_subj > self.subj_high_thresh * 1.15: # 閾値の115%超 (特に高い)
                     feedback_items["subjectivity_very_high"] = f"🧐 **主観描写が過多かも(スコア:{fmt(s_subj)})**: 物語進行とのバランスを考慮し、客観描写や行動も適度に挟んでください。"
                elif s_subj > self.subj_high_thresh:
                    feedback_items["subjectivity_high"] = f"🧐 **主観描写バランス調整の検討(スコア:{fmt(s_subj)})**: 内面描写と客観描写のバランスを見直しましょう。"
            
            # 内面描写の深さ (INTERNAL_DEPTH)
            s_depth = dfrs.get(self.DFRSMetrics_cls_ref.INTERNAL_DEPTH.value)
            if isinstance(s_depth, (int, float)) and s_depth < self.depth_low_thresh:
                feedback_items["internal_depth_low"] = f"🧠 **内面掘り下げ不足(スコア:{fmt(s_depth)})**: 行動の背景にある動機や葛藤、過去の経験の影響などをより具体的に描写し、深みを与えましょう。"

            # 感情の複雑性 (EMOTION_COMPLEXITY)
            s_complex = dfrs.get(self.DFRSMetrics_cls_ref.EMOTION_COMPLEXITY.value)
            if isinstance(s_complex, (int, float)) and s_complex < self.complex_low_thresh:
                 feedback_items["emotion_complexity_low"] = f"🎭 **感情表現の単調さ(スコア:{fmt(s_complex)})**: 喜びと不安など、複数の感情が混在する複雑な心理状態を描写し、キャラクターに人間味を与えましょう。"

            # 感情的相互作用 (ETI) - 低すぎる場合に指摘
            s_eti = dfrs.get(self.DFRSMetrics_cls_ref.ETI.value)
            if isinstance(s_eti, (int, float)) and s_eti < self.eti_low_thresh:
                feedback_items["eti_low"] = f"💬 **感情的相互作用の不足(スコア:{fmt(s_eti)})**: キャラクター間の感情的なやり取りや反応が少ないようです。感情がぶつかり合うようなシーンや、共感・反発を示す描写を増やすと、関係性がより鮮明になります。"

            # 象徴密度 (SYMBOLIC_DENSITY) - 低すぎる場合に指摘
            s_sym = dfrs.get(self.DFRSMetrics_cls_ref.SYMBOLIC_DENSITY.value)
            if isinstance(s_sym, (int, float)) and s_sym < self.symbolic_low_thresh:
                feedback_items["symbolic_density_low"] = f"🔮 **象徴性の不足(スコア:{fmt(s_sym)})**: 物語に深みを与える象徴的なアイテム、言葉、情景などが少ないようです。テーマ性やキャラクター心理を暗示する象徴を効果的に配置することで、読者の解釈の幅が広がります。"

        except AttributeError as e_attr_subj:
            self.logger.error(f"SubjectivityFeedback生成中に属性エラー: {e_attr_subj}", exc_info=True)
            feedback_items["error"] = "(主観性フィードバック生成エラー: 属性参照エラー)"
        except Exception as e_gen_subj:
            self.logger.warning(f"SubjectivityFeedback生成中に予期せぬエラー: {e_gen_subj}", exc_info=True)
            if not feedback_items: feedback_items["error"] = "(主観性フィードバック生成中に不明なエラー)"
        
        if not feedback_items:
            feedback_items["all_good_subj"] = "主観性関連の指標は概ね良好です。この調子でキャラクターの内面を豊かに描き出してください。"
            
        return "\n".join(f"- {msg}" for msg in feedback_items.values() if msg)


class FluctuationFeedbackStrategyV49: # Implicitly implements FeedbackStrategyProto
    """揺らぎ表現に関するフィードバック生成戦略 (v4.9α 最適化版)"""
    if TYPE_CHECKING:
        ConfigRefType = ConfigProtoType
        FeedbackContextRefType = FeedbackContextV49Type
        TwoDimTempParamsRefType = TwoDimTempParamsType
    else:
        ConfigRefType = 'ConfigProtocol'
        FeedbackContextRefType = 'FeedbackContextV49'
        TwoDimTempParamsRefType = 'TwoDimensionalTemperatureParams'

    def __init__(self, config: ConfigRefType): # type: ignore
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        
        two_dim_params: Optional[TwoDimTempParamsRefType] = None # type: ignore
        if self.config.temperature_config and hasattr(self.config.temperature_config, 'two_dimensional_params'): # type: ignore
            two_dim_params = self.config.temperature_config.two_dimensional_params # type: ignore
        
        if two_dim_params:
            self.fluc_low_thresh: float = getattr(two_dim_params, 'low_fluctuation_threshold', 2.5)
            self.fluc_high_thresh: float = getattr(two_dim_params, 'high_fluctuation_threshold', 4.2)
        else:
            self.logger.warning("FluctuationFeedback: TemperatureConfigのtwo_dimensional_paramsが見つからないため、デフォルト閾値を使用。")
            self.fluc_low_thresh, self.fluc_high_thresh = 2.5, 4.2
            
        self.DFRSMetrics_cls_ref: Optional[Type[DFRSMetricsV49EnumType]] = _get_global_type('DFRSMetricsV49', enum.EnumMeta) # type: ignore
        if not self.DFRSMetrics_cls_ref:
            self.logger.error("DFRSMetricsV49 Enumクラスがロードできませんでした。FluctuationFeedbackStrategyが正しく機能しない可能性があります。")

        self.logger.debug(f"FluctuationFeedback閾値: Fluc(L/H)=({self.fluc_low_thresh}/{self.fluc_high_thresh})")

    def generate(self, context: FeedbackContextRefType) -> str: # type: ignore
        feedback_parts: List[str] = []
        if not context or not context.dfrs_scores: return ""
        if not self.DFRSMetrics_cls_ref: return "(揺らぎFBエラー: DFRS指標定義なし)"
        
        try:
            fluc_intensity_score = context.dfrs_scores.get(self.DFRSMetrics_cls_ref.FLUCTUATION_INTENSITY.value)
            if isinstance(fluc_intensity_score, (int, float)):
                s_fluc_display = fmt(fluc_intensity_score)
                if fluc_intensity_score < self.fluc_low_thresh:
                    feedback_parts.append(f"🌊 **表現の揺らぎ追加提案 (スコア:{s_fluc_display})**: 会話や描写に、ためらい、言い淀み、思考の断片、行動の僅かな変化などの「揺らぎ」をもう少し加えることで、キャラクターの人間らしさやシーンのリアリティが増すかもしれません。")
                elif fluc_intensity_score > self.fluc_high_thresh:
                    feedback_parts.append(f"🌪️ **表現の揺らぎ過多の可能性 (スコア:{s_fluc_display})**: 表現の揺らぎがやや過剰で、読者が状況を掴みづらくなったり、物語のテンポが損なわれたりする可能性があります。意図的でない限り、揺らぎの頻度や強度を調整し、より明確な描写とのバランスを検討してください。")
            else:
                 feedback_parts.append(f"🌊 **表現の揺らぎの検討**: 揺らぎ強度スコアが不明です。キャラクターの感情や思考の揺れ動きを、セリフや行動の細かな変化で表現することを検討してみてください。")
        except AttributeError as e_attr_fluc:
            self.logger.error(f"FluctuationFeedback生成中に属性エラー: {e_attr_fluc}", exc_info=True)
            feedback_parts.append(f"(揺らぎフィードバック生成中に内部エラーが発生しました: 属性参照エラー)")
        except Exception as e_gen_fluc:
            self.logger.warning(f"FluctuationFeedback生成中に予期せぬエラー: {e_gen_fluc}", exc_info=True)
            feedback_parts.append(f"(揺らぎフィードバック生成中に不明なエラー)")
            
        return "\n".join(f"- {part}" for part in feedback_parts if part) if feedback_parts else ""

class QualityFeedbackStrategyV49: # Implicitly implements FeedbackStrategyProto
    """内容・品質に関するフィードバック生成戦略 (v4.9α 最適化版)"""
    if TYPE_CHECKING:
        ConfigRefType = ConfigProtoType
        FeedbackContextRefType = FeedbackContextV49Type
        TwoDimTempParamsRefType = TwoDimTempParamsType
    else:
        ConfigRefType = 'ConfigProtocol'
        FeedbackContextRefType = 'FeedbackContextV49'
        TwoDimTempParamsRefType = 'TwoDimensionalTemperatureParams'

    def __init__(self, config: ConfigRefType): # type: ignore
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")

        two_dim_params: Optional[TwoDimTempParamsRefType] = None # type: ignore
        if self.config.temperature_config and hasattr(self.config.temperature_config, 'two_dimensional_params'): # type: ignore
            two_dim_params = self.config.temperature_config.two_dimensional_params # type: ignore
        
        if two_dim_params:
            self.novelty_low_thresh: float = getattr(two_dim_params, 'low_novelty_threshold', 3.0)
            self.richness_low_thresh: float = getattr(two_dim_params, 'low_richness_threshold', 3.5)
        else:
            self.logger.warning("QualityFeedback: TemperatureConfigのtwo_dimensional_paramsが見つからないため、デフォルト閾値を使用。")
            self.novelty_low_thresh, self.richness_low_thresh = 3.0, 3.5
            
        self.DFRSMetrics_cls_ref: Optional[Type[DFRSMetricsV49EnumType]] = _get_global_type('DFRSMetricsV49', enum.EnumMeta) # type: ignore
        if not self.DFRSMetrics_cls_ref:
            self.logger.error("DFRSMetricsV49 Enumクラスがロードできませんでした。QualityFeedbackStrategyが正しく機能しない可能性があります。")

        self.logger.debug(f"QualityFeedback閾値: Novelty(L)={self.novelty_low_thresh}, Richness(L)={self.richness_low_thresh}")

    def generate(self, context: FeedbackContextRefType) -> str: # type: ignore
        feedback_parts: List[str] = []
        if not context or not context.dfrs_scores: return ""
        if not self.DFRSMetrics_cls_ref: return "(品質FBエラー: DFRS指標定義なし)"
        
        dfrs = context.dfrs_scores
        try:
            s_novel = dfrs.get(self.DFRSMetrics_cls_ref.CONTENT_NOVELTY.value)
            if isinstance(s_novel, (int, float)) and s_novel < self.novelty_low_thresh:
                feedback_parts.append(f"🎲 **内容の新規性向上提案 (スコア:{fmt(s_novel)})**: 物語の展開やキャラクターの反応が予測可能に感じられるかもしれません。読者を驚かせるような意外な要素、新しい情報の提示、キャラクターの予期せぬ一面などを盛り込み、より独創的な内容を目指しましょう。")

            s_rich = dfrs.get(self.DFRSMetrics_cls_ref.EXPRESSION_RICHNESS.value)
            if isinstance(s_rich, (int, float)) and s_rich < self.richness_low_thresh:
                feedback_parts.append(f"🎨 **表現の豊かさ向上提案 (スコア:{fmt(s_rich)})**: 言葉選びや文構造がやや単調になっている可能性があります。類語、比喩、多様な感覚描写、文の長短の変化などを意識し、より彩り豊かで読者の五感に訴えかける表現を目指してください。")
            
            # LLM評価スコアに基づくフィードバック (例: overall)
            llm_overall_score = context.llm_scores.get(ScoreKeysLLMEnumType.OVERALL.value) # type: ignore
            if isinstance(llm_overall_score, (int, float)) and llm_overall_score < 3.0: # 閾値は調整可能
                feedback_parts.append(f"⭐ **総合品質の向上検討 (LLM評価:{fmt(llm_overall_score)})**: LLMによる総合評価がやや低めです。物語の魅力、キャラクターの行動原理、対話の自然さなど、全体的な質の向上を目指し、具体的な問題点があればそれらを修正してください。")

        except AttributeError as e_attr_qual:
            self.logger.error(f"QualityFeedback生成中に属性エラー: {e_attr_qual}", exc_info=True)
            feedback_parts.append(f"(品質フィードバック生成中に内部エラーが発生しました: 属性参照エラー)")
        except Exception as e_gen_qual:
            self.logger.warning(f"QualityFeedback生成中に予期せぬエラー: {e_gen_qual}", exc_info=True)
            if not feedback_parts: feedback_parts.append(f"(品質フィードバック生成中に不明なエラー)")
            
        return "\n".join(f"- {part}" for part in feedback_parts if part) if feedback_parts else ""


# --- Composite Strategy ---
class CompositeFeedbackStrategyV49: # Implicitly implements FeedbackStrategyProtocol
    """複数のフィードバック戦略を組み合わせて実行する戦略 (v4.9α 最適化版)"""
    if TYPE_CHECKING: FeedbackContextRefType = FeedbackContextV49Type; FeedbackStrategyProtoType = FeedbackStrategyProtocol # type: ignore
    else: FeedbackContextRefType = 'FeedbackContextV49'; FeedbackStrategyProtoType = 'FeedbackStrategyProtocol'

    def __init__(self, individual_strategies: List[FeedbackStrategyProtoType]): # type: ignore
        if not individual_strategies:
            raise ValueError("CompositeFeedbackStrategyには、少なくとも1つの個別戦略が必要です。")
        self.strategies = individual_strategies
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        self.logger.info(f"CompositeFeedbackStrategy初期化完了。構成戦略数: {len(self.strategies)}")

    def generate(self, context: FeedbackContextRefType) -> str: # type: ignore
        all_feedback_lines: List[str] = []
        seen_normalized_feedback: set[str] = set() # 重複フィードバックを避けるためのセット
        
        self.logger.debug(f"複合フィードバック生成開始。対象バージョン: v{context.version if context else 'N/A'}")
        
        for strategy_instance in self.strategies:
            strategy_name = strategy_instance.__class__.__name__
            try:
                feedback_from_strategy = strategy_instance.generate(context)
                if feedback_from_strategy and feedback_from_strategy.strip():
                    # 個別フィードバックを改行で分割し、主要部分のみを抽出・正規化して重複チェック
                    for line in feedback_from_strategy.splitlines():
                        cleaned_line = line.strip().lstrip('- ') # 先頭のハイフンやスペース除去
                        if cleaned_line:
                            normalized_line_for_dedup = re.sub(r'\s+', ' ', cleaned_line).lower() # 空白正規化と小文字化
                            if normalized_line_for_dedup not in seen_normalized_feedback:
                                all_feedback_lines.append(f"- {cleaned_line}") # 箇条書き形式で追加
                                seen_normalized_feedback.add(normalized_line_for_dedup)
                                self.logger.debug(f"  戦略 '{strategy_name}' からフィードバック追加: {cleaned_line[:80]}...")
                            else:
                                self.logger.debug(f"  戦略 '{strategy_name}' からのフィードバックは重複のためスキップ: {cleaned_line[:80]}...")
            except Exception as e_strat_gen:
                self.logger.error(f"個別フィードバック戦略 '{strategy_name}' の実行中にエラー: {e_strat_gen}", exc_info=True)
        
        if not all_feedback_lines:
            return "(特に具体的な改善点の指摘はありません。全体的に良好か、分析対象外の可能性があります。)"
        
        final_composite_feedback = "**総合改善提案:**\n" + "\n".join(all_feedback_lines)
        self.logger.info(f"複合フィードバック生成完了。提案項目数: {len(all_feedback_lines)}")
        self.logger.debug(f"最終複合フィードバック (プレビュー):\n{final_composite_feedback[:300]}...")
        return final_composite_feedback


# --- Feedback Manager ---
class FeedbackManagerV49: # Implicitly implements FeedbackManagerProtocol
    """フィードバック生成戦略を管理し、設定に基づいて適切な戦略を実行します (v4.9α 最適化版)"""
    if TYPE_CHECKING:
        ConfigRefType = ConfigProtoType
        FeedbackContextRefType = FeedbackContextV49Type
        FeedbackStrategyProtoType = FeedbackStrategyProtocol
    else:
        ConfigRefType = 'ConfigProtocol'
        FeedbackContextRefType = 'FeedbackContextV49'
        FeedbackStrategyProtoType = 'FeedbackStrategyProtocol'

    def __init__(self, config: ConfigRefType): # type: ignore
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        self.available_strategies: Dict[str, FeedbackStrategyProtoType] = self._create_and_register_strategies() # type: ignore
        
        # CompositeStrategy は、利用可能な全戦略を使って動的に生成
        self.composite_strategy: Optional[CompositeFeedbackStrategyV49] = None
        if self.available_strategies:
            self.composite_strategy = CompositeFeedbackStrategyV49(list(self.available_strategies.values()))
        else:
            self.logger.warning("利用可能な個別フィードバック戦略がないため、複合戦略は作成されませんでした。")

        self.logger.info(
            f"FeedbackManagerV49初期化完了。利用可能戦略: {list(self.available_strategies.keys())}"
            f"{'+composite' if self.composite_strategy else ''}"
        )

    def _create_and_register_strategies(self) -> Dict[str, FeedbackStrategyProtoType]: # type: ignore
        """設定に基づいて利用可能なフィードバック戦略のインスタンスを作成・登録します。"""
        strategies_to_create: Dict[str, FeedbackStrategyProtoType] = {} # type: ignore
        
        # 戦略キーと対応するクラス名のマッピング
        strategy_class_map: Dict[str, Type[FeedbackStrategyProtoType]] = { # type: ignore
            "phase_tone": PhaseToneFeedbackStrategyV49,
            "subjectivity": SubjectivityFeedbackStrategyV49,
            "fluctuation": FluctuationFeedbackStrategyV49,
            "quality": QualityFeedbackStrategyV49,
        }
        
        for strategy_key, strategy_class_obj in strategy_class_map.items():
            try:
                # 各戦略クラスのインスタンスを作成 (config を渡す)
                strategies_to_create[strategy_key] = strategy_class_obj(self.config) # type: ignore
                self.logger.debug(f"フィードバック戦略 '{strategy_key}' ({strategy_class_obj.__name__}) を正常に初期化・登録しました。")
            except Exception as e_create_strat:
                self.logger.error(f"フィードバック戦略 '{strategy_key}' ({strategy_class_obj.__name__}) の初期化に失敗しました: {e_create_strat}", exc_info=True)
        
        return strategies_to_create

    def get_feedback(self, context: FeedbackContextRefType, strategy_key_override: Optional[str] = None) -> str: # type: ignore
        """
        指定された戦略キー、または設定ファイルで定義されたデフォルト戦略に基づいてフィードバックを生成します。
        """
        # FeedbackStrategyConfigV49 からデフォルト戦略キーを取得
        default_strategy_key = "context_aware" # フォールバック
        if self.config.feedback_config: # type: ignore
            default_strategy_key = getattr(self.config.feedback_config, 'strategy_type', "context_aware") # type: ignore
        
        target_strategy_key = strategy_key_override if strategy_key_override else default_strategy_key
        
        self.logger.info(f"フィードバック要求受信: 戦略キー='{target_strategy_key}', 対象バージョン=v{context.version if context else 'N/A'}")

        if target_strategy_key in ["composite", "context_aware"]: # "context_aware" も複合戦略として扱う
            if self.composite_strategy:
                self.logger.debug(f"複合戦略 '{target_strategy_key}' を使用してフィードバックを生成します。")
                return self.composite_strategy.generate(context)
            else:
                self.logger.error("複合戦略が利用不可です。空のフィードバックを返します。")
                return "(エラー: 複合フィードバック戦略が初期化されていません)"
        elif target_strategy_key in self.available_strategies:
            selected_strategy = self.available_strategies[target_strategy_key]
            strategy_name = selected_strategy.__class__.__name__
            self.logger.debug(f"個別戦略 '{target_strategy_key}' ({strategy_name}) を使用してフィードバックを生成します。")
            try:
                return selected_strategy.generate(context)
            except Exception as e_single_strat_gen:
                self.logger.error(f"個別戦略 '{target_strategy_key}' ({strategy_name}) の実行中にエラー: {e_single_strat_gen}", exc_info=True)
                return f"(エラー: フィードバック戦略 '{target_strategy_key}' の実行に失敗しました)"
        else: # 未知の戦略キーの場合
            self.logger.warning(f"未知のフィードバック戦略キー '{target_strategy_key}' が指定されました。デフォルトの複合戦略を使用します。")
            if self.composite_strategy:
                return self.composite_strategy.generate(context)
            else:
                self.logger.error("未知の戦略キーが指定され、かつ複合戦略も利用不可です。")
                return "(エラー: 指定されたフィードバック戦略が見つからず、フォールバックもできませんでした)"

# =============================================================================
# -- Part 9 終了点
# =============================================================================
# =============================================================================
# -- Part 10: Dialogue Settings & Style Manager (v4.9α - 改善版 v2.2 - Standard Style Optimized & Stability Fix)
# =============================================================================
# v4.9α: ジョブ固有設定クラスと対話スタイルマネージャクラス。
#        ConfigProtocolを参照し、設定値の取得方法を更新。
# 改善版: Config参照の明確化、引数パース処理の改善、Enumキー変換強化。
# v2変更点: (v2の安定版をベースとする)
# - DialogueSettingsV49:
#   - __init__: ログメッセージ調整、バージョン情報付加。
#   - copy_and_validate_weights: Enumキー変換ロジックのログ強化。
#   - update_from_args: 型変換の堅牢性向上、ログ追加。
#   - model_dump: コメント追加、既存ロジック維持。
# - DialogStyleManagerV49:
#   - __init__, load_from_file: ログメッセージ調整。
#   - suggest_style_for_scene: analysis_resultsからのdominant_phase取得ロジックを改善。
# v2.2 Update:
# - DialogStyleManagerV49.__init__:
#   - AttributeError: '_config' の問題を修正。v2同様に self.config を一貫して使用。
# - DialogStyleManagerV49._load_base_templates:
#   - "standard" スタイルの prompt_additions を【決定版プロンプト Part 5】および「究極指針」に基づき全面的に刷新。
#     現代読者向けの読みやすさ、漫画的魅力と小説的深みの両立、Show/Tellバランス、
#     効果的なリズム・語彙・オノマトペ・間合いの活用を指示する内容に変更。
# - 既存メソッドの完全性を維持し、v2の構造を尊重。
# - ロガー名に .v2.2 を付加。
# - load_json_utility_func の参照を globals().get('load_json_file', globals().get('load_json')) に統一。

from typing import TYPE_CHECKING, Counter as TypingCounter, Set, List, Dict, Optional, Tuple, Callable, Type, Union, Any
import enum
import pathlib
import hashlib
import json
import math
import statistics
import re
import logging
import copy
import argparse
from collections import defaultdict
from datetime import datetime

# --- グローバルスコープで利用可能であることを期待する変数 (Part 0 などで定義済み) ---
_get_global_type_func_pt10: Optional[Callable[[str, Optional[type]], Optional[Type[Any]]]] = globals().get('_get_global_type')
JSONSCHEMA_AVAILABLE = globals().get('JSONSCHEMA_AVAILABLE', False)
# Part 2の load_json_file を想定 (フォールバックとして load_json も許容)
load_json_utility_func: Optional[Callable[[Union[str, pathlib.Path]], Optional[Any]]] = \
    globals().get('load_json_file', globals().get('load_json'))


if TYPE_CHECKING:
    from __main__ import ( # type: ignore[attr-defined]
        ConfigProtocol, SettingsProtocol, FeedbackStrategyProtocol, StyleManagerProtocol,
        SubjectiveIntensityLevel, PsychologicalPhaseV49, EmotionalToneV49,
        SubjectivityCategoryV49, FluctuationCategoryV49, ScoreKeys, DFRSMetricsV49,
        FinalSelectionKeysV49, InitialSelectionKeysV49,
        FeedbackContextV49, SubjectivityKeywordEntryV49, FluctuationPatternEntryV49,
        BaseModel, ValidationError, PYDANTIC_AVAILABLE, AppConfigV49
    )
    from typing import TypeAlias
    ConfigProto: TypeAlias = ConfigProtocol
    SubjectiveIntensityLevelType: TypeAlias = SubjectiveIntensityLevel
    DFRSMetricsEnumType: TypeAlias = DFRSMetricsV49
    FinalSelectionKeysEnumType: TypeAlias = FinalSelectionKeysV49
    InitialSelectionKeysEnumType: TypeAlias = InitialSelectionKeysV49
    SceneInfoDictTypeRef: TypeAlias = Dict[str, Any]

else:
    ConfigProto = 'ConfigProtocol'
    SubjectiveIntensityLevelType = 'SubjectiveIntensityLevel'
    DFRSMetricsEnumType = 'DFRSMetricsV49'
    FinalSelectionKeysEnumType = 'FinalSelectionKeysV49'
    InitialSelectionKeysEnumType = 'InitialSelectionKeysV49'
    SceneInfoDictTypeRef = Dict[str, Any]

# =============================================================================
# -- Dialogue Settings (v4.9α - 改善版 v2 準拠)
# =============================================================================
class DialogueSettingsV49: # Implicitly implements SettingsProtocol
    """個別の対話生成ジョブ設定 (v4.9α - 改善版 v2)"""

    def __init__(self, config: ConfigProto): # type: ignore
        self.config: DialogueSettingsV49.ConfigProto = config # type: ignore # _config ではなく config を使用
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}.v2.2") # Version updated
        system_version_settings = getattr(self.config, 'SYSTEM_VERSION', 'N/A_DlgSettings_v2.2')
        self.logger.info(f"DialogueSettingsV49 (System Version: {system_version_settings}) 初期化中...")

        ff_cfg_source = getattr(self.config, 'feature_flags', None)
        adapt_cfg_source = getattr(self.config, 'adaptation_config', None)
        fb_cfg_source = getattr(self.config, 'feedback_config', None)
        ext_cfg_source = getattr(self.config, 'loaded_external_configs', None)

        def get_config_attribute(source_object: Optional[Any], attribute_key: str, default_val: Any) -> Any:
            return getattr(source_object, attribute_key, default_val) if source_object else default_val

        self.dialogue_mode: str = get_config_attribute(ff_cfg_source, 'dialogue_mode', "auto")
        self.dfrs_evaluation_enabled: bool = get_config_attribute(ff_cfg_source, 'dfrs_evaluation_enabled', True)
        self.dfrs_evaluate_all_loops: bool = get_config_attribute(ff_cfg_source, 'dfrs_evaluate_all_loops', True)
        self.dfrs_for_initial_selection: bool = get_config_attribute(ff_cfg_source, 'dfrs_for_initial_selection', True)
        self.json_export_enabled: bool = get_config_attribute(ff_cfg_source, 'json_export_enabled', True)
        self.json_schema_validation: bool = get_config_attribute(ff_cfg_source, 'json_schema_validation', True) and JSONSCHEMA_AVAILABLE
        self.advanced_nlp_enabled: bool = get_config_attribute(ff_cfg_source, 'advanced_nlp_enabled', True)
        self.ml_emotion_enabled: bool = get_config_attribute(ff_cfg_source, 'ml_emotion_enabled', False)
        self.phase_tagging_enabled: bool = get_config_attribute(ff_cfg_source, 'phase_tagging_enabled', True)
        self.persistent_cache_enabled: bool = get_config_attribute(ff_cfg_source, 'persistent_cache_enabled', True)
        self.subjective_focus: bool = get_config_attribute(ff_cfg_source, 'subjective_focus_enabled', True)
        self.phase_tone_prompt_modulation_enabled: bool = get_config_attribute(ff_cfg_source, 'phase_tone_prompt_modulation_enabled', True)
        self.save_prompts: bool = get_config_attribute(ff_cfg_source, 'save_prompts', False)
        self.save_evaluations: bool = get_config_attribute(ff_cfg_source, 'save_evaluations', False)
        self.save_rejected_candidates: bool = get_config_attribute(ff_cfg_source, 'save_rejected_candidates', False)

        self.adaptation_strategy_enabled: bool = get_config_attribute(adapt_cfg_source, 'enabled', True)
        self.adaptation_strategy_type: str = get_config_attribute(adapt_cfg_source, 'strategy_type', "probabilistic_history")
        self.log_phase_tone_transitions: bool = get_config_attribute(adapt_cfg_source, 'log_transitions', True)

        self.feedback_strategy_type: str = get_config_attribute(fb_cfg_source, 'strategy_type', "context_aware")

        self.style_template: str = get_config_attribute(ext_cfg_source, 'default_style_template', "standard")
        file_settings_from_ext_cfg = get_config_attribute(ext_cfg_source, 'file_settings', None)
        custom_style_file_path_raw_val = get_config_attribute(file_settings_from_ext_cfg, 'custom_style_file_path', None)
        self.custom_style_file_path: Optional[str] = str(custom_style_file_path_raw_val) if custom_style_file_path_raw_val else None
        
        self.nlp_model_name: str = get_config_attribute(ext_cfg_source, 'nlp_model_name', "ja_core_news_lg")
        self.ml_emotion_model: Optional[str] = get_config_attribute(ext_cfg_source, 'ml_emotion_model', None)
        self.use_lightweight_ml_model: bool = get_config_attribute(ext_cfg_source, 'use_lightweight_ml_model', False)

        SubjectiveIntensityLevel_cls_local: Optional[Type[SubjectiveIntensityLevelType]] = _get_global_type_func_pt10('SubjectiveIntensityLevel', enum.EnumMeta) if _get_global_type_func_pt10 else None # type: ignore
        default_subjective_intensity_str_val = get_config_attribute(ext_cfg_source, 'default_subjective_intensity', 'medium')
        if SubjectiveIntensityLevel_cls_local:
            try:
                self.subjective_intensity: SubjectiveIntensityLevelType = SubjectiveIntensityLevel_cls_local(str(default_subjective_intensity_str_val).lower()) # type: ignore
            except ValueError:
                self.logger.warning(f"ExternalConfigsの主観強度 '{default_subjective_intensity_str_val}' 不正。MEDIUM使用。")
                self.subjective_intensity = SubjectiveIntensityLevel_cls_local.MEDIUM # type: ignore
        else:
            self.logger.error("SubjectiveIntensityLevel Enum未ロード。主観強度 'medium' (文字列)設定。")
            self.subjective_intensity = 'medium' # type: ignore

        if hasattr(self.config, 'cache_dir') and isinstance(self.config.cache_dir, pathlib.Path): # type: ignore
            self.cache_dir: pathlib.Path = self.config.cache_dir # type: ignore
        else:
            self.logger.warning("AppConfigに有効な 'cache_dir' (Path型) 属性なし。フォールバックパス使用。")
            AppConfigV49_cls_ref_for_cache: Optional[Type[Any]] = _get_global_type_func_pt10('AppConfigV49') if _get_global_type_func_pt10 else None
            fallback_cache_dir_path_str = getattr(AppConfigV49_cls_ref_for_cache, 'CACHE_DIR_STR', "./cache/ndgs_v49_settings_fallback_cache") if AppConfigV49_cls_ref_for_cache else "./cache/ndgs_v49_settings_fallback_cache"
            self.cache_dir = pathlib.Path(fallback_cache_dir_path_str)
            if AppConfigV49_cls_ref_for_cache and hasattr(AppConfigV49_cls_ref_for_cache, '_ensure_dir_exists'):
                AppConfigV49_cls_ref_for_cache._ensure_dir_exists(self.cache_dir) # type: ignore
            else:
                try: self.cache_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e_mkdir: self.logger.error(f"フォールバックキャッシュディレクトリ '{self.cache_dir}' 作成失敗: {e_mkdir}")

        self.auto_normalize_weights: bool = False

        DFRSMetrics_enum_cls_local: Optional[Type[DFRSMetricsEnumType]] = _get_global_type_func_pt10('DFRSMetricsV49', enum.EnumMeta) if _get_global_type_func_pt10 else None # type: ignore
        FinalSelectionKeys_enum_cls_local: Optional[Type[FinalSelectionKeysEnumType]] = _get_global_type_func_pt10('FinalSelectionKeysV49', enum.EnumMeta) if _get_global_type_func_pt10 else None # type: ignore
        InitialSelectionKeys_enum_cls_local: Optional[Type[InitialSelectionKeysEnumType]] = _get_global_type_func_pt10('InitialSelectionKeysV49', enum.EnumMeta) if _get_global_type_func_pt10 else None # type: ignore

        dfrs_weights_source_data = getattr(self.config, 'dfrs_weights', {})
        
        selection_weights_config_obj = get_config_attribute(ext_cfg_source, 'selection_weights_config', None)
        
        final_selection_weights_from_yaml_raw: Dict[str, float] = {}
        initial_selection_weights_from_yaml_raw: Dict[str, float] = {}
        if selection_weights_config_obj:
            final_weights_obj = getattr(selection_weights_config_obj, 'final', None)
            if final_weights_obj and hasattr(final_weights_obj, 'weights'):
                final_selection_weights_from_yaml_raw = getattr(final_weights_obj, 'weights', {})
            
            initial_weights_obj = getattr(selection_weights_config_obj, 'initial', None)
            if initial_weights_obj and hasattr(initial_weights_obj, 'weights'):
                initial_selection_weights_from_yaml_raw = getattr(initial_weights_obj, 'weights', {})

        default_final_weights_as_str_keys: Dict[str, float] = {}
        if FinalSelectionKeys_enum_cls_local:
            default_final_weights_as_str_keys = {
                member.value: 0.5 for member in FinalSelectionKeys_enum_cls_local # type: ignore
                if hasattr(member, 'name') and member.name != 'UNKNOWN' and hasattr(member, 'value')
            }
        
        default_initial_weights_as_str_keys: Dict[str, float] = {}
        if InitialSelectionKeys_enum_cls_local:
            default_initial_weights_as_str_keys = {
                member.value: 0.5 for member in InitialSelectionKeys_enum_cls_local # type: ignore
                if hasattr(member, 'name') and member.name != 'UNKNOWN' and hasattr(member, 'value')
            }

        self.enhanced_dfrs_weights: Dict[DFRSMetricsEnumType, float] = self.copy_and_validate_weights( # type: ignore
            dfrs_weights_source_data, DFRSMetrics_enum_cls_local, "DFRS Weights (enhanced_dfrs_weights)"
        )
        self.final_selection_weights: Dict[FinalSelectionKeysEnumType, float] = self.copy_and_validate_weights( # type: ignore
            {**default_final_weights_as_str_keys, **final_selection_weights_from_yaml_raw},
            FinalSelectionKeys_enum_cls_local, "Final Selection Weights",
            auto_normalize=True
        )
        self._validated_initial_candidate_weights: Dict[InitialSelectionKeysEnumType, float] = self.copy_and_validate_weights( # type: ignore
            {**default_initial_weights_as_str_keys, **initial_selection_weights_from_yaml_raw},
            InitialSelectionKeys_enum_cls_local, "Initial Candidate Weights",
            auto_normalize=self.auto_normalize_weights
        )

        generation_params_from_cfg = get_config_attribute(ext_cfg_source, 'generation_params', None)
        
        default_generation_params_map = {
            "target_length": 4000, "feedback_loops": 3,
            "min_feedback_loops": 1, "min_score_threshold": 4.5
        }
        
        current_generation_params_to_use: Dict[str, Any]
        if isinstance(generation_params_from_cfg, dict):
            current_generation_params_to_use = generation_params_from_cfg
        else:
            current_generation_params_to_use = default_generation_params_map
            if generation_params_from_cfg is not None:
                self.logger.warning(f"ExternalConfigs.generation_params が予期しない型 ({type(generation_params_from_cfg)})。デフォルト値使用。")

        self.feedback_loops: int = int(current_generation_params_to_use.get('feedback_loops', default_generation_params_map['feedback_loops']))
        self.min_feedback_loops: int = int(current_generation_params_to_use.get('min_feedback_loops', default_generation_params_map['min_feedback_loops']))
        self.min_score_threshold: float = float(current_generation_params_to_use.get('min_score_threshold', default_generation_params_map['min_score_threshold']))
        self.target_length: int = int(current_generation_params_to_use.get('target_length', default_generation_params_map['target_length']))
        
        self.logger.info(f"DialogueSettingsV49 初期化完了。目標文字数: {self.target_length}, FBループ: {self.feedback_loops}, 最低スコア閾値: {self.min_score_threshold}")

    @property
    def config_property(self) -> ConfigProto: # Renamed from 'config' to avoid direct attribute conflict warning in some linters
        return self.config # Access the stored config

    @property
    def initial_candidate_weights(self) -> Dict[InitialSelectionKeysEnumType, float]: # type: ignore
        return self._validated_initial_candidate_weights.copy()

    def copy_and_validate_weights(
        self,
        source_weights_data: Optional[Dict[Any, Any]],
        target_enum_class_ref: Optional[Type[enum.Enum]],
        weights_set_name: str,
        auto_normalize: Optional[bool] = None
    ) -> Dict[enum.Enum, float]:
        logger_cw = self.logger.getChild(f"copy_and_validate_weights.{weights_set_name.replace(' ', '_')}")
        final_auto_normalize_flag = auto_normalize if auto_normalize is not None else self.auto_normalize_weights

        if source_weights_data is None or not (target_enum_class_ref and isinstance(target_enum_class_ref, enum.EnumMeta)):
            logger_cw.warning(f"'{weights_set_name}': 入力データまたは対象Enumクラス不正。空辞書返却。")
            return {}
        if not isinstance(source_weights_data, dict):
            logger_cw.error(f"'{weights_set_name}': 入力データ非辞書型(型: {type(source_weights_data)})。処理中止。")
            return {}

        enum_value_to_member_lookup: Dict[str, enum.Enum] = {
            str(member.value).lower(): member for member in target_enum_class_ref if hasattr(member, 'value')
        }
        enum_name_to_member_lookup: Dict[str, enum.Enum] = {
            member.name.lower(): member for member in target_enum_class_ref if hasattr(member, 'name')
        }
        
        processed_enum_keyed_weights: Dict[enum.Enum, float] = {}
        for key_from_source_data, value_from_source_data in source_weights_data.items():
            resolved_enum_member: Optional[enum.Enum] = None
            
            if isinstance(key_from_source_data, target_enum_class_ref):
                resolved_enum_member = key_from_source_data
            elif isinstance(key_from_source_data, str):
                normalized_key_string = key_from_source_data.strip().lower()
                resolved_enum_member = enum_value_to_member_lookup.get(normalized_key_string)
                if not resolved_enum_member:
                    resolved_enum_member = enum_name_to_member_lookup.get(normalized_key_string)
                if not resolved_enum_member:
                    try:
                        resolved_enum_member = target_enum_class_ref(key_from_source_data.strip()) # type: ignore
                    except ValueError:
                         logger_cw.warning(f"'{weights_set_name}': 文字列キー '{key_from_source_data}' 変換不可。無視。")
            else:
                logger_cw.warning(f"'{weights_set_name}': 不正キー型 {type(key_from_source_data)} ('{key_from_source_data}')。無視。")

            if resolved_enum_member:
                unknown_member_ref = getattr(target_enum_class_ref, 'UNKNOWN', None)
                if unknown_member_ref and resolved_enum_member == unknown_member_ref:
                    if not (isinstance(key_from_source_data, str) and key_from_source_data.strip().lower() == 'unknown'):
                        logger_cw.info(f"'{weights_set_name}': キー '{key_from_source_data}' UNKNOWN解決。重み除外。")
                    continue

                try:
                    weight_as_float = float(value_from_source_data)
                    if weight_as_float >= 0.0:
                        processed_enum_keyed_weights[resolved_enum_member] = weight_as_float
                    else:
                        logger_cw.warning(f"'{weights_set_name}': キー '{str(key_from_source_data)}' 負重み ({weight_as_float})。0.0扱い。")
                        processed_enum_keyed_weights[resolved_enum_member] = 0.0
                except (ValueError, TypeError):
                    logger_cw.warning(f"'{weights_set_name}': キー '{str(key_from_source_data)}' 重み値 '{value_from_source_data}' float変換不可。無視。")
        
        if not self._validate_and_normalize_weights(processed_enum_keyed_weights, weights_set_name, final_auto_normalize_flag):
            logger_cw.warning(f"'{weights_set_name}' 検証/正規化問題あり。")
        return processed_enum_keyed_weights

    def _validate_and_normalize_weights(self, weights_map_to_validate: Dict[enum.Enum, float], map_description: str, perform_normalization: bool) -> bool:
        logger_val = self.logger.getChild(f"_validate_and_normalize_weights.{map_description.replace(' ', '_')}")
        is_data_struct_valid = True
        if not isinstance(weights_map_to_validate, dict):
            logger_val.error(f"'{map_description}' 検証エラー: 非辞書型。処理中止。")
            return False

        keys_to_be_removed: List[enum.Enum] = []
        for enum_key_item, weight_value_item in list(weights_map_to_validate.items()):
            if not isinstance(enum_key_item, enum.Enum):
                logger_val.warning(f"'{map_description}' 検証: 不正キー型 {type(enum_key_item)} ('{enum_key_item}')。削除。")
                keys_to_be_removed.append(enum_key_item); is_data_struct_valid = False; continue
            
            if not isinstance(weight_value_item, (int, float)) or weight_value_item < 0.0:
                logger_val.warning(f"'{map_description}' 検証: キー '{enum_key_item.name}' 重み値 ({weight_value_item}) 不正。0.0設定。")
                weights_map_to_validate[enum_key_item] = 0.0; is_data_struct_valid = False
            elif isinstance(weight_value_item, int):
                weights_map_to_validate[enum_key_item] = float(weight_value_item)

        for key_to_remove_item in keys_to_be_removed:
            if key_to_remove_item in weights_map_to_validate: weights_map_to_validate.pop(key_to_remove_item, None)

        if not weights_map_to_validate:
            logger_val.warning(f"'{map_description}' 検証: 有効重みなし。")
            return False

        keys_to_be_excluded_from_sum: Set[enum.Enum] = set()
        DFRSMetrics_enum_ref_for_norm: Optional[Type[DFRSMetricsEnumType]] = _get_global_type_func_pt10('DFRSMetricsV49', enum.EnumMeta) if _get_global_type_func_pt10 else None # type: ignore
        if map_description == "DFRS Weights (enhanced_dfrs_weights)" and DFRSMetrics_enum_ref_for_norm: # type: ignore
            final_eodf_key_ref = getattr(DFRSMetrics_enum_ref_for_norm, 'FINAL_EODF_V49', None)
            if final_eodf_key_ref and final_eodf_key_ref in weights_map_to_validate:
                keys_to_be_excluded_from_sum.add(final_eodf_key_ref) # type: ignore
        
        weights_for_summation_calc = {
            k_sum: v_sum for k_sum, v_sum in weights_map_to_validate.items() if k_sum not in keys_to_be_excluded_from_sum
        }
        
        if not weights_for_summation_calc:
            if keys_to_be_excluded_from_sum and all(k_all in keys_to_be_excluded_from_sum for k_all in weights_map_to_validate.keys()):
                logger_val.debug(f"'{map_description}' 検証: 全重み除外対象。合計1.0チェック/正規化スキップ。")
                return is_data_struct_valid
            else:
                logger_val.warning(f"'{map_description}' 検証: 合計計算対象の有効重みなし。")
                return False

        sum_of_target_weights = sum(weights_for_summation_calc.values())
        logger_val.debug(f"'{map_description}' 合計計算対象重み合計(正規化前): {sum_of_target_weights:.4f}")

        if not math.isclose(sum_of_target_weights, 1.0, abs_tol=0.01):
            logger_val.warning(f"'{map_description}' 重み合計 ({sum_of_target_weights:.4f}) が1.0から乖離。")
            if perform_normalization:
                if sum_of_target_weights > 1e-6:
                    logger_val.info(f"  -> '{map_description}': 指示により自動正規化実行...")
                    for key_to_normalize_item in weights_for_summation_calc.keys():
                        if key_to_normalize_item in weights_map_to_validate:
                            weights_map_to_validate[key_to_normalize_item] = round(weights_map_to_validate[key_to_normalize_item] / sum_of_target_weights, 4)
                    logger_val.info(f"  -> '{map_description}': 正規化後重み合計約1.0。")
                else:
                    logger_val.error(f"  -> '{map_description}': 重み合計ゼロ/極小。自動正規化不可。")
                    is_data_struct_valid = False
        return is_data_struct_valid

    def update_settings_based_on_mode(self, mode: str, auto_suggested: bool = True) -> None:
        self.logger.info(f"対話モード '{mode}' に基づきDialogueSettings更新 (自動推奨: {auto_suggested})。")
        pass

    def update_from_args(self, args: argparse.Namespace) -> None:
        self.logger.info("コマンドライン引数に基づき DialogueSettings 更新開始...")
        updated_attributes_count = 0
        args_as_dict = vars(args)

        simple_attribute_update_map_config = {
            'dialogue_mode': 'dialogue_mode', 'style_template': 'style_template',
            'custom_style_file': 'custom_style_file_path',
            'phase_tone_modulation': 'phase_tone_prompt_modulation_enabled',
            'subjective_focus': 'subjective_focus',
            'dfrs_all_loops': 'dfrs_evaluate_all_loops',
            'dfrs_enabled': 'dfrs_evaluation_enabled',
            'json_export': 'json_export_enabled',
            'schema_validation': 'json_schema_validation',
            'advanced_nlp': 'advanced_nlp_enabled',
            'ml_emotion': 'ml_emotion_enabled',
            'phase_tagging': 'phase_tagging_enabled',
            'persistent_cache': 'persistent_cache_enabled',
            'adaptation': 'adaptation_strategy_enabled',
            'log_transitions': 'log_phase_tone_transitions',
            'save_prompts': 'save_prompts',
            'save_evals': 'save_evaluations',
            'save_rejected': 'save_rejected_candidates',
            'nlp_model_name': 'nlp_model_name',
            'ml_emotion_model': 'ml_emotion_model',
            'adapt_strat': 'adaptation_strategy_type',
            'feedback_strat': 'feedback_strategy_type',
            'loops': 'feedback_loops',
            'min_loops': 'min_feedback_loops',
            'min_score': 'min_score_threshold',
            'length': 'target_length',
            'use_lightweight_ml': 'use_lightweight_ml_model'
        }

        for arg_name_in_args, setting_attr_target_name in simple_attribute_update_map_config.items():
            if arg_name_in_args in args_as_dict and args_as_dict[arg_name_in_args] is not None:
                new_value_from_cli = args_as_dict[arg_name_in_args]
                try:
                    current_value_of_setting = getattr(self, setting_attr_target_name, None)
                    expected_type_of_setting = type(current_value_of_setting) if current_value_of_setting is not None else type(new_value_from_cli)
                    
                    converted_value_to_set: Any
                    if isinstance(new_value_from_cli, expected_type_of_setting):
                        converted_value_to_set = new_value_from_cli
                    else:
                        if expected_type_of_setting == bool:
                            converted_value_to_set = str(new_value_from_cli).lower() in ['true', '1', 'yes', 'on', 't']
                        elif expected_type_of_setting == int:
                            converted_value_to_set = int(new_value_from_cli)
                        elif expected_type_of_setting == float:
                            converted_value_to_set = float(new_value_from_cli)
                        elif expected_type_of_setting == str:
                            converted_value_to_set = str(new_value_from_cli)
                        else:
                            self.logger.warning(f"引数 '{arg_name_in_args}' 更新スキップ: 型 ({expected_type_of_setting.__name__}) 変換不明。")
                            continue
                            
                    setattr(self, setting_attr_target_name, converted_value_to_set)
                    updated_attributes_count += 1
                    self.logger.debug(f"  設定 '{setting_attr_target_name}' CLI引数 '{arg_name_in_args}' により '{converted_value_to_set}' に更新。")

                    if setting_attr_target_name == 'dfrs_evaluate_all_loops' and converted_value_to_set and not self.dfrs_evaluation_enabled:
                        self.dfrs_evaluation_enabled = True; updated_attributes_count += 1
                        self.logger.debug("    -> dfrs_evaluate_all_loops=Trueのためdfrs_evaluation_enabled=Trueに自動更新。")
                    elif setting_attr_target_name == 'dfrs_evaluation_enabled' and not converted_value_to_set and self.dfrs_evaluate_all_loops:
                        self.dfrs_evaluate_all_loops = False; updated_attributes_count += 1
                        self.logger.debug("    -> dfrs_evaluation_enabled=Falseのためdfrs_evaluate_all_loops=Falseに自動更新。")
                    
                    if setting_attr_target_name == 'json_schema_validation' and converted_value_to_set and not JSONSCHEMA_AVAILABLE:
                        self.logger.warning("CLIでJSONスキーマ検証有効化もjsonschemaライブラリ利用不可。設定無視。")
                        setattr(self, setting_attr_target_name, False)
                except (ValueError, TypeError) as e_type_conversion:
                    self.logger.warning(f"引数 '{arg_name_in_args}' から設定 '{setting_attr_target_name}' への値変換エラー: {e_type_conversion}。スキップ。")
                except Exception as e_setattr_unexpected:
                    self.logger.warning(f"引数 '{arg_name_in_args}' による設定 '{setting_attr_target_name}' 更新中予期せぬエラー: {e_setattr_unexpected}")

        auto_normalize_weights_arg = args_as_dict.get('auto_normalize_weights')
        if auto_normalize_weights_arg is not None and isinstance(auto_normalize_weights_arg, bool):
            if self.auto_normalize_weights != auto_normalize_weights_arg:
                self.auto_normalize_weights = auto_normalize_weights_arg; updated_attributes_count += 1
                self.logger.debug(f"  設定 'auto_normalize_weights' CLIにより '{self.auto_normalize_weights}' に更新。")

        SubjectiveIntensityLevel_cls_for_args: Optional[Type[SubjectiveIntensityLevelType]] = _get_global_type_func_pt10('SubjectiveIntensityLevel', enum.EnumMeta) if _get_global_type_func_pt10 else None # type: ignore
        intensity_arg_string_val = args_as_dict.get('subjective_intensity')
        if SubjectiveIntensityLevel_cls_for_args and intensity_arg_string_val is not None: # type: ignore
            try:
                new_intensity_enum_val = SubjectiveIntensityLevel_cls_for_args(str(intensity_arg_string_val).lower()) # type: ignore
                if self.subjective_intensity != new_intensity_enum_val:
                    self.subjective_intensity = new_intensity_enum_val # type: ignore
                    updated_attributes_count += 1
                    self.logger.debug(f"  設定 'subjective_intensity' CLIにより '{getattr(self.subjective_intensity,'value','N/A')}' に更新。")
            except ValueError:
                self.logger.warning(f"CLI指定主観強度 '{intensity_arg_string_val}' 無効。現設定 ({getattr(self.subjective_intensity,'value','N/A')}) 維持。")

        def parse_and_apply_weights_from_cli_arg(
            arg_key_name_cli: str, target_settings_attr: str,
            target_enum_class_for_keys: Optional[Type[enum.Enum]], log_label_for_weights: str
        ) -> bool:
            if not (target_enum_class_for_keys and isinstance(target_enum_class_for_keys, enum.EnumMeta)):
                self.logger.error(f"{log_label_for_weights} 重み引数解析スキップ: 対象Enumクラス({target_enum_class_for_keys})不正。")
                return False

            weights_str_value_from_arg = args_as_dict.get(arg_key_name_cli)
            if not weights_str_value_from_arg or not isinstance(weights_str_value_from_arg, str): return False

            self.logger.info(f"CLI引数 '{arg_key_name_cli}' から '{log_label_for_weights}' カスタム重み解析: '{weights_str_value_from_arg}'")
            parsed_weights_from_cli: Dict[enum.Enum, float] = {}
            enum_val_to_member_lookup_cli = {str(e_member.value).lower(): e_member for e_member in target_enum_class_for_keys if hasattr(e_member, 'value')}
            enum_name_to_member_lookup_cli = {e_member.name.lower(): e_member for e_member in target_enum_class_for_keys if hasattr(e_member, 'name')}

            try:
                for item_pair_str in weights_str_value_from_arg.split(','):
                    if '=' not in item_pair_str:
                        self.logger.warning(f"{log_label_for_weights} 重み引数不正項目 '{item_pair_str}'。スキップ。")
                        continue
                    
                    key_part_str, value_part_str = item_pair_str.split('=', 1)
                    normalized_key_from_cli = key_part_str.strip().lower()
                    
                    enum_key_resolved: Optional[enum.Enum] = None
                    enum_key_resolved = enum_val_to_member_lookup_cli.get(normalized_key_from_cli)
                    if not enum_key_resolved: enum_key_resolved = enum_name_to_member_lookup_cli.get(normalized_key_from_cli)
                    if not enum_key_resolved:
                        try: enum_key_resolved = target_enum_class_for_keys(key_part_str.strip()) # type: ignore
                        except ValueError: pass

                    if enum_key_resolved:
                        try:
                            weight_val_from_cli = float(value_part_str.strip())
                            if weight_val_from_cli >= 0.0: parsed_weights_from_cli[enum_key_resolved] = weight_val_from_cli
                            else:
                                self.logger.warning(f"{log_label_for_weights} 重み引数: キー'{enum_key_resolved.name}' 負重み ({weight_val_from_cli})。0.0扱い。")
                                parsed_weights_from_cli[enum_key_resolved] = 0.0
                        except ValueError:
                            self.logger.warning(f"{log_label_for_weights} 重み引数: キー'{enum_key_resolved.name}' 重み値 '{value_part_str.strip()}' float変換不可。スキップ。")
                    else:
                        self.logger.warning(f"{log_label_for_weights} 重み引数: 不明キー '{key_part_str.strip()}' 無視。")
                
                if not parsed_weights_from_cli:
                    self.logger.warning(f"{log_label_for_weights} 重み引数 '{weights_str_value_from_arg}' から有効重み解析不可。")
                    return False

                current_weights_on_self: Dict[enum.Enum, float] = getattr(self, target_settings_attr, {})
                temp_merged_weights_for_cli = current_weights_on_self.copy()
                temp_merged_weights_for_cli.update(parsed_weights_from_cli)

                if self._validate_and_normalize_weights(temp_merged_weights_for_cli, f"{log_label_for_weights} (CLI適用後)", self.auto_normalize_weights):
                    setattr(self, target_settings_attr, temp_merged_weights_for_cli)
                    self.logger.info(f"{log_label_for_weights} CLIから更新・検証/正規化完了 (正規化: {self.auto_normalize_weights})。")
                    return True
                else:
                    self.logger.error(f"{log_label_for_weights} 重み引数適用後検証/正規化失敗。設定未更新。")
                    return False
            except Exception as e_parse_cli_weights:
                self.logger.error(f"{log_label_for_weights} 重み引数 '{weights_str_value_from_arg}' 解析中予期せぬエラー: {e_parse_cli_weights}", exc_info=True)
                return False

        if parse_and_apply_weights_from_cli_arg('dfrs_weights_override', 'enhanced_dfrs_weights', (_get_global_type_func_pt10('DFRSMetricsV49', enum.EnumMeta) if _get_global_type_func_pt10 else None), "DFRS(v4.9)重み"): # type: ignore
            updated_attributes_count += 1
        if parse_and_apply_weights_from_cli_arg('final_weights_override', 'final_selection_weights', (_get_global_type_func_pt10('FinalSelectionKeysV49', enum.EnumMeta) if _get_global_type_func_pt10 else None), "最終選択重み"): # type: ignore
            updated_attributes_count += 1
        if parse_and_apply_weights_from_cli_arg('initial_weights_override', '_validated_initial_candidate_weights', (_get_global_type_func_pt10('InitialSelectionKeysV49', enum.EnumMeta) if _get_global_type_func_pt10 else None), "初期候補選択重み"): # type: ignore
            updated_attributes_count += 1

        if updated_attributes_count > 0:
            self.logger.info(f"DialogueSettings: CLIから {updated_attributes_count} 箇所設定更新。")
        else:
            self.logger.debug("DialogueSettings: CLIによる設定更新なし。")

    def model_dump(self, mode: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        self.logger.debug(f"DialogueSettingsV49 カスタム model_dump 実行 (mode: {mode}, kwargs: {kwargs})")
        dump_data_dict: Dict[str, Any] = {}
        is_json_compatible_mode = mode == 'json'

        attributes_to_include_in_dump_list = [
            'dialogue_mode', 'style_template', 'custom_style_file_path',
            'subjective_focus', 'subjective_intensity',
            'dfrs_evaluation_enabled', 'dfrs_evaluate_all_loops', 'dfrs_for_initial_selection',
            'json_export_enabled', 'json_schema_validation',
            'advanced_nlp_enabled', 'nlp_model_name',
            'ml_emotion_enabled', 'ml_emotion_model', 'use_lightweight_ml_model',
            'phase_tagging_enabled', 'phase_tone_prompt_modulation_enabled',
            'persistent_cache_enabled', 'cache_dir',
            'adaptation_strategy_enabled', 'adaptation_strategy_type', 'log_phase_tone_transitions',
            'feedback_strategy_type',
            'save_prompts', 'save_evaluations', 'save_rejected_candidates',
            'feedback_loops', 'min_feedback_loops',
            'min_score_threshold', 'target_length', 'auto_normalize_weights'
        ]

        for attribute_name_to_dump in attributes_to_include_in_dump_list:
            if hasattr(self, attribute_name_to_dump):
                value_to_be_dumped = getattr(self, attribute_name_to_dump)
                if is_json_compatible_mode:
                    if isinstance(value_to_be_dumped, enum.Enum):
                        dump_data_dict[attribute_name_to_dump] = value_to_be_dumped.value
                    elif isinstance(value_to_be_dumped, datetime):
                        dump_data_dict[attribute_name_to_dump] = value_to_be_dumped.isoformat()
                    elif isinstance(value_to_be_dumped, pathlib.Path):
                        dump_data_dict[attribute_name_to_dump] = str(value_to_be_dumped)
                    elif not callable(value_to_be_dumped):
                        dump_data_dict[attribute_name_to_dump] = value_to_be_dumped
                else:
                    if not callable(value_to_be_dumped):
                        dump_data_dict[attribute_name_to_dump] = value_to_be_dumped
        
        def format_weights_dict_for_dump(
            weights_dict_internal: Dict[enum.Enum, float], is_json_mode: bool
        ) -> Dict[Union[str, enum.Enum], float]:
            if is_json_mode:
                return {(k_enum.value if hasattr(k_enum, 'value') else str(k_enum)): v_float
                        for k_enum, v_float in weights_dict_internal.items()}
            return weights_dict_internal

        if hasattr(self, 'enhanced_dfrs_weights') and isinstance(self.enhanced_dfrs_weights, dict):
            dump_data_dict['enhanced_dfrs_weights'] = format_weights_dict_for_dump(self.enhanced_dfrs_weights, is_json_compatible_mode)
        if hasattr(self, 'final_selection_weights') and isinstance(self.final_selection_weights, dict):
            dump_data_dict['final_selection_weights'] = format_weights_dict_for_dump(self.final_selection_weights, is_json_compatible_mode)
        if hasattr(self, '_validated_initial_candidate_weights') and isinstance(self._validated_initial_candidate_weights, dict):
            dump_data_dict['initial_candidate_weights'] = format_weights_dict_for_dump(self._validated_initial_candidate_weights, is_json_compatible_mode)

        self.logger.debug(f"カスタム model_dump 結果 (DialogueSettingsV49): {len(dump_data_dict)}属性ダンプ。")
        return dump_data_dict

# =============================================================================
# -- Dialogue Style Manager (v4.9α - 改善版 v2.2 - Standard Style Optimized & Stability Fix)
# =============================================================================
class DialogStyleManagerV49: # Implicitly implements StyleManagerProtocol
    logger = logging.getLogger(f"{__name__}.DialogStyleManagerV49.v2.2") # Version updated
    if TYPE_CHECKING:
        ConfigProtoRef = ConfigProtocol
        SceneInfoDictTypeRef = Dict[str, Any]
        PsychologicalPhaseV49Ref = PsychologicalPhaseV49
    else:
        ConfigProtoRef = 'ConfigProtocol'
        SceneInfoDictTypeRef = Dict[str, Any]
        PsychologicalPhaseV49Ref = 'PsychologicalPhaseV49'

    def __init__(self, config: ConfigProtoRef): # type: ignore
        self.config: DialogStyleManagerV49.ConfigProtoRef = config # type: ignore # Correctly typed and named
        self.custom_templates: Dict[str, Dict[str, Any]] = {}
        system_version_style_mgr = getattr(self.config, 'SYSTEM_VERSION', 'N/A_StyleMgr_v2.2')
        self.logger.info(f"DialogStyleManagerV49 (System Version: {system_version_style_mgr}) 初期化開始。")
        
        custom_style_file_path_val: Optional[str] = None
        # Access loaded_external_configs through self.config, which is now correctly initialized
        loaded_ext_configs = getattr(self.config, 'loaded_external_configs', None)
        file_settings_from_cfg = getattr(loaded_ext_configs, 'file_settings', None) if loaded_ext_configs else None
        if file_settings_from_cfg:
            custom_style_file_path_val = getattr(file_settings_from_cfg, 'custom_style_file_path', None)
        
        if custom_style_file_path_val and isinstance(custom_style_file_path_val, str):
            style_file_as_path = pathlib.Path(custom_style_file_path_val)
            if not style_file_as_path.is_absolute() and hasattr(self.config, 'config_dir') and \
               isinstance(self.config.config_dir, (str, pathlib.Path)) and self.config.config_dir: # type: ignore
                resolved_style_path = pathlib.Path(self.config.config_dir) / style_file_as_path # type: ignore
                self.logger.debug(f"カスタムスタイルファイル相対パス '{style_file_as_path}' を '{resolved_style_path}' に解決。")
                style_file_as_path = resolved_style_path.resolve()
            
            self.load_from_file(style_file_as_path)
        elif custom_style_file_path_val:
             self.logger.warning(f"設定カスタムスタイルファイルパス '{custom_style_file_path_val}' 文字列でない。カスタムスタイルロードなし。")
        else:
            self.logger.info("カスタムスタイルファイルパス未設定。カスタムスタイルロードなし。")

    def _load_base_templates(self) -> Dict[str, Dict[str, Any]]:
        base_style_presets_map: Dict[str, Any] = {}
        # Access loaded_external_configs through self.config
        if hasattr(self.config, 'loaded_external_configs') and self.config.loaded_external_configs: # type: ignore
            base_style_presets_map = getattr(self.config.loaded_external_configs, 'prompt_style_presets', {}) # type: ignore
        
        if not isinstance(base_style_presets_map, dict):
            self.logger.warning(f"設定 'prompt_style_presets' 非辞書型 (型: {type(base_style_presets_map)})。基本テンプレート空扱い。")
            base_style_presets_map = {}
        
        # --- ▼▼▼ "standard" スタイルの prompt_additions を「究極指針」ベースに刷新 (v2.1からの変更を維持) ▼▼▼ ---
        if 'standard' not in base_style_presets_map:
            self.logger.debug("基本スタイル 'standard' が設定ファイルにないため、最適化されたデフォルト定義を使用します。")
        
        base_style_presets_map['standard'] = {
            "description": "究極指針準拠 - 現代的かつ高品質な標準文体 (最適化版 v2.2)", # Description updated
            "prompt_additions": [
                # 1. 基本文体と読者層への意識
                "文体は、現代の読者が直感的に理解しやすく、心地よく読み進められる、明確かつ洗練された現代日本語を基本とします。",
                "特に、軽快なテンポと深い感情描写が求められる漫画原作の小説化においては、視覚的なイメージ喚起と心理的な没入感のバランスを重視してください。",
                
                # 2. 表現の具体性と「見せる描写」
                "登場人物の感情や状況は、具体的な行動、表情、仕草、五感を通じた情景描写（Show）を通じて、読者が追体験できるように表現することを優先してください。",
                "抽象的な説明よりも、具体的で感覚的なディテールを重視し、読者の想像力を刺激してください。",
                
                # 3. リズムとテンポ、会話の自然さ
                "文の長短、句読点の打ち方、改行、段落構成を巧みに操り、シーンの感情の起伏や展開に合わせて、よどみのない自然で心地よいリズムとテンポを生み出してください。",
                "セリフは、キャラクターの性格や感情、状況がにじみ出るような、自然で生き生きとしたものにしてください。地の文とのバランスを適切に保ち、会話が物語を効果的に推進するように構成してください。",
                
                # 4. 語彙と比喩
                "語彙は、シーンの雰囲気とキャラクターの感情に完全に合致した、現代的で分かりやすく、かつ表現力豊かな言葉を選んでください。陳腐な言い回しや過度に難解な言葉は避け、読者がスムーズに意味を理解できることを優先してください。",
                "比喩表現は、読者の理解を助け、感情的な深みを与える場合に限り、新鮮で的確、かつ作品のトーンから逸脱しない自然な範囲で使用してください。",
                
                # 5. 抑制と効果的な強調
                "過度な装飾や感傷的な言葉の濫用は避け、抑制の効いた筆致を基本としながらも、シーンの重要なポイントや感情の高まりにおいては、読者の心に強く残るような効果的な強調表現（例：短い文の連続、印象的な感覚描写、象徴的なアイテムの提示など）を意識してください。",
                
                # 6. オノマトペと「間」の活用（限定的かつ効果的に）
                "オノマトペや擬態語は、臨場感を高めたり、キャラクターの心理を暗示したりするために真に効果的であると判断される場合に限り、小説の文体として自然に溶け込む形で、かつ抑制的に使用してください。過度な使用や漫画的な描き文字のような表現は避けてください。",
                "三点リーダーやダッシュ、効果的な改行や空白行は、セリフや思考の「間」、感情の余韻、緊張感の醸成、場面転換の合図として、漫画のコマ割りのように戦略的に活用し、読書体験に自然なリズムと奥行きを与えてください。"
            ]
        }
        self.logger.info("DialogStyleManager: 'standard'スタイルのprompt_additionsを「究極指針」ベースに更新/設定しました。")
        # --- ▲▲▲ "standard" スタイルの prompt_additions を刷新 ▲▲▲ ---
        return base_style_presets_map

    def register_custom_template(self, name: str, description: str, prompt_additions: Union[List[str], str]) -> None:
        # (このメソッドはv2から変更なし)
        if not (isinstance(name, str) and name.strip()):
            raise ValueError("カスタムスタイルの名前（name）は、空でない文字列である必要があります。")
        
        normalized_style_name = name.lower().strip()
        
        base_templates_loaded = self._load_base_templates()
        if normalized_style_name in base_templates_loaded:
            self.logger.warning(f"登録しようとしているカスタムスタイル名 '{normalized_style_name}' は、基本テンプレート名と重複しています。"
                                "get_template呼び出し時には、このカスタムテンプレートが優先されます。")
        
        if not (isinstance(description, str) and description.strip()):
            raise ValueError(f"スタイル '{normalized_style_name}' の説明（description）は、空でない文字列である必要があります。")
        
        formatted_prompt_additions: List[str] = []
        if isinstance(prompt_additions, str):
            formatted_prompt_additions = [line.strip() for line in prompt_additions.splitlines() if line.strip()]
        elif isinstance(prompt_additions, list):
            formatted_prompt_additions = [str(item).strip() for item in prompt_additions if isinstance(item, str) and str(item).strip()]
        else:
            raise ValueError(f"スタイル '{normalized_style_name}' のプロンプト追加指示（prompt_additions）は、文字列または文字列のリストである必要があります。")
            
        if not formatted_prompt_additions:
             self.logger.warning(f"スタイル '{normalized_style_name}' に有効なプロンプト追加指示がありません。空の指示リストとして登録されます。")

        self.custom_templates[normalized_style_name] = {
            "description": description.strip(),
            "prompt_additions": formatted_prompt_additions
        }
        self.logger.info(f"カスタムスタイル '{normalized_style_name}' を登録または更新しました。指示行数: {len(formatted_prompt_additions)}")

    def load_from_file(self, filepath_to_load: Union[str, pathlib.Path]) -> None:
        # (このメソッドはv2から変更なし、load_json_utility_func を使用)
        style_file_path_obj = pathlib.Path(filepath_to_load)
        self.logger.info(f"カスタムスタイル定義ファイルを '{style_file_path_obj}' から読み込みます...")
        
        if not style_file_path_obj.is_file():
            self.logger.error(f"指定されたカスタムスタイルファイルが見つかりません: {style_file_path_obj}")
            return

        exception_mgr = getattr(self.config, '_exception_manager', None)
        
        if not (exception_mgr and load_json_utility_func and callable(load_json_utility_func)):
            self.logger.error("ExceptionManager または load_json ユーティリティがAppConfigに設定されていないか利用できません。カスタムスタイルのファイルからのロード処理を中止します。")
            return

        load_successful, loaded_data_from_file, file_load_error_msg = exception_mgr.safe_file_operation( # type: ignore
            operation_description=f"カスタムスタイルファイルロード ({style_file_path_obj.name})",
            file_operation_callable=load_json_utility_func,
            args_tuple=(style_file_path_obj,)
        )

        if not load_successful or not isinstance(loaded_data_from_file, dict):
            self.logger.error(f"カスタムスタイルファイル '{style_file_path_obj}' のロードまたはJSONパースに失敗しました。エラー: {file_load_error_msg or '不明エラー'}")
            return
        
        num_loaded_successfully = 0; num_skipped_due_to_error = 0
        for style_name_from_file, template_data_from_file in loaded_data_from_file.items():
            if isinstance(template_data_from_file, dict):
                description_from_file = template_data_from_file.get("description")
                prompt_additions_from_file = template_data_from_file.get("prompt_additions")
                
                if description_from_file is None or prompt_additions_from_file is None:
                    self.logger.warning(f"ファイル内のスタイル '{style_name_from_file}' の定義に 'description'/'prompt_additions' 欠落。スキップ。")
                    num_skipped_due_to_error += 1; continue
                try:
                    self.register_custom_template(style_name_from_file, description_from_file, prompt_additions_from_file)
                    num_loaded_successfully += 1
                except ValueError as e_register_val_err:
                    self.logger.warning(f"ファイルロードスタイル '{style_name_from_file}' 登録中検証エラー: {e_register_val_err}。スキップ。")
                    num_skipped_due_to_error += 1
                except Exception as e_register_generic_err:
                    self.logger.error(f"ファイルロードスタイル '{style_name_from_file}' 登録中予期せぬエラー: {e_register_generic_err}。スキップ。", exc_info=True)
                    num_skipped_due_to_error += 1
            else:
                self.logger.warning(f"ファイル内スタイル '{style_name_from_file}' データ非辞書型 (型: {type(template_data_from_file)})。スキップ。")
                num_skipped_due_to_error += 1
                
        self.logger.info(f"カスタムスタイルファイル '{style_file_path_obj}' から {num_loaded_successfully}件ロード・登録 (スキップ: {num_skipped_due_to_error}件)。")

    def get_template(self, style_name_requested: str) -> Dict[str, Any]:
        # (このメソッドはv2から変更なし)
        normalized_requested_style_name = style_name_requested.lower().strip() if isinstance(style_name_requested, str) else "standard"
        
        if normalized_requested_style_name in self.custom_templates:
            self.logger.debug(f"カスタムスタイル '{normalized_requested_style_name}' を取得しました。")
            return copy.deepcopy(self.custom_templates[normalized_requested_style_name])
            
        base_templates_map = self._load_base_templates() # "standard" はここで必ず定義される
        if normalized_requested_style_name in base_templates_map:
            base_template_data = base_templates_map[normalized_requested_style_name]
            if isinstance(base_template_data, dict):
                description_text_base = base_template_data.get("description", f"{normalized_requested_style_name} (基本テンプレート - 説明なし)")
                prompt_additions_raw_base = base_template_data.get("prompt_additions")
                prompt_additions_list_base: List[str] = []
                if isinstance(prompt_additions_raw_base, str):
                    prompt_additions_list_base = [line.strip() for line in prompt_additions_raw_base.splitlines() if line.strip()]
                elif isinstance(prompt_additions_raw_base, list):
                    prompt_additions_list_base = [str(item).strip() for item in prompt_additions_raw_base if isinstance(item, str) and str(item).strip()]
                
                self.logger.debug(f"基本スタイル '{normalized_requested_style_name}' を取得しました。")
                return {"description": description_text_base, "prompt_additions": copy.deepcopy(prompt_additions_list_base)}
            else: # base_templates_map[normalized_requested_style_name] が dict でない稀なケース
                self.logger.warning(f"基本スタイル '{normalized_requested_style_name}' のデータ形式が不正です。'standard' にフォールバックします。")
                # この場合、normalized_requested_style_name が "standard" であっても、再帰呼び出しで正しい "standard" が返る
                return self.get_template("standard")

        # normalized_requested_style_name が "standard" でもなく、カスタムにも基本にもない場合 (通常は発生しないはず)
        self.logger.error(f"要求されたスタイル '{style_name_requested}' が見つからず、'standard' の取得もできませんでした。最終フォールバックを使用します。")
        return copy.deepcopy({
            "description": "Standard (最終フォールバック定義 - 重大エラー)",
            "prompt_additions": ["標準的な現代日本小説の文体で、明確かつ自然に記述してください。"]
        })


    def list_available_styles(self) -> List[Dict[str, str]]:
        # (このメソッドはv2から変更なし)
        all_styles_map: Dict[str, Dict[str, str]] = {}
        
        base_templates_loaded = self._load_base_templates()
        for base_name, base_data_dict in base_templates_loaded.items():
            normalized_base_name = base_name.lower().strip()
            description_for_base = base_data_dict.get("description", f"{base_name} (基本テンプレート - 説明なし)") if isinstance(base_data_dict, dict) else f"{base_name} (基本テンプレート - データ不正)"
            all_styles_map[normalized_base_name] = {"name": base_name, "description": description_for_base}
            
        for custom_name_norm, custom_info_dict in self.custom_templates.items():
            all_styles_map[custom_name_norm] = {"name": custom_name_norm, "description": custom_info_dict["description"]}
            
        return sorted(list(all_styles_map.values()), key=lambda style_dict_item: style_dict_item['name'])

    def get_style_prompt_addition_text(self, style_name_to_apply: str) -> str:
        # (このメソッドはv2から変更なし)
        template_data_for_style = self.get_template(style_name_to_apply)
        prompt_additions_list_val = template_data_for_style.get("prompt_additions", [])
        
        if isinstance(prompt_additions_list_val, list) and prompt_additions_list_val:
            return "\n".join(f"- {item_text}" for item_text in prompt_additions_list_val if item_text)
        return ""

    @classmethod
    def suggest_style_for_scene(
        cls,
        scene_info_data_dict: SceneInfoDictTypeRef, # type: ignore
        analysis_results_dict_param: Optional[Dict[str, Any]] = None,
        config_obj_param: Optional[ConfigProtoRef] = None # type: ignore
    ) -> str:
        # (このメソッドはv2.1から変更なし、キーワードマップ充実)
        logger_suggest_style = logging.getLogger(f"{cls.__module__}.{cls.__qualname__}.suggest_style_for_scene.v2.2")
        
        style_suggestion_scores: Dict[str, float] = defaultdict(float)
        style_suggestion_scores["standard"] = 0.5

        style_suggestion_keyword_config_map: Dict[str, Any] = {}
        if config_obj_param and hasattr(config_obj_param, 'loaded_external_configs') and config_obj_param.loaded_external_configs: # type: ignore
            raw_keyword_map_from_config = getattr(config_obj_param.loaded_external_configs, 'style_suggestion_keyword_map', None) # type: ignore
            if isinstance(raw_keyword_map_from_config, dict):
                style_suggestion_keyword_config_map = raw_keyword_map_from_config
        
        if not style_suggestion_keyword_config_map:
            style_suggestion_keyword_config_map = {
                "introspective": (["心理", "思考", "内面", "葛藤", "感じる", "思う", "記憶", "過去", "内省", "意識"], 1.0),
                "poetic": (["詩的", "風景", "美しい", "情景", "色彩", "光", "影", "空気感", "静寂", "抒情的"], 1.0),
                "cinematic": (["視覚的", "映像的", "カメラ", "カット", "ズーム", "アングル", "構図", "光と影", "シーン"], 1.0),
                "action_oriented": (["行動", "アクション", "追跡", "戦闘", "爆発", "速度", "衝撃", "破壊", "ダイナミック"], 0.8),
                "formal": (["厳粛", "公式", "儀式", "丁寧", "敬語", "格調高い", "荘重"], 0.7),
                "minimalist": (["簡潔", "最小限", "余白", "静寂", "抑制", "客観的", "省筆"], 0.9)
            }
            logger_suggest_style.debug("スタイル提案キーワードマップ設定ファイルになし。フォールバック定義使用。")

        scene_text_content_parts = []
        # "notes" もキーワード検索の対象に含める
        for key_in_scene_info in ["name", "purpose", "atmosphere", "constraints", "environmentElements", "spatialCharacteristics", "notes"]:
            value_from_scene = scene_info_data_dict.get(key_in_scene_info)
            if isinstance(value_from_scene, str) and value_from_scene.strip():
                scene_text_content_parts.append(value_from_scene.lower())
            elif isinstance(value_from_scene, list):
                scene_text_content_parts.extend(str(item).lower() for item in value_from_scene if isinstance(item, str) and item.strip())
        combined_scene_text_lower = " ".join(scene_text_content_parts)
        
        logger_suggest_style.debug(f"スタイル提案用シーン結合テキスト(小文字化): '{combined_scene_text_lower[:250]}...' (全長: {len(combined_scene_text_lower)})")

        for style_name_key, style_config_data in style_suggestion_keyword_config_map.items():
            keywords_for_style: List[str] = []
            weight_for_style: float = 1.0

            if isinstance(style_config_data, tuple) and len(style_config_data) == 2 and isinstance(style_config_data[0], list) and isinstance(style_config_data[1], (int, float)):
                keywords_for_style, weight_for_style = style_config_data[0], float(style_config_data[1])
            elif isinstance(style_config_data, list):
                keywords_for_style = [str(k) for k in style_config_data if isinstance(k, str)]
            
            if keywords_for_style and weight_for_style > 0:
                num_keyword_hits = sum(1 for kw_str in keywords_for_style if kw_str.lower() in combined_scene_text_lower)
                if num_keyword_hits > 0:
                    style_suggestion_scores[style_name_key] += num_keyword_hits * weight_for_style
                    logger_suggest_style.debug(f"  スタイル '{style_name_key}': キーワードヒット数 {num_keyword_hits}, 加算スコア {num_keyword_hits * weight_for_style:.2f}")
            else:
                logger_suggest_style.warning(f"スタイル提案キーワードマップ スタイル '{style_name_key}' 定義不正。")

        PsychologicalPhaseV49_cls_for_suggest: Optional[Type[PsychologicalPhaseV49Ref]] = _get_global_type_func_pt10('PsychologicalPhaseV49', enum.EnumMeta) if _get_global_type_func_pt10 else None # type: ignore
        if isinstance(analysis_results_dict_param, dict) and PsychologicalPhaseV49_cls_for_suggest: # type: ignore
            analysis_summary_partial_dict = analysis_results_dict_param.get("analysis_summary_partial")
            if isinstance(analysis_summary_partial_dict, dict):
                dominant_phase_str_from_summary = analysis_summary_partial_dict.get("dominant_phase")
                if isinstance(dominant_phase_str_from_summary, str):
                    try:
                        dominant_phase_enum_val = PsychologicalPhaseV49_cls_for_suggest(dominant_phase_str_from_summary) # type: ignore
                        logger_suggest_style.debug(f"  スタイル提案用優勢位相検出: '{dominant_phase_enum_val.value}'") # type: ignore
                        
                        if dominant_phase_enum_val == PsychologicalPhaseV49_cls_for_suggest.INTERNAL_PROCESSING: # type: ignore
                            style_suggestion_scores["introspective"] = style_suggestion_scores.get("introspective", 0.0) + 0.5
                        elif dominant_phase_enum_val == PsychologicalPhaseV49_cls_for_suggest.ACTION_EVENT: # type: ignore
                            style_suggestion_scores["action_oriented"] = style_suggestion_scores.get("action_oriented", 0.0) + 0.5
                    except ValueError:
                        logger_suggest_style.warning(f"  スタイル提案: analysis_results 優勢位相文字列 '{dominant_phase_str_from_summary}' Enum変換不可。")
                else:
                     logger_suggest_style.debug("  スタイル提案: analysis_results dominant_phase 文字列でない。位相調整スキップ。")
            else:
                logger_suggest_style.debug("  スタイル提案: analysis_results に 'analysis_summary_partial'/'dominant_phase' 未発見。位相調整スキップ。")
        elif isinstance(analysis_results_dict_param, dict):
            logger_suggest_style.warning("  スタイル提案: PsychologicalPhaseV49 Enumクラス未ロード。位相スコア調整不可。")

        suggestion_score_threshold = 0.7
        if config_obj_param and hasattr(config_obj_param, 'loaded_external_configs') and config_obj_param.loaded_external_configs: # type: ignore
             suggestion_score_threshold = float(getattr(config_obj_param.loaded_external_configs, 'style_suggestion_threshold', 0.7)) # type: ignore

        eligible_styles_above_threshold = {
            style_key_name: score_val for style_key_name, score_val in style_suggestion_scores.items()
            if style_key_name != "standard" and score_val >= suggestion_score_threshold
        }
        
        final_recommended_style = "standard"
        if eligible_styles_above_threshold:
            final_recommended_style = sorted(eligible_styles_above_threshold.items(), key=lambda item_pair: item_pair[1], reverse=True)[0][0]
        
        self.logger.info(f"シーン '{scene_info_data_dict.get('name','N/A')}' 推奨スタイル: '{final_recommended_style}' "
                                  f"(スコア: {style_suggestion_scores.get(final_recommended_style, 0.0):.2f}). "
                                  f"全スタイルスコア: { {k:round(v,2) for k,v in style_suggestion_scores.items()} }")
        return final_recommended_style

# =============================================================================
# -- Part 10 終了点 (DialogueSettingsV49, DialogStyleManagerV49 クラス定義終了)
# =============================================================================
# =============================================================================
# -- Part 11: Exception Manager & Structured Error (v4.9α - 改善版)
# =============================================================================
# v4.9α: 構造化エラークラスと例外管理クラス。エラーコード体系の整理、
#        原因推定の強化、Configからの設定読み込みに対応。
# 改善版: エラーコード体系化、原因推定・リトライ判定強化、Config参照改善。

# --- Structured Error Class (v4.9α - 改善版) ---
class StructuredErrorV49(Exception):
    """構造化エラー情報を保持するカスタム例外 (v4.9α - message属性追加版)"""
    if TYPE_CHECKING: ContextDataType = Optional[Dict[str, Any]]
    else: ContextDataType = Optional[Dict[str, Any]]

    def __init__(self, message: Union[str, Exception], code: str = "UNKNOWN.GENERIC", source: str = "Unknown",
                 details: Optional[Dict[str, Any]] = None, context_data: ContextDataType = None,
                 root_cause: Optional[str] = None, is_retryable: Optional[bool] = None,
                 original_exception: Optional[Exception] = None, category: Optional[str] = None):
        
        eff_msg: str
        if isinstance(message, Exception):
            # 例外インスタンスが渡された場合、その最初の引数をメッセージとするか、例外クラス名を使用
            eff_msg = str(message.args[0]) if message.args else type(message).__name__
        else:
            eff_msg = str(message) # 文字列が渡された場合はそのまま使用

        super().__init__(eff_msg) # Exceptionの初期化子には効果的なメッセージを渡す
        
        # --- ▼▼▼ 改善提案に基づき self.message 属性を追加 ▼▼▼ ---
        self.message: str = eff_msg
        # --- ▲▲▲ ここまで追加 ▲▲▲ ---
        
        self.code: str = code
        self.source: str = source
        self.details: Dict[str, Any] = details if details is not None else {}
        self.context_data: Dict[str, Any] = context_data.copy() if context_data is not None else {}
        self.root_cause: Optional[str] = root_cause
        self.is_retryable: Optional[bool] = is_retryable
        self.original_exception: Optional[Exception] = original_exception if original_exception is not None else (message if isinstance(message, Exception) else None)
        self.category: str = category if category is not None else self._extract_category_from_code(code)

        if self.original_exception:
            self.details["original_exception_type"] = type(self.original_exception).__name__
            try:
                orig_msg_content = str(self.original_exception.args[0]) if self.original_exception.args else "(No message content)"
            except Exception:
                orig_msg_content = "(Failed to get original message content)"
            
            max_len = 500 # 元のメッセージの最大長
            self.details["original_exception_message"] = orig_msg_content[:max_len] + ('...' if len(orig_msg_content) > max_len else '')

    @staticmethod
    def _extract_category_from_code(code: str) -> str:
        # (このメソッドは変更なし)
        return code.split('.')[0].upper() if isinstance(code, str) and '.' in code else "UNKNOWN"

    def to_dict(self, include_context: bool = True, context_max_len: int = 200) -> Dict[str, Any]:
        # (このメソッドは変更なし、ただし self.message が存在することを前提にできる)
        # ... (既存の to_dict の実装) ...
        ctx = {}
        if include_context and isinstance(self.context_data, dict):
            sensitive = {'key', 'token', 'password', 'secret'}
            for k, v in self.context_data.items():
                k_str = str(k); v_repr = ""; is_long = False
                try:
                    if isinstance(k_str, str) and any(s in k_str.lower() for s in sensitive): v_repr = "*** MASKED ***"
                    elif isinstance(v, (str,int,float,bool,list,tuple,dict,type(None),pathlib.Path,datetime,enum.Enum)): # pathlib, datetime, enum を追加
                         v_repr = v.value if isinstance(v, enum.Enum) else str(v)
                         is_long = isinstance(v, (str, list, dict)) and len(v_repr) > context_max_len
                    else: v_repr = f"<{type(v).__name__}>"
                    key_name = f"{k_str}_snippet" if is_long else k_str
                    ctx[key_name] = v_repr[:context_max_len] + ('...' if is_long else '')
                except Exception: ctx[k_str] = "(SerializationError)"
        return {"error_category": self.category, "error_code": self.code, "error_source": self.source,
                "error_message": self.message, # self.message を直接使用
                "error_details": self.details, "context_data_summary": ctx if include_context else "(omitted)",
                "root_cause": self.root_cause, "is_retryable": self.is_retryable,
                "original_exception_type": self.details.get("original_exception_type")}


    def __str__(self) -> str:
        # (このメソッドは変更なし、super().__str__() は self.args[0]、つまり eff_msg / self.message を返す)
        base = self.message # super().__str__() の代わりに self.message を直接使用しても良い
        snippet = base.split('\n')[0][:250] + ('...' if len(base) > 250 or '\n' in base else '')
        msg = f"[{self.source}:{self.code}] {snippet}"
        if self.root_cause: msg += f" (推定原因: {self.root_cause})"
        if self.is_retryable is not None: msg += f" [Retry:{self.is_retryable}]"
        if orig_type := self.details.get("original_exception_type"): msg += f" (Orig:{orig_type})"
        return msg

# --- Exception Manager (v4.9α - 改善版) ---
class ExceptionManagerV49: # Implicitly implements ExceptionManagerProtocol
    """例外処理とリトライ戦略を管理 (v4.9α 改善版)"""
    if TYPE_CHECKING: ConfigProto = ConfigProtocol; StructuredErrorType = StructuredErrorV49 # type: ignore
    else: ConfigProto = 'ConfigProtocol'; StructuredErrorType = 'StructuredErrorV49'

    def __init__(self, config: ConfigProto, logger_instance: Optional[logging.Logger] = None): # type: ignore
        """初期化"""
        self.config = config; self.logger = logger_instance or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.error_counts: Counter = Counter()
        ext_cfg = self.config.loaded_external_configs; err_cfg_model = getattr(ext_cfg, 'error_config', None) if ext_cfg else None # type: ignore
        # --- エラーコード定義 (デフォルト + Configからの上書き) ---
        default_sev: Dict[str, Literal["RECOVERABLE", "FATAL", "WARNING_ONLY"]] = { # type: ignore
             "UNKNOWN.GENERIC": "FATAL", "INTERNAL.ERROR": "FATAL", "API.TIMEOUT": "RECOVERABLE", "API.RATE_LIMIT": "RECOVERABLE",
             "API.QUOTA_EXCEEDED": "RECOVERABLE", "API.INTERNAL_ERROR": "RECOVERABLE", "API.SERVICE_UNAVAILABLE": "RECOVERABLE",
             "API.CONNECTION_ERROR": "RECOVERABLE", "API.ABORTED": "RECOVERABLE", "API.SAFETY_BLOCK": "FATAL", "API.PERMISSION_DENIED": "FATAL",
             "API.UNAUTHENTICATED": "FATAL", "API.NOT_FOUND": "FATAL", "API.BAD_REQUEST": "FATAL", "API.INVALID_ARGUMENT": "FATAL",
             "API.INVALID_API_KEY": "FATAL", "API.MODEL_NOT_FOUND": "FATAL", "API.FAILED_PRECONDITION": "FATAL", "FILE.NOT_FOUND": "FATAL",
             "FILE.PERMISSION": "FATAL", "FILE.IO_ERROR": "RECOVERABLE", "FILE.IS_A_DIRECTORY": "FATAL", "FILE.NOT_A_DIRECTORY": "FATAL",
             "DB.LOCK": "RECOVERABLE", "DB.OPERATIONAL": "RECOVERABLE", "DB.INTEGRITY": "FATAL", "DB.GENERIC": "RECOVERABLE",
             "NETWORK.CONNECTION_ERROR": "RECOVERABLE", "NETWORK.TIMEOUT": "RECOVERABLE", "CONFIG.MISSING": "FATAL", "CONFIG.PARSE_ERROR": "FATAL",
             "CONFIG.SCHEMA_MISMATCH": "FATAL", "CONFIG.LOAD_ERROR": "FATAL", "MODEL.VALIDATION_ERROR": "FATAL", "INPUT_DATA.INVALID": "FATAL",
             "NLP_ML.MODEL_LOAD_ERROR": "FATAL", "NLP_ML.PROCESSING_ERROR": "RECOVERABLE", "GENERATION.INITIAL_CRITICAL": "FATAL",
             "GENERATION.IMPROVE_UNRECOVERABLE": "FATAL", "GENERATION.INVALID_PARAM": "FATAL", "GENERATION.IMPROVE_MISSING_DATA": "FATAL",
             "GENERATION.FINAL_SELECTION_FAILED": "FATAL", "GENERATION.UNEXPECTED_LOOP_EXIT": "FATAL", "GENERATION.INITIAL_CANDIDATE_EMPTY": "FATAL",
             "GENERATION.IMPROVE_FAILED": "RECOVERABLE", "GENERATION.CONSECUTIVE_ERRORS": "FATAL", "SYSTEM.MEMORY_ERROR": "FATAL",
             "SYSTEM.IMPORT_ERROR": "FATAL", "SYSTEM.MISSING_COMPONENT": "FATAL", "SYSTEM.OS_ERROR": "FATAL", "INTERNAL.VALUE_ERROR": "FATAL",
             "INTERNAL.TYPE_ERROR": "FATAL", "INTERNAL.ATTRIBUTE_ERROR": "FATAL", "INTERNAL.KEY_ERROR": "FATAL", "INTERNAL.INDEX_ERROR": "FATAL",
             "INTERNAL.NOT_IMPLEMENTED": "FATAL", "INTERNAL.UNEXPECTED_LOOP_EXIT": "FATAL", "INTERNAL.MISSING_MODEL": "FATAL",
             "INTERNAL.INVALID_KEY": "FATAL", "EVAL.EMPTY_TEXT": "FATAL", "EVAL.PROMPT_ERROR": "FATAL", "EVAL.API_FAILED": "RECOVERABLE",
             "EVAL.SCORE_EXTRACTION_FAILED": "RECOVERABLE", "EVAL.IMPROVE_SCORE_EXTRACTION_FAILED": "RECOVERABLE", "DFRS.EVALUATION_ERROR": "RECOVERABLE",
             "DFRS.WEIGHT_SUM_MISMATCH": "FATAL", "DFRS.FINAL_EVAL_FAIL": "WARNING_ONLY", "FILE.STATS_WRITE_ERROR": "WARNING_ONLY",
             "SAVE.FINAL_FAIL": "WARNING_ONLY", "SAVE.RESUME_FAIL": "WARNING_ONLY", "UNCLASSIFIED_ERROR": "FATAL", "MAIN.UNCAUGHT": "FATAL",
             "STATE.MISSING": "FATAL", "ApiClientInit": "FATAL", "StateInitialization": "FATAL", # 初期化系
        }
        default_strat: Dict[str, Dict[str, float]] = { # type: ignore
             "API.TIMEOUT": {"retry_delay": 5.0, "max_retries": 3, "backoff_factor": 1.5}, "API.RATE_LIMIT": {"retry_delay": 12.0, "max_retries": 5, "backoff_factor": 1.8},
             "API.QUOTA_EXCEEDED": {"retry_delay": 60.0, "max_retries": 2, "backoff_factor": 2.0}, "API.INTERNAL_ERROR": {"retry_delay": 3.0, "max_retries": 4, "backoff_factor": 1.5},
             "API.SERVICE_UNAVAILABLE": {"retry_delay": 5.0, "max_retries": 4, "backoff_factor": 1.6}, "API.CONNECTION_ERROR": {"retry_delay": 2.0, "max_retries": 5, "backoff_factor": 1.3},
             "API.ABORTED": {"retry_delay": 1.0, "max_retries": 3, "backoff_factor": 1.2}, "DB.LOCK": {"retry_delay": 0.8, "max_retries": 6, "backoff_factor": 1.2},
             "DB.OPERATIONAL": {"retry_delay": 1.5, "max_retries": 3, "backoff_factor": 1.4}, "NETWORK.CONNECTION_ERROR": {"retry_delay": 2.5, "max_retries": 4, "backoff_factor": 1.4},
             "NETWORK.TIMEOUT": {"retry_delay": 3.0, "max_retries": 3, "backoff_factor": 1.5}, "EVAL.API_FAILED": {"retry_delay": 3.0, "max_retries": 2, "backoff_factor": 1.5},
             "EVAL.SCORE_EXTRACTION_FAILED": {"retry_delay": 1.0, "max_retries": 1, "backoff_factor": 1.0}, "EVAL.IMPROVE_SCORE_EXTRACTION_FAILED": {"retry_delay": 1.0, "max_retries": 1, "backoff_factor": 1.0},
             "GENERATION.IMPROVE_FAILED": {"retry_delay": 2.0, "max_retries": 2, "backoff_factor": 1.3}, "DFRS.EVALUATION_ERROR": {"retry_delay": 1.5, "max_retries": 2, "backoff_factor": 1.2},
             "NLP_ML.PROCESSING_ERROR": {"retry_delay": 1.0, "max_retries": 1, "backoff_factor": 1.0},
        }
        cfg_sev = getattr(err_cfg_model, 'severity', {}) if err_cfg_model and isinstance(getattr(err_cfg_model,'severity',None),dict) else {}
        cfg_strat = {k:v.model_dump() for k,v in getattr(err_cfg_model, 'recovery_strategies', {}).items()} if err_cfg_model and isinstance(getattr(err_cfg_model,'recovery_strategies',None),dict) else {} # Pydanticモデルを辞書に変換
        self.ERROR_SEVERITY = {**default_sev, **cfg_sev} # type: ignore
        self.ERROR_RECOVERY_STRATEGIES = {**default_strat, **cfg_strat} # type: ignore
        self.max_retries = self.config.MAX_RETRIES; self.base_retry_delay = self.config.BASE_RETRY_DELAY
        self.logger.info(f"ExceptionManagerV49 ({self.config.SYSTEM_VERSION}) 初期化完了。")

    def _extract_context(self, args: Optional[Tuple], kwargs: Optional[Dict]) -> Dict[str, Any]:
        """操作引数からコンテキスト情報を抽出"""
        ctx={}; sensitive={'key','token','password','secret'}
        def fmt_val(v: Any, k: Optional[str]=None, max_len: int=150) -> str:
             if isinstance(k,str) and any(s in k.lower() for s in sensitive): return "*** MASKED ***"
             try:
                 ok_types=(str,int,float,bool,list,tuple,dict,type(None),pathlib.Path,datetime,enum.Enum)
                 if isinstance(v, ok_types): rep = v.value if isinstance(v,enum.Enum) else str(v); lng = isinstance(v,(str,list,dict)) and len(rep)>max_len; return rep[:max_len]+('...' if lng else '')
                 return f"<{type(v).__name__}>"
             except Exception: return "(SerializationError)"
        if kwargs: watch={'filepath','url','model_name','job_id','key_prefix','table_name','db_path','prompt','text','operation_name','source','version','loop_number','state_key','temperature','candidate_count'}; ctx.update({k:fmt_val(v,k) for k,v in kwargs.items() if k in watch or 'path' in k.lower() or 'file' in k.lower()})
        if args: ctx['path_args'] = [fmt_val(a,f"arg{i}_path") for i,a in enumerate(args) if isinstance(a,(str,pathlib.Path))][:3]; ctx['text_arg_snippets'] = [fmt_val(a,f"arg{i}_text",200) for i,a in enumerate(args) if isinstance(a,str)][:3]
        return {k:v for k,v in ctx.items() if v is not None}

    def _extract_api_context(self, args: Optional[Tuple], kwargs: Optional[Dict]) -> Dict[str, Any]:
        """API呼び出し用コンテキスト抽出"""
        ctx = self._extract_context(args, kwargs)
        if kwargs:
             if cfg := kwargs.get('generation_config'): ctx['gen_cfg_summary'] = {p:getattr(cfg,p,'?') for p in ['temperature','top_k','top_p','candidate_count'] if hasattr(cfg,p)} if hasattr(cfg,'__dict__') else {k:cfg.get(k) for k in ['temperature','top_k','top_p','candidate_count'] if k in cfg} if isinstance(cfg,dict) else str(cfg)
             if opts := kwargs.get('request_options'): ctx['request_timeout'] = opts.get('timeout') if isinstance(opts,dict) else None
        return ctx

    def _extract_file_context(self, args: Optional[Tuple], kwargs: Optional[Dict]) -> Dict[str, Any]:
        """ファイル操作用コンテキスト抽出"""
        ctx = self._extract_context(args, kwargs)
        fp = kwargs.get('filepath') or kwargs.get('path') or (args[0] if args and isinstance(args[0],(str,pathlib.Path)) else None)
        if fp: ctx['filepath'] = str(fp)
        return ctx

    def _extract_nlp_context(self, args: Optional[Tuple], kwargs: Optional[Dict]) -> Dict[str, Any]:
        """NLP処理用コンテキスト抽出"""
        return self._extract_context(args, kwargs)

    def _identify_error_code(self, exc: Exception) -> Tuple[str, str]:
        """例外からエラーコードと発生源を特定"""
        # --- ライブラリの動的チェックとクラス取得 ---
        g_ex = globals().get('google_exceptions') if GOOGLE_API_AVAILABLE else None
        yaml_err = globals().get('yaml').YAMLError if YAML_AVAILABLE and hasattr(globals().get('yaml'), 'YAMLError') else None # type: ignore
        ValError = globals().get('ValidationError') if PYDANTIC_AVAILABLE else None
        FileLockTimeout = globals().get('Timeout') if FILELOCK_AVAILABLE else None # type: ignore
        TransError = getattr(sys.modules.get('transformers'), 'errors', type('',(),{'TransformersError':Exception})).TransformersError if TRANSFORMERS_AVAILABLE and 'transformers' in sys.modules else None # type: ignore
        SpacyErrors = getattr(sys.modules.get('spacy'), 'errors', type('',(),{'Errors':Exception})).Errors if SPACY_AVAILABLE and 'spacy' in sys.modules else None # type: ignore
        # --- カテゴリ判定 ---
        cat="UNKNOWN"; src="Unknown"; msg=str(exc.args[0]).lower() if exc.args else ""
        if g_ex and isinstance(exc, g_ex.GoogleAPIError): cat="API"
        elif isinstance(exc, (FileNotFoundError,PermissionError,IsADirectoryError,NotADirectoryError)): cat="FILE"
        elif isinstance(exc, (IOError, OSError)) and "write" in msg: cat="FILE"
        elif isinstance(exc, OSError): cat="SYSTEM"
        elif isinstance(exc, sqlite3.Error): cat="DB"
        elif FileLockTimeout and isinstance(exc, FileLockTimeout) and 'FileLock' in repr(exc): cat="DB"
        elif isinstance(exc, ImportError): cat="SYSTEM"
        elif isinstance(exc, MemoryError): cat="SYSTEM"
        elif isinstance(exc, (ConnectionError, TimeoutError)): cat="NETWORK"
        elif yaml_err and isinstance(exc, yaml_err): cat="CONFIG"
        elif ValError and isinstance(exc, ValError): cat="MODEL"
        elif isinstance(exc, (ValueError,TypeError,AttributeError,KeyError,IndexError,NotImplementedError,RuntimeError)): cat="INTERNAL"
        elif TransError and isinstance(exc, TransError): cat="NLP_ML"
        elif SpacyErrors and isinstance(exc, SpacyErrors): cat="NLP_ML"
        else: cat="UNKNOWN"
        # --- 詳細コード特定 ---
        code_base = f"{cat}.{type(exc).__name__.upper()}"
        if cat=="API" and g_ex:
            src="ApiClientV49"; code_map={g_ex.ResourceExhausted:"RATE_LIMIT", g_ex.InternalServerError:"INTERNAL_ERROR", g_ex.ServiceUnavailable:"SERVICE_UNAVAILABLE", g_ex.DeadlineExceeded:"TIMEOUT", g_ex.PermissionDenied:"PERMISSION_DENIED", g_ex.Unauthenticated:"UNAUTHENTICATED", g_ex.NotFound:"NOT_FOUND", g_ex.BadRequest:"BAD_REQUEST", g_ex.InvalidArgument:"INVALID_ARGUMENT", g_ex.Aborted:"ABORTED", g_ex.Cancelled:"CANCELLED", g_ex.FailedPrecondition:"FAILED_PRECONDITION"}
            specific = code_map.get(type(exc), f"GENERIC_{type(exc).__name__.upper()}")
            if specific=="RATE_LIMIT" and "quota" in msg: specific="QUOTA_EXCEEDED"
            if "safety policies" in msg or specific=="FAILED_PRECONDITION": specific="SAFETY_BLOCK"
            if "api key" in msg and specific=="PERMISSION_DENIED": specific="INVALID_API_KEY"
            if "model" in msg and specific=="NOT_FOUND": specific="MODEL_NOT_FOUND"
            code=f"{cat}.{specific}"
        elif cat=="FILE": src="SystemIO/DialogueManager"; code = f"{cat}.{type(exc).__name__.upper()}"
        elif cat=="DB": src="PersistentCache/DialogueManager"; code=f"{cat}.GENERIC"; # ... (lock等詳細化) ...
        elif cat=="SYSTEM" and isinstance(exc, ImportError): code=f"SYSTEM.IMPORT_ERROR.{getattr(exc,'name','?')}"
        elif cat=="INTERNAL": src="Logic"; code=f"INTERNAL.{type(exc).__name__.upper()}" # ... (詳細化) ...
        else: code=code_base
        # --- フォールバック ---
        if code not in self.ERROR_SEVERITY: generic=f"{cat}.GENERIC" if cat!="UNKNOWN" else "UNKNOWN.GENERIC"; code = generic if generic in self.ERROR_SEVERITY else "UNKNOWN.GENERIC"
        return code, src

    def is_retryable(self, error: Union[Exception, StructuredErrorType]) -> bool: # type: ignore
        """エラーがリトライ可能か判定"""
        code=""; retry_flag=None
        if isinstance(error, StructuredErrorV49): retry_flag=error.is_retryable; code=error.code # type: ignore
        elif isinstance(error, Exception): code, _ = self._identify_error_code(error)
        else: return False
        if retry_flag is not None: return retry_flag
        return self.ERROR_SEVERITY.get(code, "FATAL") == "RECOVERABLE"

    def _estimate_root_cause(self, exc: Exception, code: str, context: Dict[str, Any]) -> Tuple[str, Optional[bool]]:
        """エラーコードとコンテキストから原因を推定"""
        retry_flag = self.is_retryable(exc); cat = code.split('.')[0] if '.' in code else "UNK"
        cause_map={"API":"API通信","FILE":"ファイルアクセス","DB":"DB","SYSTEM":"システム環境","NETWORK":"ネットワーク接続","CONFIG":"設定","MODEL":"データ/検証","NLP_ML":"NLP/ML処理","INTERNAL":"内部ロジック","GENERATION":"対話生成","EVAL":"評価","DFRS":"DFRS計算","UNKNOWN":"不明"}
        cause = cause_map.get(cat,f"不明カテゴリ({cat})") + f"問題(Code:{code})"
        # TODO: コンテキストに基づき詳細原因を追記 (例: file not found -> ファイルパス)
        return cause, retry_flag

    def log_error(self, error: Union[Exception, StructuredErrorType], source_override: Optional[str]=None, include_trace: Optional[bool]=None, context_data_override: Optional[Dict[str, Any]]=None) -> StructuredErrorType: # type: ignore
        """エラーを構造化しログ出力"""
        struct_err: StructuredErrorV49 # type: ignore
        if isinstance(error, StructuredErrorV49): struct_err=error; # type: ignore
        elif isinstance(error, Exception):
            code, src = self._identify_error_code(error); final_src = source_override or src
            ctx = context_data_override or {}; cause, retry = self._estimate_root_cause(error, code, ctx)
            struct_err = StructuredErrorV49(error, code, final_src, context_data=ctx, root_cause=cause, is_retryable=retry, original_exception=error) # type: ignore
        else: struct_err = StructuredErrorV49(f"非Exception型:{type(error)}", "INTERNAL.TYPE_ERROR", source_override or "Unknown") # type: ignore
        if source_override and not isinstance(error, StructuredErrorV49): struct_err.source = source_override # type: ignore
        if context_data_override and not isinstance(error, StructuredErrorV49): struct_err.context_data.update(context_data_override) # type: ignore

        self.error_counts[struct_err.code] += 1; severity = self.ERROR_SEVERITY.get(struct_err.code, "FATAL")
        trace = include_trace
        if trace is None: trace = (severity=="FATAL" or (severity=="RECOVERABLE" and self.error_counts[struct_err.code]==1)) and severity != "WARNING_ONLY"
        log_method = self.logger.critical if severity=="FATAL" else self.logger.warning if severity=="RECOVERABLE" else self.logger.info if severity=="WARNING_ONLY" else self.logger.error
        log_msg = str(struct_err) + (f" (Count:{self.error_counts[struct_err.code]})" if self.error_counts[struct_err.code]>1 else "")
        log_method(log_msg, exc_info=struct_err.original_exception if trace else None)
        if self.logger.isEnabledFor(logging.DEBUG) and struct_err.context_data:
             try: ctx_summary=struct_err.to_dict(True).get("context_data_summary"); self.logger.debug(f"  エラーコンテキスト:\n{json.dumps(ctx_summary,indent=2,ensure_ascii=False,default=str)}")
             except Exception: self.logger.debug("  エラーコンテキスト詳細ログ失敗")
        return struct_err # type: ignore

    def handle_with_retry(
        self,
        operation_name: str,
        operation: Callable[..., T], # type: ignore
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        extract_context_func: Optional[Callable[[Optional[Tuple[Any, ...]], Optional[Dict[str, Any]]], Dict[str, Any]]] = None
    ) -> Tuple[bool, Optional[T], Optional[StructuredErrorType]]: # type: ignore
        """操作をリトライ付きで実行 (StructuredErrorV49.message 属性対応)"""
        last_structured_error: Optional[ExceptionManagerV49.StructuredErrorType] = None # type: ignore
        args_tuple = args if args is not None else ()
        kwargs_dict = kwargs if kwargs is not None else {}
        source_component = "Unknown"

        try:
            # ... (既存のsource_component推定ロジック) ...
            if hasattr(operation, '__self__') and hasattr(operation.__self__, '__class__'):
                source_component = operation.__self__.__class__.__name__
            # ... (以下同様) ...
        except Exception:
            pass

        context_extractor = extract_context_func or self._extract_context
        attempt = 0
        current_max_retries = self.max_retries # デフォルト

        while True:
            try:
                result: T = operation(*args_tuple, **kwargs_dict) # type: ignore
                if attempt > 0:
                    self.logger.info(f"操作'{operation_name}'がリトライ{attempt}回目で成功しました。")
                return True, result, None # 成功

            except Exception as e:
                context_data = {}
                try:
                    context_data = context_extractor(args_tuple, kwargs_dict)
                except Exception as ctx_err:
                    self.logger.warning(f"操作'{operation_name}'のエラーコンテキスト抽出中にエラー: {ctx_err}")
                
                structured_error = self.log_error(e, source_override=source_component, context_data_override=context_data)
                last_structured_error = structured_error

                can_retry = self.is_retryable(structured_error)
                recovery_strategy = self.ERROR_RECOVERY_STRATEGIES.get(structured_error.code, {})
                current_max_retries = int(recovery_strategy.get("max_retries", self.max_retries)) # このエラーコードに対する最大リトライ回数
                base_delay_for_this_error = recovery_strategy.get("retry_delay", self.base_retry_delay)
                
                is_last_attempt = attempt >= current_max_retries

                if can_retry and not is_last_attempt:
                    retry_delay = exponential_backoff(attempt, base_delay_for_this_error, self.config.MAX_RETRY_DELAY) # type: ignore
                    self.logger.warning(
                        f"操作'{operation_name}'失敗 ({structured_error.code}, 試行{attempt + 1}/{current_max_retries + 1})。"
                        f" 原因: {structured_error.root_cause}. {retry_delay:.3f}秒後リトライ..."
                    )
                    time.sleep(retry_delay)
                    attempt += 1
                else:
                    reason = "NotRetryable" if not can_retry else "MaxAttemptsReached"
                    final_error_code = f"FINAL.{reason}_{structured_error.code}"
                    
                    # --- ▼▼▼ structured_error.message へのアクセスと代入 ▼▼▼ ---
                    # structured_error.message は StructuredErrorV49 に追加されたので直接アクセス可能
                    original_msg_for_final = structured_error.message or str(structured_error.original_exception)
                    structured_error.message = f"最終失敗({reason}): {original_msg_for_final}"
                    # --- ▲▲▲ ここまで ▲▲▲ ---
                                        
                    structured_error.code = final_error_code
                    structured_error.is_retryable = False

                    # 最後の試行のエラーカウントに基づいてトレースバックをログに出力するか判断
                    log_exc_info = (not can_retry and self.error_counts.get(last_structured_error.code if last_structured_error else "", 0) <= 1)

                    self.logger.error(
                        f"操作'{operation_name}'最終失敗 ({reason})。最終エラーコード: {final_error_code}",
                        exc_info=log_exc_info # 変更: 致命的な最初のエラーのみトレースバック
                    )
                    return False, None, structured_error

            # ループの最後に到達することは理論上ないはずだが、念のため
            # if attempt >= current_max_retries: # リトライ回数チェック
            #    break # ループを抜ける

        # while ループを抜けた場合 (通常はリトライ失敗で return される)
        # return False, None, StructuredErrorV49(
        #    f"リトライループ({current_max_retries+1}回)を予期せず終了。最後のエラー: {last_structured_error}",
        #    "INTERNAL.UNEXPECTED_LOOP_EXIT", source_component, is_retryable=False
        #) # type: ignore

    def safe_file_operation(self, name: str, func: Callable[..., T], args: Optional[Tuple]=None, kwargs: Optional[Dict]=None) -> Tuple[bool, Optional[T], Optional[StructuredErrorType]]: # type: ignore
        return self.handle_with_retry(f"FileOp:{name}", func, args, kwargs, self._extract_file_context)
    def safe_api_call(self, name: str, func: Callable[..., T], args: Optional[Tuple]=None, kwargs: Optional[Dict]=None) -> Tuple[bool, Optional[T], Optional[StructuredErrorType]]: # type: ignore
        return self.handle_with_retry(f"ApiCall:{name}", func, args, kwargs, self._extract_api_context)
    def safe_nlp_processing(self, name: str, func: Callable[..., T], args: Optional[Tuple]=None, kwargs: Optional[Dict]=None) -> Tuple[bool, Optional[T], Optional[StructuredErrorType]]: # type: ignore
        return self.handle_with_retry(f"NlpProc:{name}", func, args, kwargs, self._extract_nlp_context)

# =============================================================================
# -- Part 11 終了点
# =============================================================================
# =============================================================================
# -- Part 12: Dialogue Generator Class (Initialization & State) (v4.9α - 修正・最適化・堅牢性向上版 v13.3)
# =============================================================================
# v4.9α: 対話生成メインクラス。依存性注入、状態管理、v4.9コンポーネント統合。
# v13.3 Update:
# - Part 13 ヘルパーメソッドからの包括的データバンドルを処理するように Part 12 のメソッドを改修。
# - `_generate_initial_dialogue` (Part 12):
#   - Part 13 の `_generate_initial_dialogue_content_and_eval` を呼び出し、その結果 (辞書) を基に
#     `VersionStateV49` オブジェクトを構築し、返す。
# - `_execute_single_loop` (Part 12):
#   - Part 13 の `_generate_improved_dialogue_content_and_eval` を呼び出す際、前バージョンの
#     完全な評価バンドルを渡す。返された結果 (辞書) を基に `VersionStateV49` オブジェクトを構築し、返す。
# - `execute_generation_loops`:
#   - `_generate_initial_dialogue` および `_execute_single_loop` から返された `VersionStateV49` オブジェクトを
#     `self.state.versions` に確実に格納する。
# - `_handle_loop_result`:
#   - VersionStateV49 の作成は呼び出し元で行い、このメソッドはエラー処理と記録に集中。
# - ログ出力の強化と維持: オブジェクトIDと主要属性の状態を追跡。
# - 既存メソッドの維持: v13.2で安定している他のメソッドのコアロジックは維持。

from typing import (
    TYPE_CHECKING, TypeVar, Set, List, Dict, Optional, Tuple, Union, Any, Type, Literal, Callable, cast, TypeAlias
)
import enum
import logging
import random
import re
import math
import statistics
from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass, field as dataclass_field
import pathlib
import copy
import time
import traceback

# --- グローバルスコープで利用可能であることを期待する変数 (Part 0 などで定義済み) ---
PYDANTIC_AVAILABLE = globals().get('PYDANTIC_AVAILABLE', False)
BaseModel: Type[Any] = globals().get('BaseModel', object) # type: ignore
ConfigDict: Type[Dict[str, Any]] = globals().get('ConfigDict', dict) # type: ignore
Field: Callable[..., Any] = globals().get('Field', lambda **kwargs: None) # type: ignore
ValidationError: Type[Exception] = globals().get('ValidationError', ValueError) # type: ignore
_get_global_type: Callable[[str, Optional[type]], Optional[Type[Any]]] = \
    globals().get('_get_global_type', lambda name, meta=None: globals().get(name))
sanitize_filename: Callable[[str, Optional[int], str], str] = \
    globals().get('sanitize_filename', lambda f, ml=None, r='_': str(f))
fmt: Callable[[Optional[Union[float, int]], int, str], str] = \
    globals().get('fmt', lambda v, p=2, na="N/A": str(v))

# --- 依存性注入用データクラス (Part 12 の先頭で定義) ---
_GeneratorDependenciesV49_cls_check_p13_3 = _get_global_type('GeneratorDependenciesV49')

if not _GeneratorDependenciesV49_cls_check_p13_3: # フォールバック定義
    if PYDANTIC_AVAILABLE:
        class GeneratorDependenciesV49(BaseModel): # type: ignore
            config: 'AppConfigV49' # type: ignore
            settings: 'DialogueSettingsV49' # type: ignore
            api_client: 'ApiClientV49' # type: ignore
            analyzer: 'AdvancedDialogueAnalyzerV49' # type: ignore
            scorer: 'SubjectivityFluctuationScorerV49' # type: ignore
            evaluator: 'EnhancedDialogFlowEvaluatorV49' # type: ignore
            adaptation_strategy: 'PhaseToneAdaptationStrategyV49' # type: ignore
            feedback_manager: 'FeedbackManagerV49' # type: ignore
            prompt_builder: 'PromptBuilderV49' # type: ignore
            style_manager: Optional['DialogStyleManagerV49'] = Field(default=None) # type: ignore
            dialogue_manager: 'DialogueManagerV49' # type: ignore
            exception_manager: 'ExceptionManagerV49' # type: ignore
            model_config = ConfigDict(arbitrary_types_allowed=True) if ConfigDict else {} # type: ignore
    else:
        @dataclass
        class GeneratorDependenciesV49: # type: ignore
            config: 'AppConfigV49' # type: ignore
            settings: 'DialogueSettingsV49' # type: ignore
            api_client: 'ApiClientV49' # type: ignore
            analyzer: 'AdvancedDialogueAnalyzerV49' # type: ignore
            scorer: 'SubjectivityFluctuationScorerV49' # type: ignore
            evaluator: 'EnhancedDialogFlowEvaluatorV49' # type: ignore
            adaptation_strategy: 'PhaseToneAdaptationStrategyV49' # type: ignore
            feedback_manager: 'FeedbackManagerV49' # type: ignore
            prompt_builder: 'PromptBuilderV49' # type: ignore
            style_manager: Optional['DialogStyleManagerV49'] = None # type: ignore
            dialogue_manager: 'DialogueManagerV49' # type: ignore
            exception_manager: 'ExceptionManagerV49' # type: ignore
    _GeneratorDependenciesV49_cls_check_p13_3 = GeneratorDependenciesV49 # type: ignore
    logging.getLogger(__name__).info("Part 12 (v13.3): GeneratorDependenciesV49 のフォールバック定義を使用しました。")

# --- Type Aliases for method signatures ---
CharacterInputType = Dict[str, Any]
SceneInfoInputType = Dict[str, Any]

# --- メインジェネレータクラス ---
class DialogueGeneratorV49:
    """NDGS v4.9α 対話生成メイン処理フロー (改善版 v13.3)"""
    if TYPE_CHECKING:
        from __main__ import ( # type: ignore[attr-defined]
            ConfigProtocol, SettingsProtocol, ApiClientProtocol, AnalyzerProtocol,
            ScorerProtocol, EvaluatorProtocol, AdaptationStrategyProtocol,
            FeedbackManagerProtocol, PromptBuilderProtocol, StyleManagerProtocol,
            DialogueManagerProtocol, ExceptionManagerProtocol,
            AppConfigV49, DialogueSettingsV49, ApiClientV49, AdvancedDialogueAnalyzerV49,
            SubjectivityFluctuationScorerV49, EnhancedDialogFlowEvaluatorV49,
            PhaseToneAdaptationStrategyV49, FeedbackManagerV49, PromptBuilderV49,
            DialogStyleManagerV49, DialogueManagerV49, ExceptionManagerV49,
            GeneratorStateV49, InputDataV49, CharacterV49, SceneInfoV49,
            VersionStateV49, LLMEvaluationScoresV49, DFRSSubScoresV49,
            StructuredErrorV49, PsychologicalPhaseV49, EmotionalToneV49,
            FinalSelectionKeysV49, ScoreKeys, DFRSMetricsV49, InitialSelectionKeysV49,
            OutputEvaluationV49, GenerationStatsV49, PhaseTransitionRecordV49, FeedbackContextV49,
            TemperatureStrategyConfigV49, ErrorConfigV49 # Part 13から移動またはここで定義期待
        )
        ScoreKeysLLMEnumType: TypeAlias = ScoreKeys.LLM
        GeneratorStateV49_T: TypeAlias = GeneratorStateV49
        StructuredErrorV49_T: TypeAlias = StructuredErrorV49
        VersionStateV49_Model_T: TypeAlias = VersionStateV49
        PsychologicalPhaseType: TypeAlias = PsychologicalPhaseV49
        EmotionalToneType: TypeAlias = EmotionalToneV49
        StructuredErrorType: TypeAlias = StructuredErrorV49_T
    else:
        ConfigProtocol = 'ConfigProtocol'; SettingsProtocol = 'SettingsProtocol'; ApiClientProtocol = 'ApiClientProtocol'
        AnalyzerProtocol = 'AnalyzerProtocol'; ScorerProtocol = 'ScorerProtocol'; EvaluatorProtocol = 'EvaluatorProtocol'
        AdaptationStrategyProtocol = 'AdaptationStrategyProtocol'; FeedbackManagerProtocol = 'FeedbackManagerProtocol'
        PromptBuilderProtocol = 'PromptBuilderProtocol'; StyleManagerProtocol = 'StyleManagerProtocol'
        DialogueManagerProtocol = 'DialogueManagerProtocol'; ExceptionManagerProtocol = 'ExceptionManagerProtocol'
        AppConfigV49 = 'AppConfigV49'; DialogueSettingsV49 = 'DialogueSettingsV49'; ApiClientV49 = 'ApiClientV49'
        AdvancedDialogueAnalyzerV49 = 'AdvancedDialogueAnalyzerV49'; SubjectivityFluctuationScorerV49 = 'SubjectivityFluctuationScorerV49'
        EnhancedDialogFlowEvaluatorV49 = 'EnhancedDialogFlowEvaluatorV49'; PhaseToneAdaptationStrategyV49 = 'PhaseToneAdaptationStrategyV49'
        FeedbackManagerV49 = 'FeedbackManagerV49'; PromptBuilderV49 = 'PromptBuilderV49'; DialogStyleManagerV49 = 'DialogStyleManagerV49'
        DialogueManagerV49 = 'DialogueManagerV49'; ExceptionManagerV49 = 'ExceptionManagerV49'
        GeneratorStateV49_T = 'GeneratorStateV49'; InputDataV49 = 'InputDataV49'; CharacterV49 = 'CharacterV49'; SceneInfoV49 = 'SceneInfoV49'
        VersionStateV49_Model_T = 'VersionStateV49'; LLMEvaluationScoresV49 = 'LLMEvaluationScoresV49'; DFRSSubScoresV49 = 'DFRSSubScoresV49'
        StructuredErrorV49_T = 'StructuredErrorV49'; PsychologicalPhaseV49 = 'PsychologicalPhaseV49'; EmotionalToneV49 = 'EmotionalToneV49'
        FinalSelectionKeysV49 = 'FinalSelectionKeysV49'; ScoreKeysLLMEnumType = 'ScoreKeys_LLM'; DFRSMetricsV49 = 'DFRSMetricsV49'
        InitialSelectionKeysV49 = 'InitialSelectionKeysV49'; OutputEvaluationV49 = 'OutputEvaluationV49'
        GenerationStatsV49 = 'GenerationStatsV49'; PhaseTransitionRecordV49 = 'PhaseTransitionRecordV49'; FeedbackContextV49 = 'FeedbackContextV49'
        TemperatureStrategyConfigV49 = 'TemperatureStrategyConfigV49'; ErrorConfigV49 = 'ErrorConfigV49'
        PsychologicalPhaseType = 'PsychologicalPhaseType'; EmotionalToneType = 'EmotionalToneType'; StructuredErrorType = 'StructuredErrorType'


    # Part 13 Helper Methods (as defined in user's snippet, with Gemini's v13.3 optimizations)
    # These are now assumed to be part of the DialogueGeneratorV49 class
    # For brevity, only method signatures are listed here, implementation is in the full code block below.
    # _create_error(...) -> 'StructuredErrorV49_T'
    # _evaluate_and_score_candidate(...) -> Dict[str, Any]
    # _select_best_initial_candidate(...) -> Optional[Dict[str, Any]]
    # _suggest_dialogue_mode(...) -> Literal["normal", "delayed", "mixed", "auto"]
    # _generate_initial_dialogue_content_and_eval(...) -> Dict[str, Any]
    # _generate_improved_dialogue_content_and_eval(...) -> Dict[str, Any]


    def __init__(self, job_id_base: str, dependencies: 'GeneratorDependenciesV49'): # type: ignore
        self.job_id_base: str = job_id_base
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}.v13.3") # v13.3 Update

        GeneratorDependenciesV49_cls = _get_global_type('GeneratorDependenciesV49')
        if not (GeneratorDependenciesV49_cls and isinstance(dependencies, GeneratorDependenciesV49_cls)):
            error_msg_deps_init_critical = (f"CRITICAL: 'dependencies'引数が予期された型ではありません。")
            self.logger.critical(error_msg_deps_init_critical)
            raise TypeError(error_msg_deps_init_critical)

        self.dep: 'GeneratorDependenciesV49' = dependencies # type: ignore
        
        self.config: 'ConfigProtocol' = self.dep.config # type: ignore
        self.settings: 'SettingsProtocol' = self.dep.settings # type: ignore
        self.api_client: 'ApiClientProtocol' = self.dep.api_client # type: ignore
        self.analyzer: 'AnalyzerProtocol' = self.dep.analyzer # type: ignore
        self.scorer: 'ScorerProtocol' = self.dep.scorer # type: ignore
        self.evaluator: 'EvaluatorProtocol' = self.dep.evaluator # type: ignore
        self.adaptation_strategy: 'AdaptationStrategyProtocol' = self.dep.adaptation_strategy # type: ignore
        self.feedback_manager: 'FeedbackManagerProtocol' = self.dep.feedback_manager # type: ignore
        self.prompt_builder: 'PromptBuilderProtocol' = self.dep.prompt_builder # type: ignore
        self.style_manager: Optional['StyleManagerProtocol'] = self.dep.style_manager # type: ignore
        self.dialogue_manager: 'DialogueManagerProtocol' = self.dep.dialogue_manager # type: ignore
        self.exception_manager: 'ExceptionManagerProtocol' = self.dep.exception_manager # type: ignore

        self.job_id: str = self._sanitize_job_id(job_id_base)
        self.state: Optional['GeneratorStateV49_T'] = None # type: ignore

        models_and_enums_to_load: Dict[str, str] = {
            "GeneratorStateModel": "GeneratorStateV49", "InputDataModel": "InputDataV49",
            "CharacterModel": "CharacterV49", "SceneInfoModel": "SceneInfoV49",
            "VersionStateModel": "VersionStateV49", "LLMEvalScoresModel": "LLMEvaluationScoresV49",
            "DFRSSubScoresModel": "DFRSSubScoresV49", "StructuredErrorModel": "StructuredErrorV49",
            "FinalSelKeys_cls": "FinalSelectionKeysV49", "LLMKeys_cls": "ScoreKeys.LLM",
            "DFRSMetrics_cls": "DFRSMetricsV49", "PsychologicalPhase_cls": "PsychologicalPhaseV49",
            "EmotionalTone_cls": "EmotionalToneV49", "OutputEvaluationModel": "OutputEvaluationV49",
            "GenerationStatsModel": "GenerationStatsV49", "PhaseTransitionRecordModel": "PhaseTransitionRecordV49",
            "FeedbackContextModel": "FeedbackContextV49", "InitialSelectionKeys_cls": "InitialSelectionKeysV49",
            "TemperatureStrategyConfigV49_cls": "TemperatureStrategyConfigV49", # For _calculate_generation_params
            "ErrorConfigV49_cls": "ErrorConfigV49" # For execute_generation_loops
        }
        init_errors_list: List[str] = []
        for attr_name, class_name_str in models_and_enums_to_load.items():
            loaded_class_obj: Optional[Type] = None
            try:
                if "." in class_name_str:
                    parent_name, child_name = class_name_str.split(".", 1)
                    parent_obj = _get_global_type(parent_name)
                    if parent_obj: loaded_class_obj = getattr(parent_obj, child_name, None)
                else:
                    loaded_class_obj = _get_global_type(class_name_str)
                
                if not loaded_class_obj: raise NameError(f"'{class_name_str}'未発見")
                setattr(self, attr_name, loaded_class_obj)
            except (NameError, AttributeError) as e_load_attr:
                init_errors_list.append(f"{attr_name} (from {class_name_str}): {e_load_attr}")
                setattr(self, attr_name, None)
        
        if init_errors_list:
            err_summary = f"CRITICAL: __init__必須クラス/Enumロード失敗: {'; '.join(init_errors_list)}"
            self.logger.critical(err_summary)
            raise ImportError(err_summary)
        
        if not getattr(self, 'GeneratorStateModel', None) or not getattr(self, 'VersionStateModel', None):
            raise AttributeError("CRITICAL: GeneratorStateModelまたはVersionStateModelがロードできませんでした。")

        self.logger.info(f"DialogueGeneratorV49 (System: {getattr(self.config, 'SYSTEM_VERSION', 'N/A')}, Job ID: {self.job_id}) 依存関係・必須クラスロード完了。") # type: ignore
        self._log_initial_settings()
        self._current_generation_params: Dict[str,Any] = {}
        self._handle_startup_cache_cleanup()
        self.logger.info(f"DialogueGeneratorV49 (Job ID: {self.job_id}) 初期化正常完了。")

    def _sanitize_job_id(self, base: str) -> str:
        prefix = getattr(self.config, 'DEFAULT_JOB_ID_PREFIX', "job_") # type: ignore
        ts = int(datetime.now(timezone.utc).timestamp())
        safe_base = base if base and isinstance(base, str) and base.strip() else f"{prefix}{ts}"
        filename_max_len = getattr(self.config, 'filename_max_length', 150) # type: ignore
        
        if callable(sanitize_filename):
            return sanitize_filename(safe_base, filename_max_len)
        else:
            self.logger.warning("sanitize_filename関数が見つかりません。簡易的なサニタイズを行います。")
            sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', safe_base)
            return sanitized[:filename_max_len]

    def _log_initial_settings(self) -> None:
        self.logger.info(f"  主要ジョブ設定: (詳細はDEBUGログで確認してください)")

    def _handle_startup_cache_cleanup(self) -> None:
        pass

    def _initialize_state(
        self,
        char_a_in: Dict[str, Any],
        char_b_in: Dict[str, Any],
        scene_in: Dict[str, Any],
        target_len: int
    ) -> Optional['GeneratorStateV49_T']: # type: ignore
        self.logger.info(f"ジョブ '{self.job_id}' の初期状態を作成しています...")
        start_time = datetime.now(timezone.utc)
        settings_snapshot: Dict[str, Any] = {}
        try:
            if hasattr(self.settings, 'model_dump') and callable(self.settings.model_dump): # type: ignore
                settings_snapshot = self.settings.model_dump(mode='json') # type: ignore
            elif PYDANTIC_AVAILABLE and isinstance(self.settings, BaseModel): # type: ignore
                exclude_fields_snapshot = {'config', 'logger', '_config', '_validated_initial_candidate_weights'}
                settings_snapshot = self.settings.model_dump(mode='json', by_alias=True, exclude_none=True, exclude=exclude_fields_snapshot, warn=False) # type: ignore
            else:
                self.logger.warning("Settings snapshot: Manual copy, may be incomplete.")
                settings_snapshot = {k: v for k, v in vars(self.settings).items() if not k.startswith('_') and not callable(v)}
            settings_snapshot['snapshot_creation_method'] = 'model_dump' if hasattr(self.settings, 'model_dump') else 'manual'
            settings_snapshot['api_model_name_at_init'] = getattr(self.config, 'DEFAULT_MODEL', 'N/A') # type: ignore
            settings_snapshot['system_version_at_init'] = getattr(self.config, 'SYSTEM_VERSION', 'N/A') # type: ignore
        except Exception as e_snap: settings_snapshot = {"error": str(e_snap)}

        if not (self.InputDataModel and self.CharacterModel and self.SceneInfoModel and # type: ignore
                self.PsychologicalPhase_cls and self.EmotionalTone_cls and self.GeneratorStateModel): # type: ignore
            self.logger.critical("初期状態作成に必要なモデルクラスがロードされていません。")
            raise ImportError("初期状態作成に必要なモデルクラス未ロード。")

        try:
            char_a_model = self.CharacterModel.model_validate(char_a_in) # type: ignore
            char_b_model = self.CharacterModel.model_validate(char_b_in) # type: ignore
            scene_model = self.SceneInfoModel.model_validate(scene_in) # type: ignore
            input_data_model = self.InputDataModel(characterA=char_a_model, characterB=char_b_model, sceneInfo=scene_model) # type: ignore
        except ValidationError as e_val: raise ValueError(f"入力データ形式無効: {e_val}") from e_val # type: ignore
        except Exception as e_input_other: raise RuntimeError(f"入力データモデル作成失敗: {e_input_other}") from e_input_other

        initial_phase = getattr(self.PsychologicalPhase_cls, 'INTRODUCTION', None) # type: ignore
        initial_tone = getattr(self.EmotionalTone_cls, 'NEUTRAL', None) # type: ignore

        state_data = {
            "job_id": self.job_id, "system_version": getattr(self.config, 'SYSTEM_VERSION', 'N/A'), # type: ignore
            "model_name": getattr(self.config, 'DEFAULT_MODEL', 'N/A'), "start_time": start_time, # type: ignore
            "input_data": input_data_model, "target_length": target_len,
            "settings_snapshot": settings_snapshot, "current_loop": 0,
            "current_intended_phase": initial_phase, "current_intended_tone": initial_tone,
            "versions": [], "error_records": [], "temperature_history": [], "adj_factor_history": []
        }
        try:
            if not self.GeneratorStateModel: raise AttributeError("self.GeneratorStateModel未設定") # type: ignore
            self.state = self.GeneratorStateModel.model_validate(state_data) # type: ignore
        except ValidationError as e_state_val: # type: ignore
             self.logger.critical(f"GeneratorStateV49 モデル検証エラー(初期化時): {e_state_val}", exc_info=True)
             raise RuntimeError(f"GeneratorStateV49 初期化検証エラー: {e_state_val}") from e_state_val
        
        if not self.state:
             self.logger.error(f"ジョブ '{self.job_id}': _initialize_state内でself.stateがNoneのままです。")
             return None
        self.logger.info(f"GeneratorStateV49 初期化完了 (Job ID: {self.job_id})。")
        
        is_resume_enabled = getattr(getattr(self.config, 'feature_flags', object()), 'enable_resume_feature', True) # type: ignore
        if is_resume_enabled:
            if self.dep.dialogue_manager and hasattr(self.dep.dialogue_manager, 'save_resume_state'): # type: ignore
                save_ok, _, err_obj = self.dep.exception_manager.safe_file_operation( # type: ignore
                    name="初期レジューム状態保存", func=self.dep.dialogue_manager.save_resume_state, args=(self.state,) # type: ignore
                )
                if not save_ok:
                    err_msg = f"初期レジューム状態の保存に失敗: {err_obj}"
                    if getattr(getattr(self.config, 'feature_flags', object()), 'resume_save_critical', False): # type: ignore
                        self.logger.critical(err_msg)
                        orig_exc = getattr(err_obj, 'original_exception', None) or (err_obj if isinstance(err_obj, Exception) else RuntimeError(str(err_obj)))
                        raise RuntimeError(err_msg) from orig_exc
                    else: self.logger.error(err_msg + " (処理続行)")
            elif getattr(getattr(self.config, 'feature_flags', object()), 'resume_save_critical', False): # type: ignore
                 self.logger.critical("DialogueManager未初期化/save_resume_stateなし。必須初期状態保存不可。")
                 raise RuntimeError("DialogueManager未初期化/save_resume_stateなし。必須初期状態保存不可。")
            else: self.logger.warning("DialogueManager未初期化/save_resume_stateなし。初期状態保存不可。")
        return self.state

    def _update_state_with_current_settings(self) -> None:
        if not self.state: self.logger.warning("GeneratorState None。設定スナップショット更新不可。"); return
        self.logger.debug("現DialogueSettingsでGeneratorState設定スナップショット更新...")
        try:
            if hasattr(self.settings, 'model_dump') and callable(self.settings.model_dump): # type: ignore
                self.state.settings_snapshot = self.settings.model_dump(mode='json') # type: ignore
            elif PYDANTIC_AVAILABLE and isinstance(self.settings, BaseModel): # type: ignore
                exclude_fields = {'config', 'logger', '_config', '_validated_initial_candidate_weights'}
                self.state.settings_snapshot = self.settings.model_dump(mode='json', by_alias=True, exclude_none=True, exclude=exclude_fields, warn=False) # type: ignore
            else:
                self.state.settings_snapshot = {k: v for k, v in vars(self.settings).items() if not k.startswith('_') and not callable(v)}
        except Exception as e: self.logger.error(f"設定スナップショット更新エラー: {e}", exc_info=True); self.state.settings_snapshot = {"error": str(e)} # type: ignore
        self.logger.info("GeneratorStateのsettings_snapshotを更新しました。")

    def _initialize_or_resume_state(
        self,
        character_a_input: Dict[str, Any],
        character_b_input: Dict[str, Any],
        scene_info_input: Dict[str, Any],
        target_length_input: int
    ) -> bool:
        self.logger.info("状態初期化/レジューム試行...")
        resumed_state: Optional['GeneratorStateV49_T'] = None # type: ignore
        is_resume_enabled = getattr(getattr(self.config, 'feature_flags', object()), 'enable_resume_feature', True) # type: ignore

        if is_resume_enabled and self.dep.dialogue_manager and hasattr(self.dep.dialogue_manager, 'load_resume_state'): # type: ignore
            load_success, loaded_data, load_error = self.dep.exception_manager.safe_file_operation( # type: ignore
                name="レジューム状態ロード", func=self.dep.dialogue_manager.load_resume_state # type: ignore
            )
            if load_success and loaded_data is not None:
                if self.GeneratorStateModel:
                    if isinstance(loaded_data, self.GeneratorStateModel): # type: ignore
                        resumed_state = loaded_data # type: ignore
                    elif isinstance(loaded_data, dict):
                        try:
                            resumed_state = self.GeneratorStateModel.model_validate(loaded_data) # type: ignore
                        except ValidationError as ve: self.logger.error(f"レジュームデータ検証エラー: {ve}。新規扱い。") # type: ignore
                        except Exception as e: self.logger.error(f"レジュームデータ変換エラー: {e}。新規扱い。", exc_info=True)
                    else: self.logger.warning(f"ロードレジュームデータ型不正({type(loaded_data)})。新規扱い。")
                else: self.logger.critical("GeneratorStateModel未ロード。レジュームデータ検証不可。")
            elif load_error: self.logger.error(f"レジューム状態ロードエラー: {load_error}")
        
        if resumed_state:
            self.state = resumed_state
            if self.state.job_id != self.job_id: # type: ignore
                 self.state.job_id = self.job_id # type: ignore
            self.logger.info(f"ジョブ '{self.job_id}' 状態レジューム (Loop: {self.state.current_loop})。") # type: ignore
            self._update_state_with_current_settings()
            if self.state.temperature_history: # type: ignore
                 self._current_generation_params['temperature'] = self.state.temperature_history[-1] # type: ignore
            return True
        else:
            self.logger.info("レジューム不可。新規ジョブとして初期化。")
            try:
                self.state = self._initialize_state(character_a_input, character_b_input, scene_info_input, target_length_input)
                return self.state is not None
            except Exception as e:
                self.logger.critical(f"新規ジョブ初期状態作成中致命的エラー: {e}", exc_info=True)
                return False

    # =============================================================================
    # ▼▼▼ Part 12: Core Loop Methods (v13.3 Update) ▼▼▼
    # =============================================================================

    def _create_version_state_from_bundle(
        self,
        version_id: int,
        loop_eval_bundle: Dict[str, Any],
        generation_time_ms: Optional[float],
        evaluation_time_ms: Optional[float],
        prompt_text: Optional[str] = None, # For initial prompt
        feedback_text_for_prompt: Optional[str] = None # For improvement prompt
    ) -> 'VersionStateV49_Model_T': # type: ignore
        """
        Helper to create and populate a VersionStateV49 object from an evaluation bundle.
        """
        if not (self.VersionStateModel and self.LLMEvalScoresModel and self.DFRSSubScoresModel): # type: ignore
            self.logger.critical("VersionState構築に必要なモデルクラスが不足しています。")
            raise ImportError("VersionState構築に必須のモデルが未ロードです。")

        error_info_final = loop_eval_bundle.get("error_info_dict") # This should be a dict from _create_error
        
        llm_scores_dict = loop_eval_bundle.get("llm_scores_dict", {})
        llm_scores_model_instance = None
        if isinstance(llm_scores_dict, dict) and "error" not in llm_scores_dict and "skipped" not in llm_scores_dict:
            try:
                llm_scores_model_instance = self.LLMEvalScoresModel.model_validate(llm_scores_dict) # type: ignore
                if isinstance(loop_eval_bundle.get("evaluation_raw"), str):
                    llm_scores_model_instance.raw_output = loop_eval_bundle["evaluation_raw"] # type: ignore
            except ValidationError as e_val: # type: ignore
                self.logger.warning(f"v{version_id}: LLMスコア辞書のLLMEvaluationScoresV49モデル検証失敗: {e_val}")
                # エラー情報をllm_scoresモデル内に記録することも検討できるが、一旦Noneとする
                llm_scores_model_instance = None # type: ignore
                if not error_info_final : error_info_final = {"code":"LLM_SCORES_VALIDATION", "message":str(e_val)}

        dfrs_scores_payload = loop_eval_bundle.get("dfrs_scores_payload")
        dfrs_scores_model_instance = None
        if isinstance(dfrs_scores_payload, dict) and "error" not in dfrs_scores_payload and "skipped" not in dfrs_scores_payload:
            try:
                dfrs_scores_model_instance = self.DFRSSubScoresModel.model_validate(dfrs_scores_payload) # type: ignore
            except ValidationError as e_val_dfrs: # type: ignore
                self.logger.warning(f"v{version_id}: DFRSスコアペイロードのDFRSSubScoresV49モデル検証失敗: {e_val_dfrs}")
                # エラー情報をdfrs_scoresモデル内に記録
                dfrs_scores_model_instance = self.DFRSSubScoresModel.model_validate({"scores": {"error": f"DFRS payload validation error: {e_val_dfrs}"}}) # type: ignore
                if not error_info_final : error_info_final = {"code":"DFRS_SCORES_VALIDATION", "message":str(e_val_dfrs)}
        elif isinstance(dfrs_scores_payload, dict) and ("error" in dfrs_scores_payload or "skipped" in dfrs_scores_payload):
             # DFRS評価自体でエラーまたはスキップされた場合、その情報を保持
             dfrs_scores_model_instance = self.DFRSSubScoresModel.model_validate({"scores": dfrs_scores_payload}) # type: ignore


        analyzer_results = loop_eval_bundle.get("analyzer_results_dict", {})
        if not isinstance(analyzer_results, dict): # Should always be a dict from Part 13
            self.logger.error(f"CRITICAL: v{version_id}: analyzer_results_dictが辞書ではありません ({type(analyzer_results)})。空の辞書でフォールバックします。")
            analyzer_results = {"error": "Received non-dict analyzer_results", "type": str(type(analyzer_results))}
            if not error_info_final : error_info_final = {"code":"ANALYZER_INVALID_TYPE", "message":"Analyzer results not a dict"}

        version_state_construct_data = {
            "version_id": version_id,
            "generated_text": loop_eval_bundle.get("text"),
            "prompt_text": prompt_text,
            "feedback_text": feedback_text_for_prompt,
            "evaluation_text_raw": loop_eval_bundle.get("evaluation_raw"),
            "llm_scores": llm_scores_model_instance,
            "dfrs_scores": dfrs_scores_model_instance,
            "analyzer_results": analyzer_results,
            "generation_time_ms": generation_time_ms,
            "evaluation_time_ms": evaluation_time_ms,
            "generation_model": self.dep.api_client.model_name, # type: ignore
            "status": "error" if error_info_final else "completed",
            "error_info": error_info_final,
            "timestamp": datetime.now(timezone.utc),
            "estimated_subjectivity": analyzer_results.get("subjectivity_score_final") if isinstance(analyzer_results, dict) else None,
            "estimated_fluctuation": analyzer_results.get("fluctuation_intensity_final") if isinstance(analyzer_results, dict) else None,
        }
        
        new_vs_instance = self.VersionStateModel.model_validate(version_state_construct_data) # type: ignore
        self.logger.debug(f"v{version_id}: VersionStateオブジェクト(obj_id:{id(new_vs_instance)})作成完了。")
        self.logger.debug(f"  作成されたVS: analyzer_results (obj_id:{id(new_vs_instance.analyzer_results)}) type: {type(new_vs_instance.analyzer_results)}, Keys: {list(new_vs_instance.analyzer_results.keys()) if new_vs_instance.analyzer_results else 'N/A'}") # type: ignore
        self.logger.debug(f"  作成されたVS: llm_scores (obj_id:{id(new_vs_instance.llm_scores)}) type: {type(new_vs_instance.llm_scores)}, Overall: {getattr(new_vs_instance.llm_scores, 'overall', 'N/A') if new_vs_instance.llm_scores else 'None'}") # type: ignore
        return new_vs_instance


    def _generate_initial_dialogue(self) -> Tuple[Optional[str], Optional['StructuredErrorV49_T'], Optional['VersionStateV49_Model_T']]: # type: ignore
        """
        (Part 12) Orchestrates initial dialogue generation by calling Part 13's helper and constructing VersionStateV49.
        """
        initial_loop_idx = 0
        current_version_id = initial_loop_idx + 1
        self.logger.info(f"--- (Part 12) 初期対話生成 (v{current_version_id}) オーケストレーション開始 ---")

        if not self.state or not self.state.input_data: # type: ignore
            # ... (error handling as in v13.2) ...
            err_msg = "初期対話生成: GeneratorStateまたは入力データ未初期化。"
            self.logger.critical(err_msg)
            # _create_error is now part of Part 13 (self)
            return None, self._create_error("InitialDialogue.NoState", err_msg, "_generate_initial_dialogue.Part12"), None
        
        # Call Part 13 helper
        # _generate_initial_dialogue_content_and_eval is now a method of self
        start_generation_time = time.perf_counter()
        eval_bundle = self._generate_initial_dialogue_content_and_eval( # type: ignore
            self.state.input_data.characterA.model_dump(by_alias=True, exclude_none=True), # type: ignore
            self.state.input_data.characterB.model_dump(by_alias=True, exclude_none=True), # type: ignore
            self.state.input_data.sceneInfo.model_dump(by_alias=True, exclude_none=True), # type: ignore
            self.state.target_length # type: ignore
        )
        generation_time_ms = (time.perf_counter() - start_generation_time) * 1000
        # evaluation_time_ms is tricky here as Part 13 helper does its own timing internally for LLM eval.
        # This generation_time_ms covers the whole Part 13 helper call.

        generated_text: Optional[str] = eval_bundle.get("text") # type: ignore
        error_info_dict = eval_bundle.get("error_info_dict")
        structured_error: Optional['StructuredErrorV49_T'] = None # type: ignore

        if error_info_dict:
            if self.StructuredErrorModel: # type: ignore
                structured_error = self.StructuredErrorModel.model_validate(error_info_dict) # type: ignore
            else: # Should not happen
                self.logger.error("StructuredErrorModel is not loaded, cannot create structured error from dict.")
                structured_error = RuntimeError(str(error_info_dict)) # type: ignore
            self.logger.error(f"Part 13ヘルパー(_generate_initial_dialogue_content_and_eval)がエラーを返しました: {error_info_dict.get('code','N/A')}")
            # Attempt to create a VersionState even with error
            version_state = self._create_version_state_from_bundle(current_version_id, eval_bundle, generation_time_ms, None, eval_bundle.get("prompt_text_debug"))
            return generated_text, structured_error, version_state

        # Successfully got a bundle, create VersionState
        # Prompt text might not be in the bundle, it's saved by Part 13 helper.
        # For now, pass None or retrieve if added to bundle.
        version_state = self._create_version_state_from_bundle(current_version_id, eval_bundle, generation_time_ms, None, eval_bundle.get("prompt_text_debug")) # TODO: Get eval time from bundle if available
        
        self.logger.info(f"--- (Part 12) 初期対話生成 (v{current_version_id}) オーケストレーション完了 ---")
        return generated_text, None, version_state


    def _execute_single_loop(self, improve_loop_num_1_based: int) -> Tuple[Optional[str], Optional['StructuredErrorV49_T'], Optional['VersionStateV49_Model_T']]: # type: ignore
        """
        (Part 12) Orchestrates a single improvement loop by calling Part 13's helper and constructing VersionStateV49.
        """
        current_version_id = improve_loop_num_1_based + 1
        self.logger.info(f"--- (Part 12) 改善ループ {improve_loop_num_1_based} (v{current_version_id}) オーケストレーション開始 ---")

        if not self.state or not self.state.versions: # type: ignore
            err_msg = "改善ループ: GeneratorStateまたはversionsリスト未初期化。"
            self.logger.critical(err_msg)
            return None, self._create_error("ImproveLoop.NoStateOrVersions", err_msg, f"_execute_single_loop.L{current_version_id}.Part12"), None

        prev_version_idx = improve_loop_num_1_based -1 # This is the 0-indexed list index for the previous VersionState
        if not (0 <= prev_version_idx < len(self.state.versions) and self.state.versions[prev_version_idx]): # type: ignore
            err_msg = f"改善ループv{current_version_id}: 前バージョン(idx:{prev_version_idx})データ未発見またはNone。"
            self.logger.error(err_msg)
            return None, self._create_error("ImproveLoop.PrevVersionNotFound", err_msg, f"_execute_single_loop.L{current_version_id}.Part12"), None
        
        prev_version_state = self.state.versions[prev_version_idx] # type: ignore
        if not prev_version_state.generated_text: # type: ignore
            err_msg = f"改善ループv{current_version_id}: 前バージョンv{prev_version_state.version_id}の生成テキストなし。" # type: ignore
            self.logger.error(err_msg)
            return None, self._create_error("ImproveLoop.PrevVersionTextMissing", err_msg, f"_execute_single_loop.L{current_version_id}.Part12"), None

        # Prepare the evaluation bundle of the PREVIOUS version to pass to the Part 13 helper
        # This assumes VersionStateV49 objects store data in a way that can be reconstructed into the bundle.
        prev_eval_bundle_for_helper = {
            "text": prev_version_state.generated_text, # type: ignore
            "evaluation_raw": prev_version_state.evaluation_text_raw, # type: ignore
            "llm_scores_dict": prev_version_state.llm_scores.model_dump(exclude_none=True) if prev_version_state.llm_scores else {}, # type: ignore
            "dfrs_scores_payload": prev_version_state.dfrs_scores.model_dump(exclude_none=True) if prev_version_state.dfrs_scores else None, # type: ignore
             # Ensure analyzer_results is a dict, even if it had an error.
            "analyzer_results_dict": prev_version_state.analyzer_results if isinstance(prev_version_state.analyzer_results, dict) else {"error": "Previous analyzer_results was not a dict"}, # type: ignore
            "error_info_dict": prev_version_state.error_info # type: ignore
        }
        self.logger.debug(f"Improve Loop v{current_version_id}: Prev Version (v{prev_version_state.version_id}, obj_id:{id(prev_version_state)}) Bundle for Part 13 Helper:") # type: ignore
        self.logger.debug(f"  Prev analyzer_results type: {type(prev_version_state.analyzer_results)}, keys: {list(prev_version_state.analyzer_results.keys()) if isinstance(prev_version_state.analyzer_results, dict) else 'N/A'}") # type: ignore
        self.logger.debug(f"  Prev llm_scores type: {type(prev_version_state.llm_scores)}, overall: {getattr(prev_version_state.llm_scores, 'overall', 'N/A') if prev_version_state.llm_scores else 'None'}") # type: ignore

        # Call Part 13 helper for generating improved dialogue and its evaluation
        start_generation_time = time.perf_counter()
        eval_bundle_improved = self._generate_improved_dialogue_content_and_eval( # type: ignore
            prev_dialogue_text=prev_version_state.generated_text, # type: ignore
            prev_version_eval_bundle=prev_eval_bundle_for_helper,
            current_version_num=current_version_id
        )
        generation_time_ms = (time.perf_counter() - start_generation_time) * 1000

        generated_text: Optional[str] = eval_bundle_improved.get("text") # type: ignore
        error_info_dict = eval_bundle_improved.get("error_info_dict")
        structured_error: Optional['StructuredErrorV49_T'] = None # type: ignore

        if error_info_dict:
            if self.StructuredErrorModel: # type: ignore
                structured_error = self.StructuredErrorModel.model_validate(error_info_dict) # type: ignore
            else:
                self.logger.error("StructuredErrorModel is not loaded.")
                structured_error = RuntimeError(str(error_info_dict)) # type: ignore
            self.logger.error(f"Part 13ヘルパー(_generate_improved_dialogue_content_and_eval)がエラーを返しました: {error_info_dict.get('code','N/A')}")
            version_state = self._create_version_state_from_bundle(current_version_id, eval_bundle_improved, generation_time_ms, None, feedback_text_for_prompt=eval_bundle_improved.get("feedback_text_used_for_prompt"))
            return generated_text, structured_error, version_state
        
        version_state = self._create_version_state_from_bundle(current_version_id, eval_bundle_improved, generation_time_ms, None, feedback_text_for_prompt=eval_bundle_improved.get("feedback_text_used_for_prompt"))
        
        self.logger.info(f"--- (Part 12) 改善ループ {improve_loop_num_1_based} (v{current_version_id}) オーケストレーション完了 ---")
        return generated_text, None, version_state
    # =============================================================================
    # ▲▲▲ Part 12: Core Loop Methods (v13.3 Update) ▲▲▲
    # =============================================================================

    # Existing methods from v13.2 (_check_loop_termination_condition, _select_final_version, _finalize_process,
    # _determine_next_intention, _save_loop_state, _re_evaluate_final_dfrs_if_needed,
    # _calculate_generation_params, _analyze_score_pattern, _calculate_adjustment_factors_v49, _clip_params)
    # would remain here, largely unchanged unless they need to adapt to how VersionStateV49 is populated
    # by the new _create_version_state_from_bundle.
    # _handle_loop_result and execute_generation_loops are significantly impacted and shown below.


    def _check_loop_termination_condition(self, version_state_data: 'VersionStateV49_Model_T', current_loop_idx_0_based: int) -> bool: # type: ignore
        # (v13.2 からのログ強化を維持)
        if not self.state: self.logger.error("_check_loop_termination_condition: GeneratorState None。安全のため終了。"); return True
        
        version_id_for_log = getattr(version_state_data, 'version_id', current_loop_idx_0_based + 1)
        vs_obj_id_for_log = id(version_state_data)
        min_score_terminate = getattr(self.settings, 'min_score_threshold', 4.5) # type: ignore
        min_loops_terminate = getattr(self.settings, 'min_feedback_loops', 1) # type: ignore

        self.logger.debug(f"Loop term check (v{version_id_for_log}, vs_obj_id:{vs_obj_id_for_log}, loop_idx {current_loop_idx_0_based}): min_score={min_score_terminate}, min_loops={min_loops_terminate}")
        
        llm_scores = getattr(version_state_data, 'llm_scores', None)
        llm_scores_id_for_log = id(llm_scores)
        llm_scores_type_for_log = type(llm_scores)
        llm_overall_for_log = getattr(llm_scores, 'overall', 'N/A') if llm_scores else 'N/A (llm_scores is None)'
        self.logger.debug(f"  _check_loop_termination_condition (v{version_id_for_log}): llm_scores (obj_id:{llm_scores_id_for_log}, type:{llm_scores_type_for_log}), Overall: {llm_overall_for_log}")

        if not (llm_scores and self.LLMKeys_cls and self.LLMEvalScoresModel and isinstance(llm_scores, self.LLMEvalScoresModel)): # type: ignore
            self.logger.warning(f"v{version_id_for_log}: LLMスコア無効/未ロード。スコアベース終了条件評価不可。")
            return (current_loop_idx_0_based + 1) >= getattr(self.settings, 'feedback_loops', 3) # type: ignore

        overall_key_value = getattr(self.LLMKeys_cls.OVERALL, 'value', 'overall') # type: ignore
        current_overall = getattr(llm_scores, overall_key_value, None)

        if not isinstance(current_overall, (int, float)):
            self.logger.warning(f"v{version_id_for_log}: LLM総合スコア数値でない({current_overall}, type {type(current_overall)})。スコアベース終了条件評価不可。")
            return False
        
        score_met = current_overall >= min_score_terminate
        loops_met = (current_loop_idx_0_based + 1) >= min_loops_terminate

        if score_met and loops_met:
            self.logger.info(f"ループ終了条件達成 (v{version_id_for_log})。")
            return True
        self.logger.debug(f"v{version_id_for_log}: スコア達成={score_met}({current_overall:.2f} vs {min_score_terminate:.2f}), ループ達成={loops_met}. 続行。")
        return False

    def _select_final_version(
        self, versions_list_to_select_from: List['VersionStateV49_Model_T'] # type: ignore
    ) -> Tuple[int, float]:
        # (v13.2 から変更なし - ログとエラーハンドリングは維持)
        self.logger.info(f"最終バージョン選択処理 ({len(versions_list_to_select_from)}候補)。")
        if not versions_list_to_select_from: self.logger.warning("評価対象バージョンリスト空。デフォルト返却。"); return 0, 0.0

        if not (self.FinalSelKeys_cls and self.LLMKeys_cls and self.DFRSMetrics_cls and self.LLMEvalScoresModel and self.DFRSSubScoresModel):
            self.logger.error("最終選択必須クラス未定義。フォールバック。")
            best_v = None; highest_s = -1.0
            if self.LLMKeys_cls and self.LLMEvalScoresModel:
                overall_key_enum = getattr(self.LLMKeys_cls, 'OVERALL', None)
                if overall_key_enum:
                    overall_key_val = getattr(overall_key_enum, 'value', 'overall')
                    for v_state in versions_list_to_select_from:
                        if v_state and not getattr(v_state, 'error_info', None) and getattr(v_state, 'llm_scores', None) and isinstance(getattr(v_state, 'llm_scores'), self.LLMEvalScoresModel): # type: ignore
                            score = getattr(v_state.llm_scores, overall_key_val, None)
                            if isinstance(score, (int, float)) and score > highest_s: highest_s = score; best_v = v_state
            if best_v: return getattr(best_v, 'version_id', 0), float(highest_s)
            return 0, 0.0

        llm_key_values = {m.value for m in self.LLMKeys_cls} # type: ignore
        dfrs_key_values = {m.value for m in self.DFRSMetrics_cls} # type: ignore
        weights = self.settings.final_selection_weights # type: ignore
        candidate_scores: List[Tuple[float, 'VersionStateV49_Model_T']] = [] # type: ignore

        for version_cand in versions_list_to_select_from:
            if not version_cand or getattr(version_cand, 'error_info', None): continue # type: ignore
            current_w_score = 0.0; current_total_w = 0.0
            self.logger.debug(f"v{getattr(version_cand, 'version_id', 'N/A')} 加重スコア計算...")
            for sel_key_enum, weight_val in weights.items():
                if not (isinstance(sel_key_enum, self.FinalSelKeys_cls) and isinstance(weight_val, (int, float)) and weight_val > 0): continue # type: ignore
                score_val: Optional[float] = None; actual_lookup_key = sel_key_enum.value

                is_llm = actual_lookup_key in llm_key_values
                is_dfrs = actual_lookup_key in dfrs_key_values
                
                if is_llm and getattr(version_cand, 'llm_scores', None) and isinstance(getattr(version_cand, 'llm_scores'), self.LLMEvalScoresModel): # type: ignore
                    llm_score_attr = getattr(version_cand.llm_scores, actual_lookup_key, None)
                    if isinstance(llm_score_attr, (int, float)): score_val = float(llm_score_attr)
                elif is_dfrs and getattr(version_cand, 'dfrs_scores', None) and isinstance(getattr(version_cand, 'dfrs_scores'), self.DFRSSubScoresModel) and isinstance(getattr(version_cand.dfrs_scores, 'scores', None), dict): # type: ignore
                    dfrs_score_attr = version_cand.dfrs_scores.scores.get(actual_lookup_key) # type: ignore
                    if isinstance(dfrs_score_attr, (int, float)): score_val = float(dfrs_score_attr)
                
                if score_val is not None:
                    current_w_score += score_val * weight_val
                    current_total_w += weight_val
                else: self.logger.debug(f"  v{getattr(version_cand,'version_id','N/A')}: スコアキー'{actual_lookup_key}'値なし。")
                
            final_score = (current_w_score / current_total_w) if current_total_w > 1e-6 else 0.0
            candidate_scores.append((final_score, version_cand))
            self.logger.debug(f"v{getattr(version_cand, 'version_id', 'N/A')}: 加重スコア={final_score:.3f}")

        if not candidate_scores: self.logger.warning("有効評価バージョンなし。デフォルト返却。"); return 0, 0.0
        candidate_scores.sort(key=lambda item: (item[0], -getattr(item[1], 'version_id', 0)), reverse=True)
        best_v_obj = candidate_scores[0][1]
        return getattr(best_v_obj, 'version_id', 0), float(candidate_scores[0][0])


    def _finalize_process(
        self, final_v_num_arg: int, final_weighted_s_arg: float, report_type_arg: str, last_err_obj_arg: Optional['StructuredErrorV49_T'] = None # type: ignore
    ) -> Optional['GeneratorStateV49_T']: # type: ignore
        # (v13.2 から変更なし - ログとエラーハンドリングは維持)
        self.logger.info(f"終了処理開始 (最終Ver:v{final_v_num_arg}, 選択時加重スコア:{final_weighted_s_arg:.3f})...")
        if not self.state: self.logger.error("終了処理: GeneratorState None。"); return None
        
        self.state.final_version = final_v_num_arg # type: ignore
        self.state.final_score_llm = final_weighted_s_arg # type: ignore
        self.state.completion_time = datetime.now(timezone.utc) # type: ignore
        self.state.complete = True # type: ignore

        if last_err_obj_arg and self.StructuredErrorModel and isinstance(last_err_obj_arg, self.StructuredErrorModel): # type: ignore
            if not self.state.last_error: # type: ignore
                self.state.last_error = last_err_obj_arg.to_dict() # type: ignore
            self.logger.warning(f"ループ中エラー({last_err_obj_arg.code})を最終エラーとして記録済。") # type: ignore
        elif last_err_obj_arg:
            self.logger.warning(f"最終エラーオブジェクト型不正({type(last_err_obj_arg)})。辞書変換試行。")
            try: self.state.last_error = dict(last_err_obj_arg) # type: ignore
            except: self.state.last_error = {"err_msg": str(last_err_obj_arg)} # type: ignore
        
        final_version_state: Optional['VersionStateV49_Model_T'] = None # type: ignore
        if self.state.versions and 0 < final_v_num_arg <= len(self.state.versions) and self.state.versions[final_v_num_arg - 1]: # type: ignore
             final_version_state = self.state.versions[final_v_num_arg - 1] # type: ignore
        
        if self.OutputEvaluationModel and final_version_state: # type: ignore
            try:
                eodf = None
                if final_version_state.dfrs_scores and self.DFRSMetrics_cls and hasattr(final_version_state.dfrs_scores,'scores') and isinstance(final_version_state.dfrs_scores.scores, dict): # type: ignore
                    eodf = final_version_state.dfrs_scores.scores.get(self.DFRSMetrics_cls.FINAL_EODF_V49.value) # type: ignore
                
                self.state.final_evaluation_summary = self.OutputEvaluationModel( # type: ignore
                    final_eodf_v49=eodf,
                    llm_scores=final_version_state.llm_scores,
                    dfrs_scores=final_version_state.dfrs_scores,
                    evaluation_feedback=final_version_state.evaluation_text_raw
                )
            except Exception as e: self.logger.error(f"最終評価サマリ作成エラー: {e}", exc_info=True)

        if self.GenerationStatsModel: # type: ignore
            try:
                duration = None
                if self.state.completion_time and self.state.start_time: # type: ignore
                    duration = (self.state.completion_time - self.state.start_time).total_seconds() # type: ignore
                
                self.state.final_generation_stats = self.GenerationStatsModel( # type: ignore
                    loops=self.state.current_loop + (1 if not self.state.complete else 0), # type: ignore # Should be current_loop for completed loops
                    overall=self.state.final_score_llm, # type: ignore
                    final_eodf_v49=getattr(self.state.final_evaluation_summary, 'final_eodf_v49', None) if self.state.final_evaluation_summary else None, # type: ignore
                    error_count=len(self.state.error_records or []), # type: ignore
                    duration_seconds=duration
                )
            except Exception as e: self.logger.error(f"最終生成統計作成エラー: {e}", exc_info=True)

        if self.dialogue_manager and hasattr(self.dialogue_manager, 'save_final_results'): # type: ignore
            try:
                self.dialogue_manager.save_final_results(self.state, report_type_arg) # type: ignore
            except Exception as e_save_final:
                if self.exception_manager: self.exception_manager.log_error(e_save_final, "SAVE.FINAL_RESULTS_CRITICAL") # type: ignore
        else: self.logger.error("DialogueManager未設定/save_final_resultsなし。最終結果保存不可。")

        self.logger.info(f"終了処理完了。ジョブ '{self.job_id}' は {'エラーあり' if self.state.last_error else '正常に'}完了しました。") # type: ignore
        return self.state

    def _determine_next_intention(self, version_state_data: 'VersionStateV49_Model_T') -> None: # type: ignore
        # (v13.2 からのログ強化を維持)
        if not self.state or not (hasattr(self.settings, 'adaptation_strategy_enabled') and self.settings.adaptation_strategy_enabled): # type: ignore
                 self.logger.debug(f"_determine_next_intention (v{getattr(version_state_data, 'version_id', 'N/A')}): GeneratorStateなし/適応戦略無効。スキップ。")
                 return

        analyzer_results = getattr(version_state_data, 'analyzer_results', None)
        last_inferred_phase = None; last_inferred_tone = None
        phase_alignment_score = None; tone_alignment_score = None
        analysis_summary_found_and_valid = False
        
        log_prefix_dni = f"_determine_next_intention (v{getattr(version_state_data, 'version_id', 'N/A')}, vs_obj_id:{id(version_state_data)})"
        
        analyzer_results_type_log = type(analyzer_results)
        analyzer_results_keys_log = list(analyzer_results.keys()) if isinstance(analyzer_results, dict) else 'N/A (Not a dict)'
        analyzer_results_id_log = id(analyzer_results)
        self.logger.debug(f"{log_prefix_dni}: analyzer_results (obj_id:{analyzer_results_id_log}): Type={analyzer_results_type_log}, Keys={analyzer_results_keys_log}")

        if isinstance(analyzer_results, dict) and analyzer_results: # Check if it's a non-empty dict
            summary_partial = analyzer_results.get("analysis_summary_partial")
            self.logger.debug(f"{log_prefix_dni}: analysis_summary_partial: Type={type(summary_partial)}, Value='{str(summary_partial)[:100]}...'")
            if isinstance(summary_partial, dict) and summary_partial: # Check if summary_partial is also a non-empty dict
                analysis_summary_found_and_valid = True
                phase_str = summary_partial.get("dominant_phase")
                if self.PsychologicalPhase_cls and phase_str: # type: ignore
                    try: last_inferred_phase = self.PsychologicalPhase_cls(phase_str) # type: ignore
                    except Exception as e_phase_conv: self.logger.warning(f"{log_prefix_dni}: Dominant phase '{phase_str}' from analyzer_results is not a valid PsychologicalPhaseV49 member. Error: {e_phase_conv}")
                elif not phase_str : self.logger.debug(f"{log_prefix_dni}: 'dominant_phase'キーがsummary_partialに存在しないか値が空です。")
                elif not self.PsychologicalPhase_cls : self.logger.error(f"{log_prefix_dni}: PsychologicalPhase_clsがロードされていません。")
                
                tone_str = summary_partial.get("dominant_tone")
                if self.EmotionalTone_cls and tone_str: # type: ignore
                    try: last_inferred_tone = self.EmotionalTone_cls(tone_str) # type: ignore
                    except Exception as e_tone_conv: self.logger.warning(f"{log_prefix_dni}: Dominant tone '{tone_str}' from analyzer_results is not a valid EmotionalToneV49 member. Error: {e_tone_conv}")
                elif not tone_str : self.logger.debug(f"{log_prefix_dni}: 'dominant_tone'キーがsummary_partialに存在しないか値が空です。")
                elif not self.EmotionalTone_cls : self.logger.error(f"{log_prefix_dni}: EmotionalTone_clsがロードされていません。")
                self.logger.debug(f"  {log_prefix_dni}: Extracted from summary_partial - Phase: {getattr(last_inferred_phase, 'value', 'N/A')}, Tone: {getattr(last_inferred_tone, 'value', 'N/A')}")
            else:
                self.logger.warning(f"{log_prefix_dni}: analyzer_results内に有効な'analysis_summary_partial' (非空の辞書)が見つかりません。Value: {summary_partial}")
        else: # analyzer_results is None or an empty dict
            self.logger.warning(f"{log_prefix_dni}: version_state_data.analyzer_resultsが無効(type: {analyzer_results_type_log})。")

        if not analysis_summary_found_and_valid:
             self.logger.warning(f"  {log_prefix_dni}: 有効な dominant_phase/tone が analyzer_results から取得できなかったため、適応戦略は現在の意図を維持するかデフォルトの挙動をします。")
        
        dfrs_scores_obj = getattr(version_state_data, 'dfrs_scores', None)
        if dfrs_scores_obj and self.DFRSMetrics_cls and hasattr(dfrs_scores_obj, 'scores') and isinstance(dfrs_scores_obj.scores, dict): # type: ignore
            pa_raw = dfrs_scores_obj.scores.get(self.DFRSMetrics_cls.PHASE_ALIGNMENT.value) # type: ignore
            if isinstance(pa_raw, (int,float)): phase_alignment_score = max(0.0, min(1.0, pa_raw/5.0))
            ta_raw = dfrs_scores_obj.scores.get(self.DFRSMetrics_cls.TONE_ALIGNMENT.value) # type: ignore
            if isinstance(ta_raw, (int,float)): tone_alignment_score = max(0.0, min(1.0, ta_raw/5.0))
            self.logger.debug(f"  {log_prefix_dni}: DFRS Alignment - Phase: {phase_alignment_score}, Tone: {tone_alignment_score}")

        next_phase, next_tone = self.dep.adaptation_strategy.suggest_next_state( # type: ignore
            current_intended_phase=self.state.current_intended_phase, # type: ignore
            current_intended_tone=self.state.current_intended_tone, # type: ignore
            last_inferred_phase=last_inferred_phase,
            last_inferred_tone=last_inferred_tone,
            last_alignment_scores=(phase_alignment_score, tone_alignment_score),
            current_generator_state=self.state # type: ignore
        )
        if next_phase is not None: self.state.current_intended_phase = next_phase # type: ignore
        if next_tone is not None: self.state.current_intended_tone = next_tone # type: ignore
        self.logger.info(f"次ループ目標意図更新: 位相={getattr(self.state.current_intended_phase,'value','N/A')}, トーン={getattr(self.state.current_intended_tone,'value','N/A')}") # type: ignore

    def _save_loop_state(self, current_version_num_for_save: int) -> None:
        # (v13.2 から変更なし)
        if not self.state: self.logger.error("..."); return
        self.logger.debug(f"ループ {current_version_num_for_save} 完了時の状態保存...")
        if self.dep.dialogue_manager and hasattr(self.dep.dialogue_manager, 'save_resume_state'): # type: ignore
            self.dep.exception_manager.safe_file_operation(name=f"ループ{current_version_num_for_save}状態保存", func=self.dep.dialogue_manager.save_resume_state, args=(self.state,)) # type: ignore
        else: self.logger.warning("DialogueManager未設定/save_resume_stateなし。状態保存スキップ。")

    # Methods for _calculate_generation_params, _analyze_score_pattern, _calculate_adjustment_factors_v49, _clip_params
    # are assumed to be the same as v13.2 and are included for completeness of Part 12.
    def _calculate_generation_params(self, version: int) -> Dict[str, Any]:
        if not self.state: return self.dep.api_client.generation_config_base.copy() # type: ignore

        base_params = self.dep.api_client.generation_config_base.copy() # type: ignore
        current_temp = base_params.get('temperature', 0.7)
        self.logger.info(f"v{version + 1} の生成パラメータ計算開始...")

        temp_config_cls = getattr(self, 'TemperatureStrategyConfigV49_cls', None)
        temp_config_data = getattr(self.config, 'temperature_config', None)
        temp_config_instance: Optional['TemperatureStrategyConfigV49'] = None # type: ignore

        if temp_config_cls and isinstance(temp_config_data, temp_config_cls): # type: ignore
            temp_config_instance = temp_config_data # type: ignore
        elif temp_config_cls and isinstance(temp_config_data, dict):
            try:
                temp_config_instance = temp_config_cls.model_validate(temp_config_data) # type: ignore
            except Exception as e:
                self.logger.warning(f"temperature_config (dict) の TemperatureStrategyConfigV49 検証失敗: {e}。デフォルト値使用。")
                temp_config_instance = temp_config_cls() # type: ignore
        elif not temp_config_cls:
             self.logger.error("TemperatureStrategyConfigV49_cls がロードされていません。")


        if temp_config_instance:
            strategy = temp_config_instance.strategy_type
            base_temp = temp_config_instance.base_temperature
            calculated_temp = base_temp

            if strategy == "fixed" and temp_config_instance.fixed_params:
                calculated_temp = temp_config_instance.fixed_params.temperature
            elif strategy == "decreasing" and temp_config_instance.decreasing_params:
                p = temp_config_instance.decreasing_params
                progress = version / max(1, self.settings.feedback_loops -1) if self.settings.feedback_loops > 1 else 0 # type: ignore
                decay_factor = (1 - p.decay_rate) ** version
                calculated_temp = p.final_temperature + (p.initial_temperature - p.final_temperature) * decay_factor
                calculated_temp = max(p.final_temperature, min(p.initial_temperature, calculated_temp))
            elif strategy == "two_dimensional" and temp_config_instance.two_dimensional_params:
                # ... (two_dimensional logic from v13.2) ...
                # This logic requires access to previous DFRS scores, which might not be available directly here
                # or needs careful handling if self.state.versions[-1] is not yet populated or evaluated for DFRS.
                # For simplicity in this focused update, we might default to base_temperature if complex dependencies arise here.
                # The existing v13.2 logic for 2D can be kept if state access is robust.
                calculated_temp = temp_config_instance.base_temperature # Simplified for now, full logic in original
                self.logger.info(f"  温度戦略 'two_dimensional' (簡略化適用/詳細はv13.2参照): {calculated_temp:.3f}")


            self.logger.info(f"  温度戦略 '{strategy}' 適用後温度: {calculated_temp:.3f}")
        else:
            self.logger.warning("  温度戦略設定なし/ロード失敗。API Clientのベース温度を使用。")
            calculated_temp = base_params.get('temperature', 0.7) # Fallback

        base_params['temperature'] = calculated_temp
        
        adj_factors = self._calculate_adjustment_factors_v49(version)
        if adj_factors:
            if self.state.adj_factor_history is None: self.state.adj_factor_history = [] # type: ignore
            self.state.adj_factor_history.append(adj_factors) # type: ignore
            base_params['temperature'] *= adj_factors.get('temperature_factor', 1.0)
            base_params['top_p'] = base_params.get('top_p', 0.95) * adj_factors.get('top_p_factor', 1.0)
            base_params['top_k'] = int(base_params.get('top_k', 40) * adj_factors.get('top_k_factor', 1.0))
            self.logger.info(f"  スコア履歴調整後温度: {base_params['temperature']:.3f} (理由: {adj_factors.get('reason', 'N/A')})")

        final_params = self._clip_params(base_params)
        if self.state.temperature_history is None: self.state.temperature_history = [] # type: ignore
        self.state.temperature_history.append(final_params['temperature']) # type: ignore
        
        self.logger.info(f"v{version + 1} 最終生成パラメータ: {final_params}")
        self._current_generation_params = final_params.copy()
        return final_params

    def _analyze_score_pattern(self, version_idx: int) -> Dict[str, Union[str, bool]]:
        # (v13.2 から変更なし)
        pattern_results = {"llm_trend": "stable", "dfrs_trend": "stable", "oscillation": False, "stagnation": False}
        if not self.state or not self.state.versions or version_idx < 2: # type: ignore
            self.logger.debug(f"_analyze_score_pattern (v{version_idx + 1}): スコア履歴3未満。傾向分析スキップ。")
            return pattern_results
        
        recent_versions = [v for v in self.state.versions[max(0, version_idx - 2) : version_idx + 1] if v is not None] # type: ignore
        if len(recent_versions) < 3: self.logger.debug(f"_analyze_score_pattern (v{version_idx + 1}): 有効VersionState3未満。傾向分析スキップ。"); return pattern_results

        llm_scores_overall = []
        for v in recent_versions:
            if v.llm_scores and hasattr(v.llm_scores, 'overall') and isinstance(v.llm_scores.overall, (int, float)): # type: ignore
                llm_scores_overall.append(v.llm_scores.overall) # type: ignore
            else:
                llm_scores_overall.append(None)

        dfrs_scores = []
        if self.DFRSMetrics_cls:
            final_eodf_key = getattr(self.DFRSMetrics_cls.FINAL_EODF_V49, 'value', 'final_eodf_v49') # type: ignore
            for v_state in recent_versions:
                if v_state.dfrs_scores and hasattr(v_state.dfrs_scores, 'scores') and isinstance(v_state.dfrs_scores.scores, dict):
                    score_val = v_state.dfrs_scores.scores.get(final_eodf_key)
                    if isinstance(score_val, (int, float)): dfrs_scores.append(score_val)
                    else: dfrs_scores.append(None)
                else: dfrs_scores.append(None)
        
        def get_trend(scores: List[Optional[float]]) -> str:
            valid_scores = [s for s in scores if isinstance(s, (float, int))]
            if len(valid_scores) < 2: return "stable"
            diffs = [valid_scores[i] - valid_scores[i-1] for i in range(1, len(valid_scores))]
            if not diffs: return "stable"
            if all(d > 0.01 for d in diffs): return "improving"
            if all(d < -0.01 for d in diffs): return "worsening"
            return "stable"

        pattern_results["llm_trend"] = get_trend(llm_scores_overall)
        pattern_results["dfrs_trend"] = get_trend(dfrs_scores)
        
        valid_llm_scores = [s for s in llm_scores_overall if s is not None]
        if len(valid_llm_scores) >= 3:
            if abs(valid_llm_scores[-1] - valid_llm_scores[-2]) < 0.05 and abs(valid_llm_scores[-2] - valid_llm_scores[-3]) < 0.05:
                pattern_results["stagnation"] = True
            if (valid_llm_scores[-1] > valid_llm_scores[-2] and valid_llm_scores[-2] < valid_llm_scores[-3]) or \
               (valid_llm_scores[-1] < valid_llm_scores[-2] and valid_llm_scores[-2] > valid_llm_scores[-3]):
                pattern_results["oscillation"] = True
        
        self.logger.debug(f"_analyze_score_pattern (v{version_idx + 1}) LLM Scores: {llm_scores_overall}, DFRS Scores: {dfrs_scores}, 結果: {pattern_results}")
        return pattern_results

    def _calculate_adjustment_factors_v49(self, version_idx: int) -> Dict[str, Any]:
        # (v13.2 から変更なし)
        base_factors = {'temperature_factor': 1.0, 'top_p_factor': 1.0, 'top_k_factor': 1.0,
                        'needs_diversity': False, 'needs_focus': False, 'reason': 'デフォルト調整なし'}
        if not self.state: return base_factors
        
        score_pattern = self._analyze_score_pattern(version_idx)
        reason_parts = []

        if score_pattern["stagnation"]:
            base_factors['temperature_factor'] *= 1.1
            base_factors['top_p_factor'] *= 0.98
            base_factors['needs_diversity'] = True
            reason_parts.append("停滞検出(温度↑,TopP↓)")
        elif score_pattern["llm_trend"] == "worsening":
            base_factors['temperature_factor'] *= 0.9
            base_factors['top_p_factor'] *= 1.02
            base_factors['needs_focus'] = True
            reason_parts.append("LLMスコア悪化(温度↓,TopP↑)")
        
        if score_pattern["oscillation"]:
            base_factors['temperature_factor'] *= 0.95
            reason_parts.append("スコア振動 -> 安定化" if "振動検出(温度↓)" not in reason_parts else "振動検出(温度↓)追加調整")
            if not reason_parts or reason_parts == ["デフォルト調整なし"]: reason_parts = ["スコア振動 -> 安定化"]
        
        if score_pattern["dfrs_trend"] == "worsening" and score_pattern["llm_trend"] != "worsening":
            base_factors['temperature_factor'] = max(base_factors['temperature_factor'], 1.05)
            if "DFRS悪化(温度↑)" not in reason_parts : reason_parts.append("DFRS悪化(温度↑)")
            if not reason_parts or reason_parts == ["デフォルト調整なし"]: reason_parts = ["DFRS悪化(温度↑)"]

        if not reason_parts or reason_parts == ["デフォルト調整なし"]: reason_parts = ["調整なし"]
        elif "デフォルト調整なし" in reason_parts and len(reason_parts) > 1:
            reason_parts.remove("デフォルト調整なし")

        base_factors['reason'] = ", ".join(reason_parts)
        
        self.logger.info(f"_calculate_adjustment_factors_v49 (v{version_idx + 1}): Pattern={score_pattern}, Factors={base_factors}")
        return base_factors

    def _clip_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # (v13.2 から変更なし)
        clipped = params.copy()
        if 'temperature' in clipped: clipped['temperature'] = round(max(0.0, min(2.0, clipped['temperature'])), 5)
        if 'top_p' in clipped: clipped['top_p'] = round(max(0.0, min(1.0, clipped['top_p'])), 5)
        if 'top_k' in clipped: clipped['top_k'] = max(1, int(clipped['top_k']))
        if 'candidate_count' in clipped : clipped['candidate_count'] = max(1, min(8, int(clipped['candidate_count'])))
        
        self.logger.debug(f"_clip_params: Input={params}, Clipped={clipped}")
        return clipped
    
    # =============================================================================
    # ▼▼▼ execute_generation_loops (Part 12 - v13.3 Update) ▼▼▼
    # =============================================================================
    def execute_generation_loops(
        self,
        character_a_data: Dict[str, Any],
        character_b_data: Dict[str, Any],
        scene_info_data: Dict[str, Any],
        target_length: int
    ) -> Tuple[Optional[str], Optional['GeneratorStateV49_T']]: # type: ignore
        self.logger.info(f"====== ジョブ '{self.job_id}' {getattr(self.config, 'SYSTEM_VERSION', 'N/A')} 対話生成ループ開始 ======") # type: ignore

        if not self._initialize_or_resume_state(character_a_data, character_b_data, scene_info_data, target_length):
            critical_err_msg = f"ジョブ '{self.job_id}' の状態初期化またはレジュームに失敗。処理中止。"
            self.logger.critical(critical_err_msg)
            last_err = None
            if self.exception_manager: last_err = self.exception_manager.log_error(RuntimeError(critical_err_msg), "ExecuteLoops.PreLoopInitFailure") # type: ignore
            return None, self._finalize_process(0, 0.0, getattr(self.settings, 'report_format', 'markdown'), last_err) # type: ignore

        if not self.state: self.logger.critical("致命的エラー: self.state が初期化されていません。"); return None, None
        if self.state.versions is None: self.state.versions = [] # type: ignore

        final_generated_text: Optional[str] = None
        last_loop_structured_error: Optional['StructuredErrorV49_T'] = None # type: ignore
        max_feedback_loops = self.settings.feedback_loops # type: ignore
        consecutive_errors_count = 0
        
        ErrorConfig_cls_local = getattr(self, 'ErrorConfigV49_cls', None)
        max_consecutive_allowed = 3 # Default
        if ErrorConfig_cls_local and isinstance(self.config.loaded_external_configs.get('error_config'), ErrorConfig_cls_local): # type: ignore
            max_consecutive_allowed = self.config.loaded_external_configs['error_config'].max_consecutive_errors # type: ignore
        elif isinstance(self.config.loaded_external_configs.get('error_config'), dict): # type: ignore
             max_consecutive_allowed = self.config.loaded_external_configs['error_config'].get('max_consecutive_errors', 3) # type: ignore
        self.logger.debug(f"連続エラー許容回数: {max_consecutive_allowed}")

        try:
            # --- 初回生成 (ループインデックス0 / バージョンID 1) ---
            self.state.current_loop = 0 # type: ignore
            self.logger.info(f"--- ループ {self.state.current_loop + 1}/{max_feedback_loops} (0-indexed: {self.state.current_loop}) 開始 ---") # type: ignore
            
            # v13.3: _generate_initial_dialogue (Part 12) now calls Part 13 helpers
            # and directly returns the VersionStateV49 object or error.
            generated_text_initial, error_initial_loop, processed_initial_vs_obj = self._generate_initial_dialogue()
            
            if processed_initial_vs_obj:
                if len(self.state.versions) > 0: # type: ignore
                    self.state.versions[0] = processed_initial_vs_obj # type: ignore
                else:
                    self.state.versions.append(processed_initial_vs_obj) # type: ignore
                self.logger.debug(f"execute_generation_loops: self.state.versions[0] (obj_id:{id(self.state.versions[0])}) 更新/設定完了。") # type: ignore
                self.logger.debug(f"  VS[0] analyzer_results (obj_id:{id(self.state.versions[0].analyzer_results)}) type: {type(self.state.versions[0].analyzer_results)}, Keys: {list(self.state.versions[0].analyzer_results.keys()) if self.state.versions[0].analyzer_results else 'N/A'}") # type: ignore
                self.logger.debug(f"  VS[0] llm_scores (obj_id:{id(self.state.versions[0].llm_scores)}) type: {type(self.state.versions[0].llm_scores)}, Overall: {getattr(self.state.versions[0].llm_scores, 'overall', 'N/A') if self.state.versions[0].llm_scores else 'None'}") # type: ignore
            elif error_initial_loop:
                 self.logger.error("初期対話生成で VersionState オブジェクトが返されず、エラーのみが返されました。")
            else: # Should not happen if _generate_initial_dialogue always returns one or the other
                 self.logger.error("初期対話生成後、_generate_initial_dialogueから有効なVersionStateもエラーも返されませんでした。")
                 if not error_initial_loop and self.exception_manager:
                     error_initial_loop = self.exception_manager.log_error(RuntimeError("Initial VersionState or error not returned"), "LoopIntegrity.NoReturnInitial") #type: ignore

            initial_version_state_to_handle: Optional['VersionStateV49_Model_T'] = self.state.versions[0] if self.state.versions and len(self.state.versions) > 0 else None # type: ignore
            
            consecutive_errors_count = self._handle_loop_result(initial_version_state_to_handle, error_initial_loop, 0, consecutive_errors_count)

            if error_initial_loop:
                self.logger.error(f"初期対話生成でエラー発生: {getattr(error_initial_loop, 'message', str(error_initial_loop))}")
                last_loop_structured_error = error_initial_loop
                if not generated_text_initial:
                    return None, self._finalize_process(0, 0.0, getattr(self.settings, 'report_format', 'markdown'), last_loop_structured_error) # type: ignore
            
            if generated_text_initial: final_generated_text = generated_text_initial

            if not initial_version_state_to_handle:
                return None, self._finalize_process(0, 0.0, getattr(self.settings, 'report_format', 'markdown'), last_loop_structured_error) # type: ignore

            self.logger.debug(f"execute_generation_loops: _check_loop_termination_condition / _determine_next_intention に渡す initial_version_state_to_handle (obj_id:{id(initial_version_state_to_handle)}).")
            if initial_version_state_to_handle.analyzer_results: # type: ignore
                self.logger.debug(f"  analyzer_results type: {type(initial_version_state_to_handle.analyzer_results)}, keys: {list(initial_version_state_to_handle.analyzer_results.keys())}") # type: ignore
            else: self.logger.debug(f"  analyzer_results is None or not a dict.")
            if initial_version_state_to_handle.llm_scores: # type: ignore
                self.logger.debug(f"  llm_scores type: {type(initial_version_state_to_handle.llm_scores)}, overall: {initial_version_state_to_handle.llm_scores.overall}") # type: ignore
            else: self.logger.debug(f"  llm_scores is None.")


            if self._check_loop_termination_condition(initial_version_state_to_handle, 0):
                self.logger.info(f"初期生成 (ループ 1) で終了条件を満たしました。")
            elif max_feedback_loops > 1 and not initial_version_state_to_handle.error_info:
                self._determine_next_intention(initial_version_state_to_handle)
                self._save_loop_state(1)
            else:
                self.logger.info("フィードバックループは実行されませんでした（最大ループ数設定または初期エラーのため）。")

            # --- 改善ループ ---
            if max_feedback_loops > 1 and not (initial_version_state_to_handle.error_info or self._check_loop_termination_condition(initial_version_state_to_handle, 0)):
                for improve_loop_num_1_based in range(1, max_feedback_loops):
                    current_loop_idx_0_based = improve_loop_num_1_based
                    self.state.current_loop = current_loop_idx_0_based # type: ignore
                    self.logger.info(f"--- ループ {current_loop_idx_0_based + 1}/{max_feedback_loops} (0-indexed: {current_loop_idx_0_based}) 開始 ---")
                    
                    generated_text_improve, error_improve_loop, processed_improve_vs_obj = self._execute_single_loop(improve_loop_num_1_based)
                    
                    if processed_improve_vs_obj:
                        if len(self.state.versions) > current_loop_idx_0_based: # type: ignore
                             self.state.versions[current_loop_idx_0_based] = processed_improve_vs_obj # type: ignore
                        else:
                             while len(self.state.versions) <= current_loop_idx_0_based: self.state.versions.append(None) #type: ignore
                             self.state.versions[current_loop_idx_0_based] = processed_improve_vs_obj # type: ignore
                        self.logger.debug(f"execute_generation_loops: self.state.versions[{current_loop_idx_0_based}] (obj_id:{id(self.state.versions[current_loop_idx_0_based])}) 更新完了。") #type: ignore
                        self.logger.debug(f"  VS[{current_loop_idx_0_based}] analyzer_results (obj_id:{id(self.state.versions[current_loop_idx_0_based].analyzer_results)}) type: {type(self.state.versions[current_loop_idx_0_based].analyzer_results)}, Keys: {list(self.state.versions[current_loop_idx_0_based].analyzer_results.keys()) if self.state.versions[current_loop_idx_0_based].analyzer_results else 'N/A'}") # type: ignore
                        self.logger.debug(f"  VS[{current_loop_idx_0_based}] llm_scores (obj_id:{id(self.state.versions[current_loop_idx_0_based].llm_scores)}) type: {type(self.state.versions[current_loop_idx_0_based].llm_scores)}, Overall: {getattr(self.state.versions[current_loop_idx_0_based].llm_scores, 'overall', 'N/A') if self.state.versions[current_loop_idx_0_based].llm_scores else 'None'}") # type: ignore

                    elif error_improve_loop:
                        self.logger.error(f"改善ループ v{current_loop_idx_0_based+1} で VersionState オブジェクトが返されず、エラーのみが返されました。")
                    else:
                        self.logger.error(f"改善ループ v{current_loop_idx_0_based+1} 後、_execute_single_loopから有効なVersionStateもエラーも返されませんでした。")
                        if not error_improve_loop and self.exception_manager:
                            error_improve_loop = self.exception_manager.log_error(RuntimeError(f"Improve VersionState or error (idx {current_loop_idx_0_based}) not returned"), "LoopIntegrity.NoReturnImprove") #type: ignore
                    
                    current_version_state_to_handle = self.state.versions[current_loop_idx_0_based] if self.state.versions and len(self.state.versions) > current_loop_idx_0_based else None # type: ignore
                    
                    consecutive_errors_count = self._handle_loop_result(current_version_state_to_handle, error_improve_loop, current_loop_idx_0_based, consecutive_errors_count)
                    
                    if error_improve_loop: last_loop_structured_error = error_improve_loop
                    if generated_text_improve: final_generated_text = generated_text_improve
                    
                    if not current_version_state_to_handle:
                         return None, self._finalize_process(0, 0.0, getattr(self.settings, 'report_format', 'markdown'), last_loop_structured_error) # type: ignore
                    
                    if current_version_state_to_handle.error_info and self.exception_manager and not self.exception_manager.is_retryable(current_version_state_to_handle.error_info.get('code', 'UNKNOWN')): # type: ignore
                        self.logger.critical(f"ループ {improve_loop_num_1_based+1}: 致命的エラー。処理中断。")
                        break
                    if consecutive_errors_count >= max_consecutive_allowed:
                        self.logger.error(f"連続エラー回数が上限({max_consecutive_allowed})に到達。処理中断。")
                        if not last_loop_structured_error and self.exception_manager: last_loop_structured_error = self.exception_manager.log_error(RuntimeError("Max consecutive errors"), "GENERATION.CONSECUTIVE_ERRORS") # type: ignore
                        break
                    
                    self.logger.debug(f"execute_generation_loops: _check_loop_termination_condition / _determine_next_intention に渡す current_version_state_to_handle (obj_id:{id(current_version_state_to_handle)}).")
                    if current_version_state_to_handle.analyzer_results: # type: ignore
                        self.logger.debug(f"  analyzer_results type: {type(current_version_state_to_handle.analyzer_results)}, keys: {list(current_version_state_to_handle.analyzer_results.keys())}") # type: ignore
                    else: self.logger.debug(f"  analyzer_results is None or not a dict.")
                    if current_version_state_to_handle.llm_scores: # type: ignore
                        self.logger.debug(f"  llm_scores type: {type(current_version_state_to_handle.llm_scores)}, overall: {current_version_state_to_handle.llm_scores.overall}") # type: ignore
                    else: self.logger.debug(f"  llm_scores is None.")

                    if self._check_loop_termination_condition(current_version_state_to_handle, current_loop_idx_0_based):
                        self.logger.info(f"改善ループ {improve_loop_num_1_based + 1} で終了条件を満たしました。")
                        break
                    
                    if improve_loop_num_1_based < max_feedback_loops -1 :
                        self._determine_next_intention(current_version_state_to_handle)
                        self._save_loop_state(current_loop_idx_0_based + 1)
                    else:
                        self.logger.info(f"最大フィードバックループ数 ({max_feedback_loops}) に到達しました（改善ループ）。")

        except Exception as e_main_loop_exec:
            self.logger.critical(f"メイン生成ループ (execute_generation_loops) 中に予期せぬ致命的エラー: {e_main_loop_exec}", exc_info=True)
            if self.dep.exception_manager: last_loop_structured_error = self.dep.exception_manager.log_error(e_main_loop_exec, "MainLoopCritical.Uncaught") # type: ignore

        selected_ver_id, final_score = self._select_final_version(self.state.versions or []) # type: ignore
        
        if selected_ver_id > 0:
            self._re_evaluate_final_dfrs_if_needed(selected_ver_id)
        else:
            self.logger.warning("_select_final_versionが0または無効なバージョンIDを返したため、最終DFRS再評価はスキップされます。")
        
        if self.state.versions and 0 < selected_ver_id <= len(self.state.versions) and self.state.versions[selected_ver_id - 1]: # type: ignore
            final_selected_version = self.state.versions[selected_ver_id - 1] # type: ignore
            if final_selected_version and not final_selected_version.error_info: # type: ignore
                final_generated_text = final_selected_version.generated_text # type: ignore
            elif final_generated_text is None and self.state.versions: # type: ignore
                for v_state_fb in reversed(self.state.versions): # type: ignore
                    if v_state_fb and not v_state_fb.error_info and v_state_fb.generated_text: # type: ignore
                        final_generated_text = v_state_fb.generated_text # type: ignore
                        self.logger.warning(f"最終選択v{selected_ver_id}エラー。フォールバックv{v_state_fb.version_id}テキスト使用。") # type: ignore
                        break
        
        finalized_state = self._finalize_process(selected_ver_id, final_score, getattr(self.settings, 'report_format', 'markdown'), last_loop_structured_error) # type: ignore

        if final_generated_text:
            self.logger.info(f"対話生成ジョブ '{self.job_id}' 完了。最終選択バージョン: v{selected_ver_id}, スコア: {final_score:.3f}")
        else:
            self.logger.error(f"対話生成ジョブ '{self.job_id}' は有効な出力を生成できませんでした。")
            
        return final_generated_text, finalized_state
    # =============================================================================
    # ▲▲▲ execute_generation_loops (Part 12 - v13.3 Update) ▲▲▲
    # =============================================================================

    # =============================================================================
    # ▼▼▼ _handle_loop_result メソッド (v13.3 Update) ▼▼▼
    # =============================================================================
    def _handle_loop_result(
        self,
        # v13.3: This method now expects a fully formed VersionStateV49 object,
        # or None if the calling loop method failed to produce one.
        processed_version_state_obj: Optional['VersionStateV49_Model_T'], # type: ignore
        loop_error_obj: Optional['StructuredErrorV49_T'], # type: ignore
        loop_idx_0_based: int, # This is the index in self.state.versions
        current_consecutive_errors: int
    ) -> int:
        if not self.state:
            self.logger.error("_handle_loop_result: GeneratorStateがNoneです。処理をスキップ。")
            return current_consecutive_errors + 1

        loop_version_id_for_log = loop_idx_0_based + 1 # 1-indexed version for logging
        error_occurred_in_loop = False
        error_info_to_record_on_state: Optional[Dict[str, Any]] = None # For self.state.error_records

        # Log the received VersionStateV49 object's critical attributes
        if processed_version_state_obj:
            vs_id = getattr(processed_version_state_obj, 'version_id', 'N/A')
            vs_obj_id = id(processed_version_state_obj)
            self.logger.debug(f"ループ {loop_version_id_for_log} 結果ハンドリング: VersionState(v{vs_id}, obj_id:{vs_obj_id}) 受領。")
            
            analyzer_res_attr = getattr(processed_version_state_obj, 'analyzer_results', "ATTR_NOT_FOUND")
            analyzer_keys = list(analyzer_res_attr.keys()) if isinstance(analyzer_res_attr, dict) else ('NOT_A_DICT' if analyzer_res_attr != "ATTR_NOT_FOUND" else "ATTR_NOT_FOUND")
            self.logger.debug(f"  _handle_loop_result (v{vs_id}): analyzer_results (obj_id:{id(analyzer_res_attr)}): Type={type(analyzer_res_attr)}, Keys={analyzer_keys}")
            
            llm_s_attr = getattr(processed_version_state_obj, 'llm_scores', "ATTR_NOT_FOUND")
            llm_overall = getattr(llm_s_attr, 'overall', 'N/A') if llm_s_attr != "ATTR_NOT_FOUND" and llm_s_attr is not None else ('N/A (llm_scores is ' + str(llm_s_attr) + ')')
            self.logger.debug(f"  _handle_loop_result (v{vs_id}): llm_scores (obj_id:{id(llm_s_attr)}): Type={type(llm_s_attr)}, Overall={llm_overall}")
        else:
            self.logger.warning(f"ループ {loop_version_id_for_log} 結果ハンドリング: processed_version_state_objがNoneです。")


        if loop_error_obj:
            error_occurred_in_loop = True
            error_info_to_record_on_state = loop_error_obj.to_dict() if hasattr(loop_error_obj, 'to_dict') else {"message": str(loop_error_obj)}
            self.logger.error(f"ループ {loop_version_id_for_log}: 処理中に外部エラー({error_info_to_record_on_state.get('code', 'N/A')}) - '{error_info_to_record_on_state.get('message', '詳細不明')}'")
        elif processed_version_state_obj and processed_version_state_obj.error_info:
            error_occurred_in_loop = True
            error_info_to_record_on_state = processed_version_state_obj.error_info
            self.logger.error(f"ループ {loop_version_id_for_log}: VersionState内にエラー情報あり({error_info_to_record_on_state.get('code', 'N/A')}) - '{error_info_to_record_on_state.get('message', '詳細不明')}'")
        elif not processed_version_state_obj: # VersionStateがNoneで、ループエラーもNoneの場合 (予期せぬ状況)
            error_occurred_in_loop = True
            err_msg_no_vs_no_err = f"ループ {loop_version_id_for_log}: processed_version_state_objがNoneだが、loop_error_objもNone。予期せぬ状態。"
            self.logger.error(err_msg_no_vs_no_err)
            if self.exception_manager:
                temp_err_obj = self.exception_manager.log_error(RuntimeError(err_msg_no_vs_no_err), "LoopIntegrity.MissingVSAndErrorInHandle") # type: ignore
                error_info_to_record_on_state = temp_err_obj.to_dict() if hasattr(temp_err_obj, 'to_dict') else {"message": str(temp_err_obj)}
            else:
                error_info_to_record_on_state = {"message": err_msg_no_vs_no_err, "code": "LoopIntegrity.MissingVSAndErrorInHandle"}

        if error_occurred_in_loop and error_info_to_record_on_state:
            if not isinstance(self.state.error_records, list): self.state.error_records = [] # type: ignore
            self.state.error_records.append(error_info_to_record_on_state) # type: ignore
            self.state.last_error = error_info_to_record_on_state # type: ignore
            # Ensure VersionState in list also reflects this error, if not already set by _create_version_state_from_bundle
            if processed_version_state_obj and not processed_version_state_obj.error_info:
                processed_version_state_obj.error_info = error_info_to_record_on_state # type: ignore
                processed_version_state_obj.status = "error" # type: ignore
            return current_consecutive_errors + 1
        else:
            # エラーなし: 呼び出し元で VersionState は self.state.versions に正しく格納されているはず。
            # ログは呼び出し元 (execute_generation_loops) で実施。
            self.logger.debug(f"ループ {loop_version_id_for_log}: 正常に完了。エラーなし。")
            return 0
    # =============================================================================
    # ▲▲▲ _handle_loop_result メソッド (v13.3 Update) ▲▲▲
    # =============================================================================

    # ... (Rest of the methods: _re_evaluate_final_dfrs_if_needed, etc. as in v13.2) ...
    # For brevity, _re_evaluate_final_dfrs_if_needed and other state-dependent helpers
    # from v13.2 are assumed to be present and correct.
    # The key focus here was the data flow into VersionStateV49 objects from loop execution.
    
    def _re_evaluate_final_dfrs_if_needed(self, final_version_id: int) -> None:
        # (v13.2 からのログ強化を維持)
        if not self.state or not self.state.versions or not (0 < final_version_id <= len(self.state.versions)) or not self.state.versions[final_version_id -1]: # type: ignore
            self.logger.warning(f"最終DFRS再評価: 選択バージョンID({final_version_id})が無効か、バージョンリスト/データ空。スキップ。")
            return

        final_version_data = self.state.versions[final_version_id - 1] # type: ignore
        if not (final_version_data and final_version_data.generated_text): # type: ignore
            self.logger.warning(f"最終DFRS再評価: v{final_version_id}テキストなし。スキップ。")
            return

        re_eval_needed = getattr(self.settings, 'dfrs_evaluate_all_loops', False) # type: ignore
        if not re_eval_needed and not getattr(final_version_data, 'dfrs_scores', None):
            self.logger.info(f"最終DFRS再評価: v{final_version_id}にDFRSスコアなし、かつ全ループDFRS評価無効。再評価実行。")
            re_eval_needed = True
        elif hasattr(final_version_data, 'dfrs_scores') and final_version_data.dfrs_scores and \
             hasattr(final_version_data.dfrs_scores, 'scores') and isinstance(final_version_data.dfrs_scores.scores, dict) and \
             "error" in final_version_data.dfrs_scores.scores:
            self.logger.info(f"最終DFRS再評価: v{final_version_id}のDFRSスコアにエラー({final_version_data.dfrs_scores.scores.get('error')})が含まれるため再評価。")
            re_eval_needed = True
        elif re_eval_needed:
            self.logger.info(f"最終DFRS再評価: 全ループDFRS評価が有効。v{final_version_id}DFRSスコアを強制再評価。")

        if re_eval_needed and self.dep.evaluator: # type: ignore
            self.logger.info(f"最終DFRSスコア再評価 (v{final_version_id})...")
            try:
                analyzer_res_for_final_dfrs = final_version_data.analyzer_results if isinstance(final_version_data.analyzer_results, dict) else {} # type: ignore

                dfrs_ok, dfrs_payload_final, dfrs_err = self.exception_manager.safe_nlp_processing( #type: ignore
                    f"最終DFRS再評価(v{final_version_id})", self.dep.evaluator.get_dfrs_scores_v49, #type: ignore
                    kwargs={'dialogue_text': final_version_data.generated_text, 'analyzer_results': analyzer_res_for_final_dfrs, # type: ignore
                            'intended_phase': getattr(final_version_data, 'intended_phase_at_generation', self.state.current_intended_phase),
                            'intended_tone': getattr(final_version_data, 'intended_tone_at_generation', self.state.current_intended_tone)}
                )
                if dfrs_ok and self.DFRSSubScoresModel and isinstance(dfrs_payload_final, dict): # type: ignore
                    final_version_data.dfrs_scores = self.DFRSSubScoresModel.model_validate(dfrs_payload_final) # type: ignore
                    self.logger.info(f"v{final_version_id}: 最終DFRSスコア再評価・格納完了。")
                elif not dfrs_ok:
                    self.logger.error(f"v{final_version_id}: 最終DFRS再評価失敗: {dfrs_err or 'Unknown DFRS error'}")
                    if self.DFRSSubScoresModel : # エラー情報をDFRSスコアオブジェクトに格納試行
                         final_version_data.dfrs_scores = self.DFRSSubScoresModel.model_validate({"scores": {"error": str(dfrs_err) if dfrs_err else "Unknown DFRS error in re_eval"}}) # type: ignore
            except Exception as e:
                self.logger.error(f"最終DFRS再評価中にエラー: {e}", exc_info=True)
        elif re_eval_needed:
            self.logger.warning("最終DFRS再評価が必要でしたが、Evaluatorが利用できません。")

# =============================================================================
# -- Part 12 終了点 (v13.3)
# =============================================================================
# =============================================================================
# -- Part 13: Generator Helper Methods (v4.9α - 修正・最適化・エラー対応版 v13.3)
# =============================================================================
# v4.9α: DialogueGeneratorV49のヘルパーメソッド群。
# v13.3 Update:
# - `_evaluate_and_score_candidate`:
#   - analyzer_results_dict を常に辞書型で返すように修正 (エラー時もエラー情報を含む辞書)。
#   - llm_scores_dict も同様に、抽出失敗時は空辞書またはエラー情報を含む辞書を返す。
#   - DFRSスコアはモデル検証用の生辞書(dfrs_scores_payload)を返す。
#   - 返り値の辞書構造を明確化し、全ての評価結果を網羅するように変更。
# - `_generate_initial_dialogue_content_and_eval` (旧 `_generate_initial_dialogue` from user's Part 13):
#   - 最終的に選択された最良の初期候補テキストに対して再度 `_evaluate_and_score_candidate` を呼び出し、
#     その完全な評価バンドル（辞書）を返すように修正。
# - `_generate_improved_dialogue_content_and_eval` (旧 `_generate_improved_dialogue` from user's Part 13):
#   - 生成された改善後テキストに対して `_evaluate_and_score_candidate` を呼び出し、
#     その完全な評価バンドル（辞書）を返すように修正。
# - ログ出力の強化: 主要なデータポイントやエラー発生時の情報ログを強化。
# - 既存ヘルパー (_create_error, _select_best_initial_candidate, _suggest_dialogue_mode) は維持・微調整。
# - 状態依存の強いメソッド (_analyze_score_pattern, _calculate_adjustment_factors_v49, _calculate_generation_params) は
#   Part 12 本体のメソッドとして管理されることを前提とする (このPartには含めない)。

from typing import (
    TYPE_CHECKING, TypeVar, Set, List, Dict, Optional, Tuple, Union, Any, Type, Literal, Callable, cast, TypeAlias
)
import enum
import logging
import random
import re
import math
import statistics
from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass, field as dataclass_field # Ensure field is imported for fallback
import pathlib
import copy
import time
import traceback

# --- グローバルスコープで利用可能であることを期待する変数 (Part 0 などで定義済み) ---
PYDANTIC_AVAILABLE = globals().get('PYDANTIC_AVAILABLE', False)
BaseModel: Type[Any] = globals().get('BaseModel', object) # type: ignore
ConfigDict: Type[Dict[str, Any]] = globals().get('ConfigDict', dict) # type: ignore
Field: Callable[..., Any] = globals().get('Field', lambda **kwargs: None) # type: ignore
ValidationError: Type[Exception] = globals().get('ValidationError', ValueError) # type: ignore
_get_global_type: Callable[[str, Optional[type]], Optional[Type[Any]]] = \
    globals().get('_get_global_type', lambda name, meta=None: globals().get(name))
sanitize_filename: Callable[[str, Optional[int], str], str] = \
    globals().get('sanitize_filename', lambda f, ml=None, r='_': str(f))
fmt: Callable[[Optional[Union[float, int]], int, str], str] = \
    globals().get('fmt', lambda v, p=2, na="N/A": str(v))

# --- ここから Part 13 のヘルパーメソッド群 ---
# (DialogueGeneratorV49 クラスのメソッドとして定義されることを想定)

class DialogueGeneratorV49: # Placeholder for context, assuming these are methods of this class

    # This __init__ is a simplified placeholder for the context of Part 13 methods.
    # The full DialogueGeneratorV49.__init__ is in Part 12.
    def __init__(self, job_id_base: str, dependencies: 'GeneratorDependenciesV49'): # type: ignore
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__qualname__}.Part13.v13.3")
        self.dep: 'GeneratorDependenciesV49' = dependencies # type: ignore
        self.config: 'ConfigProtocol' = self.dep.config # type: ignore
        self.settings: 'SettingsProtocol' = self.dep.settings # type: ignore
        self.state: Optional['GeneratorStateV49_T'] = None # type: ignore # For methods that might need it contextually

        # Load necessary model/enum classes (mirroring Part 12's __init__ for these)
        self.LLMKeys_cls: Optional[Type['ScoreKeysLLMEnumType']] = _get_global_type('ScoreKeys.LLM', enum.EnumMeta) # type: ignore
        self.DFRSMetrics_cls: Optional[Type['DFRSMetricsV49']] = _get_global_type('DFRSMetricsV49', enum.EnumMeta) # type: ignore
        self.InitialSelectionKeys_cls: Optional[Type['InitialSelectionKeysV49']] = _get_global_type('InitialSelectionKeysV49', enum.EnumMeta) # type: ignore
        self.PsychologicalPhase_cls: Optional[Type['PsychologicalPhaseV49']] = _get_global_type('PsychologicalPhaseV49', enum.EnumMeta) # type: ignore
        self.EmotionalTone_cls: Optional[Type['EmotionalToneV49']] = _get_global_type('EmotionalToneV49', enum.EnumMeta) # type: ignore
        self.StructuredErrorModel: Optional[Type['StructuredErrorV49_T']] = _get_global_type('StructuredErrorV49') # type: ignore
        self.LLMEvalScoresModel: Optional[Type['LLMEvaluationScoresV49']] = _get_global_type('LLMEvaluationScoresV49') # type: ignore
        self.DFRSSubScoresModel: Optional[Type['DFRSSubScoresV49']] = _get_global_type('DFRSSubScoresV49') # type: ignore
        self.FeedbackContextModel: Optional[Type['FeedbackContextV49']] = _get_global_type('FeedbackContextV49') # type: ignore
        self.dialogue_manager = self.dep.dialogue_manager
        self.exception_manager = self.dep.exception_manager
        self.api_client = self.dep.api_client
        self.analyzer = self.dep.analyzer
        self.evaluator = self.dep.evaluator
        self.prompt_builder = self.dep.prompt_builder


    def _create_error(self, code: str, message: Union[str, Exception], source: str, original_exception: Optional[Exception]=None, context_data: Optional[Dict[str, Any]] = None) -> 'StructuredErrorV49_T': # type: ignore
        # This is a helper method from Part 13 (user provided snippet implies its existence there)
        # For brevity, assuming StructuredErrorV49_cls is loaded in the main class __init__
        # and self.exception_manager is available.
        StructuredErrorV49_cls_create: Optional[Type['StructuredErrorV49_T']] = getattr(self, 'StructuredErrorModel', _get_global_type('StructuredErrorV49')) # type: ignore

        if StructuredErrorV49_cls_create:
            error_message_final = original_exception if original_exception and isinstance(original_exception, Exception) else message
            
            # Ensure message is a string for the model
            if not isinstance(error_message_final, str):
                error_message_final = str(error_message_final)

            err_instance = StructuredErrorV49_cls_create(
                message=error_message_final, # type: ignore
                code=code,
                source=source,
                context_data=context_data or {}, # Ensure context_data is a dict
                original_exception=original_exception,
                details=getattr(original_exception, 'details', None) if original_exception and hasattr(original_exception, 'details') else None
            )
            if self.exception_manager and hasattr(self.exception_manager, 'log_error'):
                self.exception_manager.log_error(err_instance) # type: ignore
            else:
                self.logger.error(f"[{source}:{code}] {message}" + (f" (Orig: {original_exception})" if original_exception else ""))
            return err_instance
        else:
            self.logger.critical(f"CRITICAL: StructuredErrorV49クラス未定義。 code={code}, msg={message}, src={source}")
            fallback_error = RuntimeError(f"Error: {code} at {source} - {str(message)}")
            if original_exception:
                raise fallback_error from original_exception
            raise fallback_error

    # --- Evaluation & Selection Helper Methods (v13.3 Update) ---
    def _evaluate_and_score_candidate(
        self, dialogue_text: str, candidate_label: str, # candidate_index removed, label is sufficient
        intended_phase_for_dfrs: Optional['PsychologicalPhaseV49'], # type: ignore
        intended_tone_for_dfrs: Optional['EmotionalToneV49'] # type: ignore
    ) -> Dict[str, Any]:
        """
        Evaluates a single dialogue candidate text.
        Returns a dictionary containing all evaluation artifacts.
        'analyzer_results_dict' will always be a dict, even on error.
        'llm_scores_dict' will be a dict of scores, or empty on error.
        """
        self.logger.debug(f"候補 '{candidate_label}' 評価・スコアリング開始 (テキスト長: {len(dialogue_text)})...")

        # Initialize the result dictionary with robust defaults
        current_eval_bundle: Dict[str, Any] = {
            "text": dialogue_text,
            "evaluation_raw": None,
            "llm_scores_dict": {}, # Default to empty dict
            "dfrs_scores_payload": None, # Raw dict for DFRSSubScoresV49 validation
            "analyzer_results_dict": {"status": "not_run", "error": "Analysis not attempted"}, # Default to error/status dict
            "error_info_dict": None # Overall error for this evaluation process
        }
        # Ensure critical models are loaded (should be handled by __init__, but good for helper context)
        if not (self.LLMKeys_cls and self.LLMEvalScoresModel and self.DFRSSubScoresModel and self.StructuredErrorModel and self.analyzer and self.evaluator and self.prompt_builder and self.api_client and self.exception_manager):
            msg_crit = "評価に必要なコアコンポーネントまたはモデルクラスがロードされていません。"
            self.logger.critical(msg_crit)
            current_eval_bundle["error_info_dict"] = self._create_error("EVAL.CORE_COMP_MISSING", msg_crit, f"EvalCandidate.{candidate_label}.PreCheck").to_dict()
            return current_eval_bundle

        if not dialogue_text or not dialogue_text.strip():
            msg = f"候補 '{candidate_label}' の評価対象テキストが空または空白のみです。"
            err_struct = self._create_error("EVAL.EMPTY_TEXT", msg, f"EvalCandidate.{candidate_label}.TextCheck")
            current_eval_bundle["error_info_dict"] = err_struct.to_dict()
            current_eval_bundle["analyzer_results_dict"] = {"status": "skipped_empty_text", "error": msg}
            self.logger.warning(msg)
            return current_eval_bundle

        # 1. LLM Evaluation (Scores)
        if self.settings.llm_evaluation_enabled: # type: ignore
            eval_prompt_text: Optional[str] = None
            try:
                eval_prompt_text = self.prompt_builder.create_evaluation_prompt(dialogue_text) # type: ignore
                if self.settings.save_prompts and self.dialogue_manager and candidate_label.startswith("v"): # type: ignore
                     self.exception_manager.safe_file_operation( # type: ignore
                         f"評価プロンプト保存({candidate_label})", self.dialogue_manager.save_prompt, # type: ignore
                         args=(f"evaluation_{candidate_label}_prompt", eval_prompt_text)
                     )
            except Exception as e_prompt:
                err_struct_prompt = self._create_error("EVAL.PROMPT_ERROR", f"評価プロンプト作成エラー: {e_prompt}", f"PromptBuilder.Eval.{candidate_label}", original_exception=e_prompt)
                current_eval_bundle["error_info_dict"] = err_struct_prompt.to_dict()
                current_eval_bundle["evaluation_raw"] = f"(評価プロンプト作成エラー: {e_prompt})"
                # Continue to other evals if possible

            if eval_prompt_text:
                eval_temperature = self.config.EVALUATION_TEMPERATURE # type: ignore
                api_call_ok, raw_eval_output, api_err_obj = self.exception_manager.safe_api_call( # type: ignore
                    f"対話評価API呼び出し({candidate_label})", self.api_client.generate_content, # type: ignore
                    kwargs={'prompt': eval_prompt_text, 'generation_config_overrides': {"temperature": eval_temperature}}
                )

                if not (api_call_ok and isinstance(raw_eval_output, str) and raw_eval_output.strip()):
                    eff_err = api_err_obj if api_err_obj else self._create_error("EVAL.API_FAILED_EMPTY_RESPONSE", "評価API呼び出し失敗または空応答", f"ApiClient.Eval.{candidate_label}")
                    if current_eval_bundle["error_info_dict"] is None: current_eval_bundle["error_info_dict"] = eff_err.to_dict()
                    current_eval_bundle["evaluation_raw"] = f"(評価APIエラー: {getattr(eff_err,'code','N/A')} - {str(eff_err)})"
                else:
                    current_eval_bundle["evaluation_raw"] = raw_eval_output
                    if self.settings.save_evaluations and self.dialogue_manager and candidate_label.startswith("v"): # type: ignore
                         self.exception_manager.safe_file_operation( # type: ignore
                             f"LLM評価テキスト保存({candidate_label})", self.dialogue_manager.save_evaluation, # type: ignore
                             args=(int(candidate_label[1:]), raw_eval_output) # Assuming vX format
                         )
                    try:
                        llm_scores_enum: Dict['ScoreKeysLLMEnumType', float] = self.prompt_builder.extract_scores(raw_eval_output, self.LLMKeys_cls) # type: ignore
                        current_eval_bundle["llm_scores_dict"] = {k.value: v for k, v in llm_scores_enum.items() if hasattr(k, 'value')}
                        if not current_eval_bundle["llm_scores_dict"]:
                            self.logger.warning(f"候補 '{candidate_label}': LLM評価テキストからスコア抽出できず。")
                            current_eval_bundle["llm_scores_dict"] = {"error": "No scores extracted"}
                        elif self.LLMKeys_cls.OVERALL.value not in current_eval_bundle["llm_scores_dict"]: # type: ignore
                            self.logger.warning(f"候補 '{candidate_label}': LLM総合スコア ('{self.LLMKeys_cls.OVERALL.value}') 未抽出。0.0で補完。 Extracted: {current_eval_bundle['llm_scores_dict']}") # type: ignore
                            current_eval_bundle["llm_scores_dict"][self.LLMKeys_cls.OVERALL.value] = 0.0 # type: ignore
                        else:
                            self.logger.debug(f"候補 '{candidate_label}' LLMスコア抽出成功: {current_eval_bundle['llm_scores_dict']}")
                    except Exception as e_score_ext:
                        err_struct_ext = self._create_error("EVAL.SCORE_EXTRACTION_FAILED", f"スコア抽出エラー: {e_score_ext}", f"extract_scores.{candidate_label}", original_exception=e_score_ext)
                        if current_eval_bundle["error_info_dict"] is None: current_eval_bundle["error_info_dict"] = err_struct_ext.to_dict()
                        current_eval_bundle["llm_scores_dict"] = {"error": f"Score extraction exception: {e_score_ext}"}
        else:
            self.logger.info(f"候補 '{candidate_label}': LLM評価は設定で無効化されています。")
            current_eval_bundle["evaluation_raw"] = "(LLM評価スキップ)"
            current_eval_bundle["llm_scores_dict"] = {"skipped": "LLM evaluation disabled"}


        # 2. Advanced Dialogue Analysis
        analyzer_results_dict_temp: Dict[str, Any] = {"status": "not_run_or_error"}
        if self.settings.advanced_nlp_enabled and self.analyzer: # type: ignore
            self.logger.debug(f"候補 '{candidate_label}': AdvancedDialogueAnalyzerV49 処理開始...")
            nlp_ok, nlp_data, nlp_err = self.exception_manager.safe_nlp_processing( # type: ignore
                f"NLP分析({candidate_label})", self.analyzer.analyze_and_get_results, args=(dialogue_text,) # type: ignore
            )
            if nlp_ok and isinstance(nlp_data, dict):
                analyzer_results_dict_temp = nlp_data
                self.logger.info(f"候補 '{candidate_label}': Analyzer処理完了。結果キー数: {len(analyzer_results_dict_temp)}")
            else:
                error_msg_analyzer = f"Analyzer処理失敗: {nlp_err or 'Unknown analyzer error'}"
                self.logger.error(f"候補 '{candidate_label}': {error_msg_analyzer}")
                analyzer_results_dict_temp = {"status": "error", "error": error_msg_analyzer, "details": str(nlp_err)}
                if current_eval_bundle["error_info_dict"] is None and nlp_err:
                    current_eval_bundle["error_info_dict"] = nlp_err.to_dict() if hasattr(nlp_err, 'to_dict') else self._create_error("NLP.ANALYSIS_FAILED_IN_EVAL", str(nlp_err), f"Analyzer.Eval.{candidate_label}").to_dict()
        else:
            self.logger.info(f"候補 '{candidate_label}': Advanced NLP分析は無効またはAnalyzer利用不可。分析スキップ。")
            analyzer_results_dict_temp = {"status": "skipped_disabled_or_unavailable"}
        current_eval_bundle["analyzer_results_dict"] = analyzer_results_dict_temp
        self.logger.debug(f"候補 '{candidate_label}': analyzer_results_dict (obj_id:{id(current_eval_bundle['analyzer_results_dict'])}) 設定。Type: {type(current_eval_bundle['analyzer_results_dict'])}, Keys: {list(current_eval_bundle['analyzer_results_dict'].keys()) if isinstance(current_eval_bundle['analyzer_results_dict'], dict) else 'N/A'}")


        # 3. DFRS Evaluation (uses analyzer_results if available)
        if self.settings.dfrs_evaluation_enabled and self.evaluator: # type: ignore
            self.logger.debug(f"候補 '{candidate_label}': DFRS評価処理開始...")
            # analyzer_results_dict_temp を DFRS評価に使用
            # intended_phase/tone はこのメソッドの引数から取得
            dfrs_ok, dfrs_data_payload, dfrs_err = self.exception_manager.safe_nlp_processing( # type: ignore
                f"DFRS評価({candidate_label})", self.evaluator.get_dfrs_scores_v49, # type: ignore
                kwargs={'dialogue_text': dialogue_text,
                        'analyzer_results': current_eval_bundle["analyzer_results_dict"], # Use the potentially error-containing dict
                        'intended_phase': intended_phase_for_dfrs,
                        'intended_tone': intended_tone_for_dfrs}
            )
            if dfrs_ok and isinstance(dfrs_data_payload, dict):
                current_eval_bundle["dfrs_scores_payload"] = dfrs_data_payload
                if self.DFRSMetrics_cls and isinstance(dfrs_data_payload.get("scores"), dict) :
                    eodf = dfrs_data_payload["scores"].get(self.DFRSMetrics_cls.FINAL_EODF_V49.value) # type: ignore
                    self.logger.debug(f"候補 '{candidate_label}' DFRS評価成功: eODF={fmt(eodf)}") # type: ignore
                else:
                    self.logger.debug(f"候補 '{candidate_label}' DFRS評価データ取得 (eODFキーなしまたはscoresなし): {dfrs_data_payload}")
            else:
                error_msg_dfrs = f"DFRS評価失敗: {dfrs_err or 'Unknown DFRS error'}"
                self.logger.error(f"候補 '{candidate_label}': {error_msg_dfrs}")
                # DFRSエラーは通常、致命的とはしないため、error_info_dict にはマージしないことが多い
                # 代わりに dfrs_scores_payload にエラー情報を格納
                current_eval_bundle["dfrs_scores_payload"] = {"error": str(dfrs_err) if dfrs_err else "Unknown DFRS error", "status": "error"}
        else:
            self.logger.info(f"候補 '{candidate_label}': DFRS評価スキップ (設定無効またはEvaluatorなし)。")
            current_eval_bundle["dfrs_scores_payload"] = {"status": "skipped_disabled_or_unavailable"}
        
        self.logger.debug(f"候補 '{candidate_label}' 評価・スコアリング完了。エラー状態: {current_eval_bundle.get('error_info_dict') is not None}")
        return current_eval_bundle

    def _select_best_initial_candidate(self, evaluated_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # (Part 13: User-provided snippet, minor adjustments for consistency)
        self.logger.info(f"{len(evaluated_candidates)}個の評価済み初期候補から最良のものを選択します...")
        if not self.LLMKeys_cls: # type: ignore
            self.logger.critical("ScoreKeys.LLM Enum未設定。最良候補選択不可。"); return None

        valid_candidates = []
        for c in evaluated_candidates:
            if isinstance(c, dict) and not c.get("error_info_dict"): # Check new error key
                scores_dict = c.get("llm_scores_dict")
                if isinstance(scores_dict, dict):
                    overall_score = scores_dict.get(self.LLMKeys_cls.OVERALL.value) # type: ignore
                    if isinstance(overall_score, (int, float)) and overall_score >= 0.0:
                        valid_candidates.append(c)
        
        if not valid_candidates:
            self.logger.error("有効な評価済み初期候補なし。初期生成失敗とみなします。")
            # 保存ロジックは呼び出し元で検討
            return None

        # InitialSelectionKeysV49 と DFRSMetricsV49 は __init__ でロードされることを期待
        if not (self.settings.dfrs_for_initial_selection and self.InitialSelectionKeys_cls and # type: ignore
                hasattr(self.settings, 'initial_candidate_weights') and self.DFRSMetrics_cls): # type: ignore
            # ... (フォールバックロジック: LLM総合スコアのみで選択) ... (v13.1と同様)
            self.logger.warning("LLM総合スコアのみで初期候補を選択 (DFRS初期選択無効または設定/Enum不足)。")
            valid_candidates.sort(key=lambda c: c.get("llm_scores_dict", {}).get(self.LLMKeys_cls.OVERALL.value, 0.0), reverse=True) # type: ignore
            best_fallback = valid_candidates[0] if valid_candidates else None
            if best_fallback: self.logger.info(f"最良初期候補(フォールバック): 候補{best_fallback.get('index','?')} (LLM Overall: {best_fallback.get('llm_scores_dict',{}).get(self.LLMKeys_cls.OVERALL.value,0.0):.2f})") # type: ignore
            return best_fallback

        weights_init: Dict['InitialSelectionKeysV49', float] = self.settings.initial_candidate_weights # type: ignore
        scored_cands_weighted: List[Tuple[float, Dict[str, Any]]] = []

        for cand_data in valid_candidates:
            weighted_score_val = 0.0; total_weight_val = 0.0
            llm_scores_cand = cand_data.get("llm_scores_dict", {})
            
            dfrs_payload = cand_data.get("dfrs_scores_payload")
            dfrs_scores_cand: Dict[str, Any] = {}
            if isinstance(dfrs_payload, dict) and "error" not in dfrs_payload and self.DFRSSubScoresModel:
                try:
                    # DFRSSubScoresModel.model_validate(dfrs_payload) はモデルインスタンスを返す。
                    # ここでは、その .scores 辞書が必要。
                    dfrs_model_inst = self.DFRSSubScoresModel.model_validate(dfrs_payload)
                    if dfrs_model_inst and hasattr(dfrs_model_inst, 'scores') and isinstance(dfrs_model_inst.scores, dict):
                        dfrs_scores_cand = dfrs_model_inst.scores
                except Exception as e_dfrs_val:
                    self.logger.warning(f"初期候補{cand_data.get('index','?')}のDFRSペイロード検証失敗: {e_dfrs_val}")
            
            for sel_key_enum, weight_enum_val in weights_init.items():
                if not isinstance(sel_key_enum, self.InitialSelectionKeys_cls): continue # type: ignore
                score_val_to_use: Optional[Union[int, float]] = None
                key_val_str_sel = sel_key_enum.value
                
                if isinstance(llm_scores_cand, dict) and key_val_str_sel in llm_scores_cand:
                    score_val_to_use = llm_scores_cand.get(key_val_str_sel)
                elif isinstance(dfrs_scores_cand, dict) and key_val_str_sel in dfrs_scores_cand: # Check validated DFRS scores dict
                    score_val_to_use = dfrs_scores_cand.get(key_val_str_sel)

                if isinstance(score_val_to_use, (int, float)):
                    weighted_score_val += score_val_to_use * weight_enum_val; total_weight_val += weight_enum_val
                else: self.logger.debug(f"初期候補{cand_data.get('index','?')}: スコアキー'{key_val_str_sel}'値不正/なし。")
            
            default_score_fallback = llm_scores_cand.get(self.LLMKeys_cls.OVERALL.value, 0.0) if isinstance(llm_scores_cand, dict) else 0.0 # type: ignore
            final_score_cand = (weighted_score_val / total_weight_val) if total_weight_val > 1e-6 else default_score_fallback
            scored_cands_weighted.append((final_score_cand, cand_data))
            self.logger.debug(f"  初期候補{cand_data.get('index','?')}: 加重スコア={final_score_cand:.3f} (総重み={total_weight_val:.2f})")

        if not scored_cands_weighted:
            self.logger.error("初期候補の加重スコア計算失敗。有効候補なし。フォールバック。");
            valid_candidates.sort(key=lambda c: c.get("llm_scores_dict", {}).get(self.LLMKeys_cls.OVERALL.value, 0.0), reverse=True) # type: ignore
            return valid_candidates[0] if valid_candidates else None

        scored_cands_weighted.sort(key=lambda item: (item[0], -item[1].get('index', float('inf'))), reverse=True)
        best_selected_cand = scored_cands_weighted[0][1]
        selected_overall_llm = best_selected_cand.get("llm_scores_dict", {}).get(self.LLMKeys_cls.OVERALL.value, 0.0) # type: ignore
        self.logger.info(f"最良初期候補選択(加重): 候補{best_selected_cand.get('index','?')} (加重スコア:{scored_cands_weighted[0][0]:.3f}, LLM総合:{selected_overall_llm:.2f})")

        # 却下された候補の保存ロジックはメインループ側に移譲した方が良い場合もある
        # ここではユーザー提供のコード構造を尊重
        if self.settings.save_rejected_candidates and self.dialogue_manager: # type: ignore
            for _, rejected_cand_data in scored_cands_weighted[1:]:
                reason = f"initial_selection_weighted_score_lower_than_{best_selected_cand.get('index','best')}"
                # self.dialogue_manager.save_rejected_candidate(rejected_cand_data, reason) # このメソッドが辞書を受け付けるか確認が必要
                self.logger.debug(f"却下された初期候補 {rejected_cand_data.get('index', '?')} ({reason}) の保存を試みます (実際の保存はDialogueManager次第)。")
        return best_selected_cand

    def _suggest_dialogue_mode(self, char_a: CharacterInputType, char_b: CharacterInputType, scene: SceneInfoInputType) -> Literal["normal", "delayed", "mixed", "auto"]:
        # (Part 13: User-provided snippet, assumed to be largely correct)
        atmosphere = str(scene.get("atmosphere", "")).lower()
        purpose = str(scene.get("purpose", "")).lower()
        if "緊張" in atmosphere or "対立" in purpose or "クライマックス" in purpose or "激しい" in atmosphere:
            self.logger.debug(f"対話モード推奨: 'normal' (Atmosphere='{atmosphere}', Purpose='{purpose}')")
            return "normal"
        if "内省" in purpose or "静か" in atmosphere or "回想" in purpose or "穏やか" in atmosphere:
            self.logger.debug(f"対話モード推奨: 'delayed' (Atmosphere='{atmosphere}', Purpose='{purpose}')")
            return "delayed"
        self.logger.debug(f"対話モード推奨: 'auto' (デフォルト)")
        return "auto"

    # --- Generation Helper Methods (v13.3 Update) ---
    def _generate_initial_dialogue_content_and_eval(
        self,
        char_a_input: CharacterInputType,
        char_b_input: CharacterInputType,
        scene_input: SceneInfoInputType,
        target_length_input: int
    ) -> Dict[str, Any]: # Returns a comprehensive evaluation bundle or an error dict
        """
        Generates initial dialogue candidates, selects the best, evaluates it thoroughly,
        and returns a dictionary containing all results and artifacts.
        """
        self.logger.info(f"初期対話コンテンツ生成および評価プロセス開始 (target_length:{target_length_input})...")
        
        if not self.state: # Should be initialized before calling this
            err_msg = "Generator state (self.state) is None. Cannot proceed with initial dialogue generation."
            self.logger.critical(err_msg)
            return {"error_info_dict": self._create_error("STATE.NONE_AT_INIT_HELPER", err_msg, "InitialDialogueHelper.PreCheck").to_dict()}

        initial_intended_phase = self.state.current_intended_phase
        initial_intended_tone = self.state.current_intended_tone

        try:
            prompt_text = self.prompt_builder.create_dialogue_prompt( # type: ignore
                char_a_input, char_b_input, scene_input, target_length_input, self.settings, # type: ignore
                initial_intended_phase, initial_intended_tone
            )
            if self.settings.save_prompts and self.dialogue_manager: # type: ignore
                 self.exception_manager.safe_file_operation("初期プロンプト保存", self.dialogue_manager.save_prompt, args=("initial_prompt", prompt_text)) # type: ignore
        except Exception as e_prompt:
            err_struct_prompt = self._create_error("PROMPT.INITIAL_CREATE_FAIL", f"初期プロンプト作成エラー: {e_prompt}", "InitialDialogueHelper.Prompt", original_exception=e_prompt)
            return {"error_info_dict": err_struct_prompt.to_dict()}

        num_candidates = self.config.INITIAL_CANDIDATE_COUNT # type: ignore
        # Note: _calculate_generation_params is part of Part 12, called by the main loop.
        # Here, we'd use _current_generation_params or a simplified version if this helper is more isolated.
        # For now, assume _current_generation_params is set appropriately by the caller or a default is used.
        generation_params_for_initial = getattr(self, '_current_generation_params', self.api_client.generation_config_base.copy()) # type: ignore
        if not generation_params_for_initial : generation_params_for_initial = self.api_client.generation_config_base.copy() # type: ignore

        api_call_ok, candidate_texts_list, api_err_obj = self.exception_manager.safe_api_call( # type: ignore
            "初期候補生成API", self.api_client.generate_content_with_candidates, # type: ignore
            kwargs={'prompt': prompt_text, 'candidate_count': num_candidates, 'generation_config_overrides': generation_params_for_initial}
        )

        if not (api_call_ok and candidate_texts_list and all(isinstance(t, str) and t.strip() for t in candidate_texts_list)):
            err_msg = f"初期候補生成API失敗: {api_err_obj or '有効候補なし'}"
            err_struct_cand = self._create_error("API.INITIAL_CANDIDATE_GEN_FAIL", err_msg, "InitialDialogueHelper.CandidateAPI", original_exception=api_err_obj if isinstance(api_err_obj, Exception) else RuntimeError(str(api_err_obj)))
            return {"error_info_dict": err_struct_cand.to_dict()}
        
        evaluated_candidates_list = [
            self._evaluate_and_score_candidate(txt, f"init_cand{i+1}", initial_intended_phase, initial_intended_tone)
            for i, txt in enumerate(candidate_texts_list)
        ]
        
        best_initial_candidate_eval_bundle = self._select_best_initial_candidate(evaluated_candidates_list)

        if not best_initial_candidate_eval_bundle or not best_initial_candidate_eval_bundle.get("text"):
            # If _select_best_initial_candidate returned None or a bundle without text
            err_no_best = self._create_error("GENERATION.NO_VALID_INITIAL_CANDIDATE", "有効な初期候補を選択できませんでした。", "InitialDialogueHelper.Selection")
            # Log rejected candidates if manager exists
            if self.settings.save_rejected_candidates and self.dialogue_manager: # type: ignore
                for rej_cand_bundle in evaluated_candidates_list:
                    if rej_cand_bundle != best_initial_candidate_eval_bundle: # Simple check
                        # self.dialogue_manager.save_rejected_candidate(rej_cand_bundle, "initial_candidate_not_best_or_error")
                        self.logger.debug(f"却下された初期候補 {rej_cand_bundle.get('index','?')} の保存を検討 (実際の保存はDialogueManager依存)")
            return {"error_info_dict": err_no_best.to_dict()}

        # The best_initial_candidate_eval_bundle already contains all evaluation artifacts.
        # No need to re-evaluate explicitly here if _select_best_initial_candidate works on full bundles.
        # However, the original Part 13 snippet implies re-evaluating the text of the best.
        # To align with that previous structure and ensure the "v1" label for final eval:
        
        best_text_v1 = best_initial_candidate_eval_bundle.get("text")
        if not isinstance(best_text_v1, str) or not best_text_v1.strip():
            err_empty_final_text = self._create_error("INTERNAL.EMPTY_BEST_INITIAL_TEXT_FINAL", "最良選択された初期候補のテキストが空です。", "InitialDialogueHelper.BestCandTextFinal")
            return {"error_info_dict": err_empty_final_text.to_dict()}

        self.logger.info(f"初期対話生成: 最良候補選択完了。最終評価 (v1) を実行します。テキスト長: {len(best_text_v1)}")
        final_v1_eval_bundle = self._evaluate_and_score_candidate(best_text_v1, "v1", initial_intended_phase, initial_intended_tone)
        
        # Log details of the final_v1_eval_bundle
        self.logger.debug(f"初期対話生成 (v1) 最終評価バンドル (obj_id:{id(final_v1_eval_bundle)}):")
        for key, val in final_v1_eval_bundle.items():
            if key == "analyzer_results_dict" and isinstance(val, dict):
                self.logger.debug(f"  {key} (obj_id:{id(val)}): Type={type(val)}, Keys={list(val.keys())}")
            elif key == "llm_scores_dict" and isinstance(val, dict):
                 self.logger.debug(f"  {key} (obj_id:{id(val)}): Type={type(val)}, Overall={val.get('overall', 'N/A')}")
            elif key == "dfrs_scores_payload" and self.DFRSSubScoresModel and isinstance(val, dict) and "error" not in val :
                try:
                    dfrs_model_temp = self.DFRSSubScoresModel.model_validate(val)
                    self.logger.debug(f"  {key} (obj_id:{id(val)}): Type={type(val)}, Validated eODF={getattr(dfrs_model_temp.scores, 'final_eodf_v49', 'N/A') if dfrs_model_temp.scores else 'N/A'}") # type: ignore
                except: self.logger.debug(f"  {key} (obj_id:{id(val)}): Type={type(val)}, Preview='{str(val)[:100]}...' (DFRS Model Val Fail)")
            else:
                self.logger.debug(f"  {key}: Type={type(val)}, Preview='{str(val)[:100]}...'")

        return final_v1_eval_bundle


    def _generate_improved_dialogue_content_and_eval(
        self,
        prev_dialogue_text: str,
        prev_version_eval_bundle: Dict[str, Any], # Expects a full bundle from _evaluate_and_score_candidate
        current_version_num: int # e.g., 2 for the first improvement
    ) -> Dict[str, Any]: # Returns a comprehensive evaluation bundle or an error dict
        """
        Generates an improved dialogue based on the previous version, evaluates it,
        and returns a dictionary containing all results.
        """
        self.logger.info(f"ループ {current_version_num} (改善): 改善対話生成および評価プロセス開始...")
        if not self.state:
            err_msg = "Generator state (self.state) is None. Cannot proceed with improved dialogue generation."
            self.logger.critical(err_msg)
            return {"error_info_dict": self._create_error("STATE.NONE_AT_IMPROVE_HELPER", err_msg, f"ImproveDialogueHelper.L{current_version_num}.PreCheck").to_dict()}
        
        if not (self.FeedbackContextModel and self.prompt_builder and self.api_client and self.exception_manager and self.settings): # type: ignore
            msg_crit = "改善対話生成に必要なコアコンポーネントがロードされていません。"
            self.logger.critical(msg_crit)
            return {"error_info_dict": self._create_error("IMPROVE.CORE_COMP_MISSING", msg_crit, f"ImproveDialogueHelper.L{current_version_num}.PreCheck").to_dict()}

        # Extract necessary info from prev_version_eval_bundle
        prev_llm_scores_dict = prev_version_eval_bundle.get("llm_scores_dict", {})
        prev_dfrs_payload = prev_version_eval_bundle.get("dfrs_scores_payload")
        prev_analyzer_results = prev_version_eval_bundle.get("analyzer_results_dict", {})
        prev_eval_raw = prev_version_eval_bundle.get("evaluation_raw", "")

        inferred_phase_for_fb: Optional['PsychologicalPhaseV49'] = None # type: ignore
        inferred_tone_for_fb: Optional['EmotionalToneV49'] = None # type: ignore
        if isinstance(prev_analyzer_results, dict):
            summary_partial = prev_analyzer_results.get("analysis_summary_partial", {})
            if isinstance(summary_partial, dict):
                if self.PsychologicalPhase_cls:
                    phase_str = summary_partial.get("dominant_phase")
                    if isinstance(phase_str, str): try: inferred_phase_for_fb = self.PsychologicalPhase_cls(phase_str) # type: ignore
                    except: pass
                if self.EmotionalTone_cls:
                    tone_str = summary_partial.get("dominant_tone")
                    if isinstance(tone_str, str): try: inferred_tone_for_fb = self.EmotionalTone_cls(tone_str) # type: ignore
                    except: pass
        
        # Ensure prev_dfrs_scores for feedback context is a plain dict from the model if available
        prev_dfrs_scores_for_ctx = {}
        if self.DFRSSubScoresModel and isinstance(prev_dfrs_payload, dict) and "error" not in prev_dfrs_payload:
            try:
                dfrs_model = self.DFRSSubScoresModel.model_validate(prev_dfrs_payload)
                if dfrs_model.scores: prev_dfrs_scores_for_ctx = dfrs_model.scores
            except Exception: pass # Keep it empty if validation fails

        feedback_context_data = {
            "version": current_version_num - 1,
            "dialogue_text": prev_dialogue_text,
            "intended_phase": self.state.current_intended_phase, # Current target for NEXT loop
            "intended_tone": self.state.current_intended_tone,   # Current target for NEXT loop
            "inferred_phase": inferred_phase_for_fb,
            "inferred_tone": inferred_tone_for_fb,
            "dfrs_scores": prev_dfrs_scores_for_ctx,
            "llm_scores": prev_llm_scores_dict
        }
        
        feedback_text_for_prompt: Optional[str] = "(フィードバック生成エラーまたはスキップ)"
        try:
            feedback_context_instance = self.FeedbackContextModel.model_validate(feedback_context_data) # type: ignore
            feedback_text_for_prompt = self.dep.feedback_manager.get_feedback(feedback_context_instance) # type: ignore
        except ValidationError as e_val_ctx: # type: ignore
            self.logger.error(f"FeedbackContext作成検証エラー(L{current_version_num}): {e_val_ctx.errors(include_url=False) if hasattr(e_val_ctx,'errors') else e_val_ctx}", exc_info=True) # type: ignore
            # Continue with default feedback text
        except Exception as e_other_ctx:
            self.logger.error(f"FeedbackContext作成またはフィードバック取得中エラー(L{current_version_num}): {e_other_ctx}", exc_info=True)
            # Continue

        try:
            improvement_prompt_text = self.prompt_builder.create_improvement_prompt( # type: ignore
                prev_dialogue_text,
                prev_eval_raw or "(前回の評価テキストなし)",
                feedback_context_instance if 'feedback_context_instance' in locals() else feedback_context_data, # Pass validated or raw
                self.settings, # type: ignore
                feedback_override=feedback_text_for_prompt
            )
            if self.settings.save_prompts and self.dialogue_manager: # type: ignore
                self.exception_manager.safe_file_operation(f"改善プロンプト保存(v{current_version_num})", self.dialogue_manager.save_prompt, args=(f"improvement_v{current_version_num}_prompt", improvement_prompt_text)) # type: ignore
        except Exception as e_impr_prompt:
            err_struct_impr_prompt = self._create_error("PROMPT.IMPROVE_CREATE_FAIL", f"改善プロンプト作成エラー: {e_impr_prompt}", f"L{current_version_num}.ImprovePrompt", original_exception=e_impr_prompt)
            return {"error_info_dict": err_struct_impr_prompt.to_dict()}

        # Note: _calculate_generation_params is part of Part 12.
        # Assume _current_generation_params is set by the main loop for this iteration.
        generation_params_for_improve = getattr(self, '_current_generation_params', self.api_client.generation_config_base.copy()) # type: ignore
        if not generation_params_for_improve: generation_params_for_improve = self.api_client.generation_config_base.copy() # type: ignore


        api_call_ok, improved_dialogue_text, api_err_obj = self.exception_manager.safe_api_call( # type: ignore
            f"改善対話生成(v{current_version_num})", self.api_client.generate_content, # type: ignore
            kwargs={'prompt': improvement_prompt_text, 'generation_config_overrides': generation_params_for_improve}
        )

        if not (api_call_ok and isinstance(improved_dialogue_text, str) and improved_dialogue_text.strip()):
            err_msg = f"改善対話生成API失敗(v{current_version_num}): {api_err_obj or '有効応答なし'}"
            err_struct_gen = self._create_error("API.IMPROVE_GEN_FAIL", err_msg, f"L{current_version_num}.ImproveGenAPI", original_exception=api_err_obj if isinstance(api_err_obj, Exception) else RuntimeError(str(api_err_obj)))
            return {"error_info_dict": err_struct_gen.to_dict(), "text": prev_dialogue_text} # Return prev_dialogue if improvement fails

        self.logger.info(f"ループ {current_version_num}: 改善対話生成成功。テキスト長: {len(improved_dialogue_text)}")
        
        # Now evaluate the newly generated improved_dialogue_text
        # The intended phase/tone for DFRS for *this* improved text are the current state's new targets
        # which should have been set by _determine_next_intention *before* this loop iteration started.
        current_intended_phase = self.state.current_intended_phase
        current_intended_tone = self.state.current_intended_tone

        self.logger.info(f"改善対話 (v{current_version_num}) の評価を開始します。意図位相: {getattr(current_intended_phase, 'value', 'N/A')}, 意図トーン: {getattr(current_intended_tone, 'value', 'N/A')}")
        final_eval_bundle_for_improved = self._evaluate_and_score_candidate(
            improved_dialogue_text.strip(), f"v{current_version_num}",
            current_intended_phase, current_intended_tone
        )
        
        # Add feedback_text to the bundle (it's not part of _evaluate_and_score_candidate)
        final_eval_bundle_for_improved["feedback_text_used_for_prompt"] = feedback_text_for_prompt
        
        self.logger.debug(f"改善対話 (v{current_version_num}) 最終評価バンドル (obj_id:{id(final_eval_bundle_for_improved)}):")
        for key, val in final_eval_bundle_for_improved.items():
            if key == "analyzer_results_dict" and isinstance(val, dict):
                self.logger.debug(f"  {key} (obj_id:{id(val)}): Type={type(val)}, Keys={list(val.keys())}")
            elif key == "llm_scores_dict" and isinstance(val, dict):
                 self.logger.debug(f"  {key} (obj_id:{id(val)}): Type={type(val)}, Overall={val.get('overall', 'N/A')}")
            # Add other relevant attribute logging if needed
            else:
                self.logger.debug(f"  {key}: Type={type(val)}, Preview='{str(val)[:100]}...'")

        return final_eval_bundle_for_improved

# =============================================================================
# -- Part 13 終了点
# =============================================================================
# =============================================================================
# -- Part 14: Generator Main Loop (v4.9α - 修正・最適化版)
# =============================================================================
    def execute_generation_loops(
        self,
        character_a_input: Dict[str, Any], # CharacterInputType
        character_b_input: Dict[str, Any], # CharacterInputType
        scene_info_input: Dict[str, Any],  # SceneInfoInputType
        target_length: int = 4000,
        report_type: str = "light"
    ) -> Optional['GeneratorStateV49']: # StateType
        """
        対話生成のメインループを実行します。状態の初期化/レジューム、ループ処理、最終結果の処理を行います。
        """
        self.logger.info(f"====== ジョブ '{self.job_id}' {getattr(self.config, 'SYSTEM_VERSION', 'N/A')} 対話生成ループ開始 ======")
        last_structured_error_obj: Optional['StructuredErrorV49'] = None # type: ignore

        try:
            # --- 1. 状態の初期化またはレジューム ---
            effective_target_length = getattr(self.settings, 'target_length', target_length)
            if target_length != effective_target_length and getattr(self.settings, 'target_length', None) is not None:
                self.logger.info(f"目標文字数を引数値({target_length})から設定値({effective_target_length})に更新。")

            if not self._initialize_or_resume_state(character_a_input, character_b_input, scene_info_input, effective_target_length):
                critical_error_msg = f"ジョブ '{self.job_id}' の状態初期化またはレジュームに失敗しました。処理を中止します。"
                self.logger.critical(critical_error_msg)
                # self.state が None の場合でも、エラーを記録して finalize を試みる
                error_for_finalize = getattr(self.state, 'last_error', None) if self.state and hasattr(self.state, 'last_error') else None
                if not error_for_finalize: # last_error が設定されていなければ、ここで作成
                    error_for_finalize = self._create_error("STATE.INIT_OR_RESUME_FAILED", critical_error_msg, "ExecuteLoops.PreLoopInit")
                return self._finalize_process(0, 0.0, report_type, error_for_finalize)


            if not self.state:
                critical_state_none_msg = "状態初期化/レジューム後、self.state が None です。これは予期せぬ状態です。処理を中止します。"
                self.logger.critical(critical_state_none_msg)
                err = self._create_error("INTERNAL.STATE_NONE_AFTER_INIT", critical_state_none_msg, "ExecuteLoops.PreLoop")
                return self._finalize_process(0, 0.0, report_type, err)

            if self.state.complete:
                self.logger.info(f"ジョブ '{self.job_id}' は既に完了済みです。既存の状態を返します。")
                return self.state

            # --- 2. 設定値取得 (ループ制御用) ---
            max_loops: int = getattr(self.settings, 'feedback_loops', 3)
            # min_loops_for_term_check は _check_loop_termination_condition で直接参照

            exec_config_root = getattr(self.config, 'loaded_external_configs', None)
            error_config_model: Optional['ErrorConfigV49'] = getattr(exec_config_root, 'error_config', None) if exec_config_root else None # type: ignore
            
            max_consecutive_err_val: int
            if error_config_model and hasattr(error_config_model, 'max_consecutive_errors'):
                max_consecutive_err_val = getattr(error_config_model, 'max_consecutive_errors', 3)
            else:
                # フォールバック: AppConfigV49 のクラス変数から取得
                AppConfigV49_cls_ref = globals().get('AppConfigV49')
                max_consecutive_err_val = getattr(AppConfigV49_cls_ref, 'MAX_CONSECUTIVE_ERRORS_DEFAULT', 3) if AppConfigV49_cls_ref else 3
            self.logger.debug(f"連続エラー許容回数: {max_consecutive_err_val}")
            consecutive_errors_count = 0


            # --- 3. メイン生成ループ ---
            progress_bar = None
            is_tqdm_available_flag = globals().get('TQDM_AVAILABLE', False)
            if is_tqdm_available_flag:
                 tqdm_module_local = globals().get('tqdm')
                 if callable(tqdm_module_local):
                    progress_bar = tqdm_module_local(total=max_loops, initial=self.state.current_loop, desc=f"Job {self.job_id} Loop", unit="loop", dynamic_ncols=True, leave=True)

            while self.state.current_loop < max_loops:
                current_loop_idx = self.state.current_loop # 0-indexed
                current_version_num_display = current_loop_idx + 1 # 1-indexed for display and versioning

                if progress_bar:
                     progress_bar.set_description(f"Job {self.job_id} Loop {current_version_num_display}/{max_loops}")
                self.logger.info(f"--- ループ {current_version_num_display}/{max_loops} (0-indexed: {current_loop_idx}) 開始 ---")

                version_data_from_loop = self._execute_single_loop(current_version_num_display)

                loop_error_details_dict = version_data_from_loop.get("error")
                current_loop_error_object: Optional['StructuredErrorV49'] = None # type: ignore

                if isinstance(loop_error_details_dict, dict):
                    self.logger.warning(f"ループ {current_version_num_display} でエラー発生: {loop_error_details_dict.get('error_code', 'N/A')}")
                    StructuredErrorV49_cls_local: Optional[Type['StructuredErrorV49']] = _get_global_type('StructuredErrorV49') # type: ignore
                    if StructuredErrorV49_cls_local:
                        try:
                            # model_validate は Pydantic モデル用なので、StructuredErrorV49 では使えない
                            # コンストラクタに辞書を渡して再構築する (辞書のキーとコンストラクタ引数が一致している前提)
                            current_loop_error_object = StructuredErrorV49_cls_local(
                                message=loop_error_details_dict.get('error_message', 'Loop error dictionary'),
                                code=loop_error_details_dict.get('error_code', 'UNKNOWN.LOOP_ERROR_RECONSTRUCT_FAIL'),
                                source=loop_error_details_dict.get('error_source', f"Loop{current_version_num_display}.ErrorReconstruct"),
                                details=loop_error_details_dict.get('error_details'),
                                context_data=loop_error_details_dict.get('context_data_summary'), # to_dictの出力に合わせる
                                root_cause=loop_error_details_dict.get('root_cause'),
                                is_retryable=loop_error_details_dict.get('is_retryable'),
                                # original_exception は辞書からは直接復元できないので None のまま
                            )
                        except Exception as e_restore:
                            self.logger.error(f"ループエラー辞書のStructuredErrorV49への再構成失敗: {e_restore}, 元データ: {loop_error_details_dict}")
                            current_loop_error_object = self._create_error( # エラーオブジェクト作成ヘルパーを使用
                                loop_error_details_dict.get('error_code', 'UNKNOWN.LOOP_ERROR_RESTORE_FAIL_HELPER'),
                                loop_error_details_dict.get('error_message', 'Loop error, details unavailable for helper'),
                                loop_error_details_dict.get('error_source', f"Loop{current_version_num_display}.ErrorRestoreFailHelper")
                            )
                    else:
                        self.logger.error("StructuredErrorV49クラス未定義。ループエラーを汎用例外として扱います。")
                        current_loop_error_object = RuntimeError(str(loop_error_details_dict)) # type: ignore

                consecutive_errors_count = self._handle_loop_result(version_data_from_loop, current_loop_error_object, current_version_num_display, consecutive_errors_count)

                if not self.state: # _handle_loop_result 内で self.state が None になる可能性も考慮
                    critical_loop_state_none_msg = "ループ結果処理中に self.state が None になりました。致命的エラーとして処理を中断します。"
                    self.logger.critical(critical_loop_state_none_msg)
                    last_structured_error_obj = self._create_error("INTERNAL.STATE_CORRUPTED_IN_LOOP", critical_loop_state_none_msg, "ExecuteLoops.HandleResult")
                    break # ループを抜けて終了処理へ

                latest_version_state = self.state.versions[-1] if self.state.versions else None
                current_loop_had_significant_error = bool(latest_version_state and latest_version_state.error_info)

                if current_loop_had_significant_error:
                    effective_error_for_retry_check = current_loop_error_object or \
                                                    (self._create_error(
                                                        latest_version_state.error_info.get('code', 'UNKNOWN.FROM_STATE_ERROR') if latest_version_state.error_info else 'UNKNOWN.MISSING_ERROR_INFO',
                                                        latest_version_state.error_info.get('message', 'Error info from state') if latest_version_state.error_info else 'Error info missing',
                                                        latest_version_state.error_info.get('source', f'Loop{current_version_num_display}.StateError') if latest_version_state.error_info else f'Loop{current_version_num_display}.StateErrorMissing'
                                                    ) if latest_version_state and latest_version_state.error_info else None)

                    if effective_error_for_retry_check and not self.exception_manager.is_retryable(effective_error_for_retry_check):
                        self.logger.critical(f"ループ {current_version_num_display}: 致命的(リトライ不能)エラー ({getattr(effective_error_for_retry_check, 'code', 'N/A')})。処理を中断します。")
                        last_structured_error_obj = effective_error_for_retry_check
                        break
                    if consecutive_errors_count >= max_consecutive_err_val:
                        self.logger.error(f"連続エラー回数が上限 ({max_consecutive_err_val}) に達しました。処理を中断します。")
                        last_structured_error_obj = effective_error_for_retry_check if effective_error_for_retry_check else \
                                             self._create_error("GENERATION.CONSECUTIVE_ERRORS", f"連続エラー上限({max_consecutive_err_val})到達", f"Loop{current_version_num_display}")
                        break
                else: # エラーなし
                    consecutive_errors_count = 0 # エラーがなければリセット
                    self._determine_next_intention(version_data_from_loop)
                    if not self.state: # _determine_next_intention で self.state が None になる可能性を考慮
                        critical_intention_state_none_msg = "_determine_next_intention 処理中に self.state が None になりました。中止します。"
                        self.logger.critical(critical_intention_state_none_msg)
                        last_structured_error_obj = self._create_error("INTERNAL.STATE_CORRUPTED_POST_INTENTION", critical_intention_state_none_msg, "ExecuteLoops.DetermineIntention")
                        break

                    if self._check_loop_termination_condition(version_data_from_loop, current_loop_idx):
                        self.logger.info(f"ループ終了条件に合致したため、ループ {current_version_num_display} で終了します。")
                        break

                self._save_loop_state(current_version_num_display)
                if progress_bar:
                    progress_bar.update(1)

            if progress_bar:
                progress_bar.close()

            if not self.state:
                critical_post_loop_state_none_msg = "メイン生成ループ終了後、self.state が None です。これは予期せぬ状態です。"
                self.logger.critical(critical_post_loop_state_none_msg)
                final_error_to_pass = last_structured_error_obj or self._create_error("INTERNAL.STATE_NONE_POST_LOOP", critical_post_loop_state_none_msg, "ExecuteLoops.PostLoop")
                return self._finalize_process(0, 0.0, report_type, final_error_to_pass)

            self.logger.info(f"メイン生成ループ終了 (実行ループ数: {self.state.current_loop+1 if not self.state.complete else self.state.current_loop})。最終結果の選択と処理に進みます。")

            # --- 4. 最終結果の選択と終了処理 ---
            final_version_num, final_weighted_score = self._select_final_version(self.state.versions)
            self._re_evaluate_final_dfrs_if_needed(final_version_num)

            final_selected_version_obj: Optional['VersionStateV49'] = None # type: ignore
            if 0 < final_version_num <= len(self.state.versions):
                final_selected_version_obj = self.state.versions[final_version_num - 1]
            
            final_llm_overall_for_finalize = 0.0
            if final_selected_version_obj and final_selected_version_obj.llm_scores and self.LLMKeys_cls:
                # ScoreKeys.LLM.OVERALL.value が LLMEvaluationScoresV49 の "overall" 属性と一致することを期待
                llm_overall_attr_name = self.LLMKeys_cls.OVERALL.value # type: ignore
                llm_overall_val = getattr(final_selected_version_obj.llm_scores, llm_overall_attr_name, None)
                if isinstance(llm_overall_val, (int, float)):
                    final_llm_overall_for_finalize = float(llm_overall_val)
                else:
                    self.logger.warning(f"最終バージョンのLLM総合スコア('{llm_overall_attr_name}')が取得できませんでした (値: {llm_overall_val})。")

            return self._finalize_process(final_version_num, final_llm_overall_for_finalize, report_type, last_structured_error_obj)

        except Exception as e_main_loop_uncaught:
            self.logger.critical(f"メインループ処理 (execute_generation_loops) 中に予期せぬ致命的エラーが発生しました: {e_main_loop_uncaught}", exc_info=True)
            struct_error_uncaught: Optional['StructuredErrorV49'] = None # type: ignore
            if self.exception_manager:
                struct_error_uncaught = self.exception_manager.log_error(e_main_loop_uncaught, "MainLoopCritical.Uncaught") # type: ignore
            else:
                StructuredErrorV49_cls: Optional[Type['StructuredErrorV49']] = _get_global_type('StructuredErrorV49') # type: ignore
                if StructuredErrorV49_cls:
                    struct_error_uncaught = StructuredErrorV49_cls(str(e_main_loop_uncaught), "MAIN.UNCAUGHT_EXCEPTION", "ExecuteGenerationLoops.MainTry", original_exception=e_main_loop_uncaught)
                else:
                    self.logger.error("StructuredErrorV49クラスも未定義のため、エラーオブジェクトを作成できません。")
                    if self.state:
                         self.state.last_error = {"code": "MAIN.UNCAUGHT_FATAL_NO_STRUCT_ERR", "message": str(e_main_loop_uncaught)}
                         return self.state
                    return None

            if self.state: # self.state が存在する場合のみアクセス
                if not self.state.last_error:
                    self.state.last_error = struct_error_uncaught.to_dict() if hasattr(struct_error_uncaught, 'to_dict') else str(struct_error_uncaught)
                self.state.complete = False # エラー発生時は未完了扱いにする
                if self.dialogue_manager:
                    self.logger.info("致命的エラー発生のため、現在の状態でレジュームファイルを保存します。")
                    self.exception_manager.safe_file_operation("緊急レジューム保存", self.dialogue_manager.save_resume_state, args=(self.state,)) # type: ignore
            return self._finalize_process(0, 0.0, report_type, struct_error_uncaught)
        finally:
            if hasattr(self.settings, 'adaptation_strategy_enabled') and self.settings.adaptation_strategy_enabled and \
               hasattr(self.settings, 'log_phase_tone_transitions') and self.settings.log_phase_tone_transitions:
                if self.dep.adaptation_strategy and hasattr(self.dep.adaptation_strategy, 'save_history'):
                    try:
                        self.logger.info("適応戦略履歴の最終保存を試みます...")
                        self.dep.adaptation_strategy.save_history()
                    except Exception as e_hist_save_final:
                        self.logger.error(f"適応戦略履歴の最終保存に失敗しました: {e_hist_save_final}")
            self.logger.info(f"====== ジョブ '{self.job_id}' 対話生成ループ終了 ======")


    def _execute_single_loop(self, version: int) -> Dict[str, Any]:
        """
        単一生成ループ実行。常に辞書を返す。エラー時は辞書内の "error" にエラー情報を格納。
        """
        loop_start_time = time.time()
        version_data: Dict[str, Any] = {
            "version": version, "dialogue": None, "evaluation": None, "scores": {},
            "dfrs_scores": None, "analyzer_results": None,
            "generation_time_ms": None, "evaluation_time_ms": None, "error": None
        }

        if not self.state:
            # _create_error は StructuredErrorV49 インスタンスを返す
            err_obj = self._create_error("INTERNAL.STATE_NONE_IN_LOOP", "Single loop execution with no state.", f"Loop{version}.PreCheck")
            version_data["error"] = err_obj.to_dict() # to_dict() で辞書に変換
            return version_data

        prev_dialogue: Optional[str] = None
        prev_eval_text_or_error: Union[str, Dict[str, Any], None] = None # エラー辞書も許容
        prev_dfrs_scores_dict: Optional[Dict[str, Optional[float]]] = None
        llm_scores_history_list: List[Optional[Dict[str, float]]] = []
        dfrs_scores_history_list: List[Optional[Dict[str, Optional[float]]]] = []

        previous_intended_phase = self.state.current_intended_phase
        previous_intended_tone = self.state.current_intended_tone
        previous_inferred_phase: Optional['PsychologicalPhaseV49'] = None # type: ignore
        previous_inferred_tone: Optional['EmotionalToneV49'] = None # type: ignore

        if version > 1: # 改善ループの場合
            if len(self.state.versions) >= version - 1: # 正しくは version - 1 が前のバージョン
                try:
                    # 履歴の取得 (Pydanticモデルの .model_dump() を活用)
                    VersionStateV49_Model_cls = _get_global_type('VersionStateV49')
                    for v_state_hist_entry in self.state.versions:
                        if VersionStateV49_Model_cls and isinstance(v_state_hist_entry, VersionStateV49_Model_cls):
                            llm_hist_entry = v_state_hist_entry.llm_scores.model_dump(mode='json', exclude_none=True, by_alias=True) if v_state_hist_entry.llm_scores else {}
                            llm_scores_history_list.append(llm_hist_entry) # type: ignore
                            dfrs_hist_entry = v_state_hist_entry.dfrs_scores.scores if v_state_hist_entry.dfrs_scores and v_state_hist_entry.dfrs_scores.scores else {}
                            dfrs_scores_history_list.append(dfrs_hist_entry) # type: ignore
                        else:
                            self.logger.warning(f"VersionState履歴エントリの型が不正 (v{v_state_hist_entry.version_id if hasattr(v_state_hist_entry, 'version_id') else 'N/A'})。スキップ。")
                            llm_scores_history_list.append(None) # プレースホルダー
                            dfrs_scores_history_list.append(None)

                    previous_version_state_index = version - 2 # 0-indexed
                    if 0 <= previous_version_state_index < len(self.state.versions):
                        previous_version_state = self.state.versions[previous_version_state_index]
                        prev_dialogue = previous_version_state.generated_text

                        # 前回の評価テキストまたはエラー情報を取得
                        if previous_version_state.error_info and isinstance(previous_version_state.error_info, dict):
                            prev_eval_text_or_error = previous_version_state.error_info
                        elif previous_version_state.evaluation_text_raw: # Pydantic V2では属性アクセスが推奨
                            prev_eval_text_or_error = previous_version_state.evaluation_text_raw
                        elif previous_version_state.llm_scores and previous_version_state.llm_scores.raw_output:
                            prev_eval_text_or_error = previous_version_state.llm_scores.raw_output
                        else:
                            prev_eval_text_or_error = "(評価テキスト欠落または前バージョンでエラー)"
                        self.logger.debug(f"Loop {version}: Prev eval type: {type(prev_eval_text_or_error)}, content preview: {str(prev_eval_text_or_error)[:100]}")


                        prev_dfrs_scores_dict = dfrs_scores_history_list[previous_version_state_index] if dfrs_scores_history_list and previous_version_state_index < len(dfrs_scores_history_list) else None

                        if isinstance(previous_version_state.analyzer_results, dict):
                            summary = previous_version_state.analyzer_results.get("analysis_summary_partial")
                            if isinstance(summary, dict):
                                Phase_cls_loop: Optional[Type['PsychologicalPhaseV49']] = _get_global_type('PsychologicalPhaseV49', enum.EnumMeta) # type: ignore
                                Tone_cls_loop: Optional[Type['EmotionalToneV49']] = _get_global_type('EmotionalToneV49', enum.EnumMeta) # type: ignore
                                if Phase_cls_loop:
                                    phase_val_str = summary.get("dominant_phase")
                                    if isinstance(phase_val_str, str):
                                        try: previous_inferred_phase = Phase_cls_loop(phase_val_str) # type: ignore
                                        except ValueError: self.logger.warning(f"v{version}改善用: 前ループの推論 dominant_phase '{phase_val_str}' 不正。")
                                    elif isinstance(phase_val_str, Phase_cls_loop): previous_inferred_phase = phase_val_str # type: ignore
                                if Tone_cls_loop:
                                    tone_val_str = summary.get("dominant_tone")
                                    if isinstance(tone_val_str, str):
                                        try: previous_inferred_tone = Tone_cls_loop(tone_val_str) # type: ignore
                                        except ValueError: self.logger.warning(f"v{version}改善用: 前ループの推論 dominant_tone '{tone_val_str}' 不正。")
                                    elif isinstance(tone_val_str, Tone_cls_loop): previous_inferred_tone = tone_val_str # type: ignore
                    else:
                         raise IndexError(f"Requested previous version index {previous_version_state_index} out of bounds for versions list (len: {len(self.state.versions)})")

                except Exception as e_prev_data:
                    # _create_error は StructuredErrorV49 インスタンスを返す
                    err_struct = self._create_error(
                        "INTERNAL.PREVIOUS_DATA_ERROR", # より具体的なエラーコード
                        f"v{version}改善に必要な前バージョンデータの取得/解釈エラー: {e_prev_data}",
                        f"Loop{version}.PrevDataFetchError",
                        original_exception=e_prev_data
                    )
                    version_data["error"] = err_struct.to_dict() # to_dict() で辞書に変換
                    return version_data
            else:
                err_struct = self._create_error(
                    "INTERNAL.INSUFFICIENT_HISTORY_FOR_IMPROVEMENT", # より具体的なエラーコード
                    f"v{version}改善に必要な前バージョンデータが不足 (versions len: {len(self.state.versions)})",
                    f"Loop{version}.PrevDataMissing"
                )
                version_data["error"] = err_struct.to_dict()
                return version_data

        generation_start_time = time.time()
        generated_text_or_error_obj: Union[str, Dict[str, Any], 'StructuredErrorV49'] # type: ignore

        if version == 1: # 初期生成
            if not self.state.input_data:
                err = self._create_error("INPUT_DATA.MISSING_FOR_INITIAL_GENERATION", "初期生成に必要な入力データ(self.state.input_data)がありません。", f"Loop{version}.InitialGenInputError")
                version_data["error"] = err.to_dict()
                return version_data
            try:
                # _generate_initial_dialogue はエラー時にエラー辞書を返す可能性がある
                gen_output_tuple = self._generate_initial_dialogue(
                    self.state.input_data.characterA.model_dump(by_alias=True, exclude_none=True),
                    self.state.input_data.characterB.model_dump(by_alias=True, exclude_none=True),
                    self.state.input_data.sceneInfo.model_dump(by_alias=True, exclude_none=True),
                    self.state.target_length
                )
                generated_text_or_error_obj = gen_output_tuple[0] # str またはエラー辞書
                version_data["evaluation"] = gen_output_tuple[1]    # str またはエラー辞書
                version_data["scores"] = gen_output_tuple[2]        # Dict[str, float]
                version_data["dfrs_scores"] = gen_output_tuple[3]   # Optional[Dict[str, Optional[float]]]
                # analyzer_results は _evaluate_and_score_candidate で設定されるのでここでは直接設定しない

                if isinstance(generated_text_or_error_obj, str):
                    version_data["dialogue"] = generated_text_or_error_obj
                elif isinstance(generated_text_or_error_obj, dict): # エラー辞書が返された場合
                    version_data["error"] = generated_text_or_error_obj
                    return version_data # エラーがあるのでここで終了
                else: # 予期せぬ型
                    err_unexpected = self._create_error("INTERNAL.UNEXPECTED_INIT_GEN_RESULT_TYPE", f"初期生成結果が予期せぬ型: {type(generated_text_or_error_obj)}", f"Loop{version}.InitialGenResultType")
                    version_data["error"] = err_unexpected.to_dict()
                    return version_data
            except Exception as e_init_gen_call:
                self.logger.error(f"初期対話生成(_generate_initial_dialogue)呼び出し中エラー: {e_init_gen_call}", exc_info=True)
                err_struct = self.exception_manager.log_error(e_init_gen_call, f"Loop{version}.InitialGenCallUnhandled") # type: ignore
                version_data["error"] = err_struct.to_dict()
                return version_data
        else: # version > 1 (改善ループ)
            if prev_dialogue is None or prev_eval_text_or_error is None:
                err = self._create_error("INTERNAL.MISSING_DATA_FOR_IMPROVEMENT", "改善生成に必要な前バージョンデータ(prev_dialogueまたはprev_eval_text_or_error)が不足。", f"Loop{version}.ImproveGenPrepError")
                version_data["error"] = err.to_dict()
                return version_data

            # _generate_improved_dialogue は str または StructuredErrorV49 インスタンスを返す
            generated_text_or_error_obj = self._generate_improved_dialogue(
                prev_dialogue, prev_eval_text_or_error, version,
                llm_scores_history_list, dfrs_scores_history_list,
                previous_intended_phase, previous_intended_tone,
                previous_inferred_phase, previous_inferred_tone,
                prev_dfrs_scores_dict
            )
            if isinstance(generated_text_or_error_obj, _get_global_type('StructuredErrorV49')): # type: ignore
                version_data["error"] = generated_text_or_error_obj.to_dict()
                return version_data
            elif isinstance(generated_text_or_error_obj, str) and generated_text_or_error_obj.strip():
                version_data["dialogue"] = generated_text_or_error_obj
            else: # 空文字列や予期せぬ型
                err_invalid_improve = self._create_error("INTERNAL.INVALID_IMPROVEMENT_RESULT", f"改善対話生成結果が無効/空 (型: {type(generated_text_or_error_obj)})。", f"Loop{version}.ImproveGenResult")
                version_data["error"] = err_invalid_improve.to_dict()
                return version_data

        version_data["generation_time_ms"] = (time.time() - generation_start_time) * 1000

        # --- 生成結果の評価 (エラーがなければ) ---
        if version_data.get("error") is None and isinstance(version_data.get("dialogue"), str):
            dialogue_text_for_eval = version_data["dialogue"]
            evaluation_start_time = time.time()

            if version > 1: # 改善ループの場合のみここで再評価
                evaluation_data_dict = self._evaluate_and_score_candidate(
                    dialogue_text=dialogue_text_for_eval,
                    candidate_index=0,
                    version=version
                )
                # version_data に評価結果をマージ (textは上書きしない)
                for key_from_eval, val_from_eval in evaluation_data_dict.items():
                    if key_from_eval != "text":
                        version_data[key_from_eval] = val_from_eval
            elif version == 1:
                self.logger.debug(f"ループ {version}: 初期生成(_generate_initial_dialogue)での評価結果を流用。")
                # version_data["analyzer_results"] が _generate_initial_dialogue -> _evaluate_and_score_candidate で
                # 既に設定されていることを確認。もしなければ警告。
                if version_data.get("analyzer_results") is None:
                    self.logger.warning(f"ループ {version}: version_data に analyzer_results が見つかりません。Analyzer結果が欠落している可能性があります。")


            version_data["evaluation_time_ms"] = (time.time() - evaluation_start_time) * 1000

        loop_duration_seconds = (time.time() - loop_start_time)
        self.logger.info(f"--- ループ {version} 処理完了 ({loop_duration_seconds:.2f}秒) ---")

        if self.dialogue_manager and version_data.get("error") is None:
            dialogue_to_save = version_data.get("dialogue")
            if isinstance(dialogue_to_save, str):
                self.exception_manager.safe_file_operation(f"対話保存(v{version})", self.dialogue_manager.save_dialogue, args=(version, dialogue_to_save)) # type: ignore
            
            evaluation_to_save = version_data.get("evaluation") # これはLLMの生評価テキスト
            if version > 0 and isinstance(evaluation_to_save, str): # バージョン0は初期候補なので評価テキストを毎回保存しない
                self.exception_manager.safe_file_operation(f"LLM評価テキスト保存(v{version})", self.dialogue_manager.save_evaluation, args=(version, evaluation_to_save)) # type: ignore

        return version_data

    def _handle_loop_result(
        self,
        version_data: Dict[str, Any],
        loop_error_obj: Optional['StructuredErrorV49'], # StructuredErrorV49 インスタンスまたは None
        version: int,
        consecutive_errors: int
    ) -> int:
        if not self.state:
            self.logger.error("_handle_loop_result: Generator state is None. Cannot process loop result.")
            return consecutive_errors + 1 # エラーとしてカウント

        current_loop_had_error = bool(loop_error_obj or version_data.get("error"))
        error_dict_to_store: Optional[Dict[str, Any]] = None

        if current_loop_had_error:
            if loop_error_obj and hasattr(loop_error_obj, 'to_dict'):
                error_dict_to_store = loop_error_obj.to_dict()
            elif isinstance(version_data.get("error"), dict):
                error_dict_to_store = version_data["error"]
            else:
                # _create_error は StructuredErrorV49 インスタンスを返す
                fallback_err_obj = self._create_error("UNKNOWN.LOOP_ERROR_UNSPECIFIED_IN_HANDLE", f"Loop {version} unspecified error during handling", f"Loop{version}.ResultHandleFallback")
                error_dict_to_store = fallback_err_obj.to_dict()
            
            self.logger.warning(f"ループ {version} でエラー処理: {error_dict_to_store.get('error_code', 'N/A')}")
            if 'loop_number' not in error_dict_to_store: # ループ番号がなければ追加
                 error_dict_to_store['loop_number'] = version

            self.state.error_records.append(error_dict_to_store)
            if not self.state.last_error: # 最初の致命的エラーを記録
                self.state.last_error = error_dict_to_store
            
            new_consecutive_errors = consecutive_errors + 1
        else: # エラーなし
            error_dict_to_store = None # エラーがないことを明示
            new_consecutive_errors = 0 # エラーカウントをリセット

        # VersionStateV49 オブジェクトの作成と self.state.versions への追加
        VersionState_cls: Optional[Type['VersionStateV49_Model']] = _get_global_type('VersionStateV49') # type: ignore
        LLMEvalScoreModel_cls: Optional[Type['LLMEvaluationScoresV49']] = _get_global_type('LLMEvaluationScoresV49') # type: ignore
        DFRSSubScoreModel_cls: Optional[Type['DFRSSubScoresV49']] = _get_global_type('DFRSSubScoresV49') # type: ignore

        if not (VersionState_cls and LLMEvalScoreModel_cls and DFRSSubScoreModel_cls):
            missing_models_str = ", ".join([name for cls, name in [(VersionState_cls, "VS"), (LLMEvalScoreModel_cls, "LLMScores"), (DFRSSubScoreModel_cls, "DFRSScores")] if not cls])
            error_msg_models = f"ループ結果記録に必要なモデル({missing_models_str})未定義。"
            self.logger.error(error_msg_models)
            # このエラー自体も記録
            err_obj_state_record = self._create_error("INTERNAL.MISSING_MODEL_FOR_STATE_RECORD", error_msg_models, f"HandleLoopResult.V{version}.ModelsMissing")
            if not self.state.last_error: self.state.last_error = err_obj_state_record.to_dict()
            self.state.error_records.append(err_obj_state_record.to_dict())
            # current_loop_had_error が True でなくても、このケースはエラーとして扱う
            return new_consecutive_errors + 1 if not current_loop_had_error else new_consecutive_errors

        try:
            dialogue_text_final = version_data.get("dialogue")
            if not (current_loop_had_error) and not (isinstance(dialogue_text_final, str) and dialogue_text_final.strip()):
                # エラーが報告されていないのにダイアログテキストがないのは問題
                missing_text_msg_handle = f"ループ {version}: エラーなしと報告されたが、version_data に有効な 'dialogue' テキストなし。"
                self.logger.error(missing_text_msg_handle)
                err_obj_dialogue = self._create_error("INTERNAL.INVALID_DIALOGUE_IN_SUCCESS_RESULT", missing_text_msg_handle, f"HandleLoopResult.V{version}.DialogueMissing")
                if not self.state.last_error: self.state.last_error = err_obj_dialogue.to_dict()
                self.state.error_records.append(err_obj_dialogue.to_dict())
                error_dict_to_store = err_obj_dialogue.to_dict() # このエラーを VersionState にも記録
                # このケースもエラーとして扱う
                new_consecutive_errors = consecutive_errors + 1 if not current_loop_had_error else new_consecutive_errors


            llm_scores_data_vs = version_data.get("scores", {})
            llm_eval_raw_text_vs = version_data.get("evaluation")
            dfrs_scores_data_vs = version_data.get("dfrs_scores") # None or Dict or ErrorDict
            analyzer_results_data_vs = version_data.get("analyzer_results")

            llm_scores_model_instance_vs = LLMEvalScoreModel_cls.model_validate(llm_scores_data_vs) if isinstance(llm_scores_data_vs, dict) else None
            if llm_scores_model_instance_vs and isinstance(llm_eval_raw_text_vs, str):
                 llm_scores_model_instance_vs.raw_output = llm_eval_raw_text_vs

            dfrs_scores_model_instance_vs: Optional['DFRSSubScoresV49'] = None # type: ignore
            if isinstance(dfrs_scores_data_vs, dict) and "error" not in dfrs_scores_data_vs:
                try:
                    dfrs_scores_model_instance_vs = DFRSSubScoreModel_cls.model_validate(dfrs_scores_data_vs)
                except ValidationError as ve_dfrs_state:
                    self.logger.warning(f"DFRSスコアのDFRSSubScoresV49モデル検証失敗 (Loop {version}): {ve_dfrs_state.errors(include_url=False) if hasattr(ve_dfrs_state,'errors') else ve_dfrs_state}")
                    dfrs_scores_model_instance_vs = DFRSSubScoreModel_cls.model_validate({"scores": {"error": f"DFRS validation error: {ve_dfrs_state}"}}) # type: ignore
            elif isinstance(dfrs_scores_data_vs, dict) and "error" in dfrs_scores_data_vs: # DFRS評価でエラーがあった場合
                 dfrs_scores_model_instance_vs = DFRSSubScoreModel_cls.model_validate(dfrs_scores_data_vs) # type: ignore
            
            version_state_data_dict = {
                "version_id": version,
                "generated_text": dialogue_text_final,
                "evaluation_text_raw": llm_eval_raw_text_vs if isinstance(llm_eval_raw_text_vs, str) else "(評価テキストなし/エラー時)",
                "llm_scores": llm_scores_model_instance_vs,
                "dfrs_scores": dfrs_scores_model_instance_vs,
                "generation_time_ms": version_data.get("generation_time_ms"),
                "evaluation_time_ms": version_data.get("evaluation_time_ms"),
                "generation_model": getattr(self.api_client, 'model_name', getattr(self.config, 'DEFAULT_MODEL', "N/A")),
                "status": "error" if error_dict_to_store else "completed",
                "error_info": error_dict_to_store, # ここでエラー辞書を格納
                "timestamp": datetime.now(timezone.utc),
                "analyzer_results": analyzer_results_data_vs,
                # 主観性・揺らぎスコアも VersionState に追加する場合 (analyzer_results 内などから取得)
                "estimated_subjectivity": analyzer_results_data_vs.get("subjectivity_score_final") if isinstance(analyzer_results_data_vs, dict) else None,
                "estimated_fluctuation": analyzer_results_data_vs.get("fluctuation_intensity_final") if isinstance(analyzer_results_data_vs, dict) else None,
            }
            new_version_state_obj = VersionState_cls.model_validate(version_state_data_dict)
            self.state.versions.append(new_version_state_obj)
            self.logger.debug(f"ループ {version}: VersionState作成・追加完了 (エラー状態: {current_loop_had_error})。")

        except ValidationError as ve_state_handle_final:
            error_details_final = ve_state_handle_final.errors(include_url=False) if hasattr(ve_state_handle_final, 'errors') else str(ve_state_handle_final)
            self.logger.error(f"ループ {version} VersionState作成最終検証エラー: {error_details_final}", exc_info=True)
            err_final = self._create_error("INTERNAL.VERSION_STATE_FINAL_VALIDATION_ERROR", f"VersionState最終検証エラー: {error_details_final}", f"HandleLoopResult.V{version}.StateFinalValidation", original_exception=ve_state_handle_final)
            if not self.state.last_error: self.state.last_error = err_final.to_dict()
            self.state.error_records.append(err_final.to_dict())
            return new_consecutive_errors + 1 # エラーとしてカウント
        except Exception as e_state_create_handle_final:
            self.logger.error(f"ループ {version} VersionState作成中最終エラー: {e_state_create_handle_final}", exc_info=True)
            err_final_unexp = self.exception_manager.log_error(e_state_create_handle_final, f"HandleResultV{version}.StateFinalCreation") # type: ignore
            if not self.state.last_error: self.state.last_error = err_final_unexp.to_dict()
            self.state.error_records.append(err_final_unexp.to_dict())
            return new_consecutive_errors + 1 # エラーとしてカウント

        if not current_loop_had_error: # エラーがなかった場合のみループカウンタをインクリメント
            self.state.current_loop += 1
            
        return new_consecutive_errors

    def _re_evaluate_final_dfrs_if_needed(self, final_v_num: int) -> None:
        """必要なら最終バージョンのDFRSスコアを再計算する"""
        if not self.state or not (0 < final_v_num <= len(self.state.versions)):
            self.logger.warning(f"_re_evaluate_final_dfrs_if_needed: 無効なfinal_v_num ({final_v_num}) または stateなし。スキップ。")
            return

        dfrs_enabled = getattr(self.settings, 'dfrs_evaluation_enabled', False)
        # dfrs_evaluate_all_loops が False の場合のみ、最終版の再評価を検討
        dfrs_eval_all_loops_flag = getattr(self.settings, 'dfrs_evaluate_all_loops', False)

        if dfrs_enabled and not dfrs_eval_all_loops_flag:
            final_v_state = self.state.versions[final_v_num - 1]
            DFRSSubScores_cls: Optional[Type['DFRSSubScoresV49']] = _get_global_type('DFRSSubScoresV49') # type: ignore
            if not DFRSSubScores_cls:
                self.logger.error("DFRS再評価スキップ: DFRSSubScoresV49モデルクラス未定義。")
                return

            needs_re_eval = False
            if final_v_state.dfrs_scores is None:
                needs_re_eval = True
                self.logger.info(f"最終バージョンv{final_v_num}のDFRSスコア未計算(None)のため再計算。")
            elif isinstance(final_v_state.dfrs_scores, DFRSSubScores_cls) and \
                 hasattr(final_v_state.dfrs_scores, 'scores') and \
                 isinstance(final_v_state.dfrs_scores.scores, dict) and \
                 "error" in final_v_state.dfrs_scores.scores: # 'scores' 辞書内に 'error' キーがあるか確認
                needs_re_eval = True
                self.logger.info(f"最終バージョンv{final_v_num}のDFRSスコアで以前エラー発生のため再計算。 Error: {final_v_state.dfrs_scores.scores.get('error')}")

            if needs_re_eval:
                self.logger.info(f"最終選択バージョン v{final_v_num} のDFRSスコアを再計算します...")
                analyzer_res_data_final = final_v_state.analyzer_results # これは辞書を期待

                if not (self.dep.evaluator and hasattr(self.dep.evaluator, 'get_dfrs_scores_v49')):
                    self.logger.error("DFRS再評価スキップ: Evaluator未設定またはget_dfrs_scores_v49メソッドなし。")
                    return
                
                intended_phase_final = self.state.current_intended_phase # 最終ループ終了時の意図
                intended_tone_final = self.state.current_intended_tone
                self.logger.debug(f"  DFRS再評価時(v{final_v_num})の意図位相: {getattr(intended_phase_final, 'value', 'N/A')}, 意図トーン: {getattr(intended_tone_final, 'value', 'N/A')}")

                ok_re_eval, res_dict_re_eval, err_obj_re_eval = self.exception_manager.safe_nlp_processing( # type: ignore
                    f"最終DFRS再評価(v{final_v_num})",
                    self.dep.evaluator.get_dfrs_scores_v49,
                    kwargs={
                        'dialogue_text': final_v_state.generated_text,
                        'analyzer_results': analyzer_res_data_final,
                        'intended_phase': intended_phase_final,
                        'intended_tone': intended_tone_final
                    }
                )
                if ok_re_eval and isinstance(res_dict_re_eval, dict):
                    self.state.final_dfrs_scores = res_dict_re_eval # GeneratorState直下の最終DFRSスコアを更新
                    try:
                        final_v_state.dfrs_scores = DFRSSubScores_cls.model_validate(res_dict_re_eval)
                        self.logger.info(f"v{final_v_num} DFRSスコア再計算・VersionState更新完了。")
                    except ValidationError as model_val_err_re_eval:
                         self.logger.error(f"再計算DFRS結果のDFRSSubScoresV49検証失敗(v{final_v_num}): {model_val_err_re_eval.errors(include_url=False) if hasattr(model_val_err_re_eval,'errors') else model_val_err_re_eval}")
                    except Exception as model_err_re_eval:
                         self.logger.error(f"再計算DFRS結果のVersionStateへのモデル設定中エラー(v{final_v_num}): {model_err_re_eval}", exc_info=True)
                else:
                    error_message_re_eval = str(err_obj_re_eval) if err_obj_re_eval else "不明なDFRS再評価エラー"
                    self.logger.error(f"最終バージョン v{final_v_num} のDFRSスコア再計算失敗: {error_message_re_eval}")
                    error_info_dict_re_eval = err_obj_re_eval.to_dict() if hasattr(err_obj_re_eval, 'to_dict') else {"code": "DFRS.FINAL_RE_EVAL_ERROR_UNKNOWN", "message": error_message_re_eval}
                    final_v_state.dfrs_scores = DFRSSubScores_cls.model_validate({"scores": {"error": error_info_dict_re_eval}})
                    if self.state.final_dfrs_scores is None: self.state.final_dfrs_scores = {}
                    if isinstance(self.state.final_dfrs_scores, dict): self.state.final_dfrs_scores["error_in_final_re_eval"] = error_info_dict_re_eval
        else:
             self.logger.info(f"最終バージョンのDFRSスコア再評価は不要です (DFRS無効または全ループで評価済み)。")

# =============================================================================
# -- Part 14 終了点
# =============================================================================
# =============================================================================
# -- Part 15: Dialogue Manager, Arg Parser (v4.9α - 改善版)
# =============================================================================
# v4.9α: ファイルI/O管理クラス、コマンドライン引数パーサー、
#        そしてスクリプト全体のエントリーポイントとなるmain関数。
# 改善版: Pydanticモデル連携強化(保存/ロード/エクスポート)、引数パーサー整理、
#         main関数での依存性注入とエラーハンドリング改善、SyntaxError修正。

# --- Dialogue Manager (io/manager_v49.py 相当 - 改善版) ---
class DialogueManagerV49: # Implicitly implements DialogueManagerProtocol
    """対話生成ジョブのファイル入出力を管理 (v4.9α 改善版)"""
    if TYPE_CHECKING:
        # ConfigProto = ConfigProtocol # この行は削除またはコメントアウトしても良い
        StateType = GeneratorStateV49
        StructuredErrorType = StructuredErrorV49
        OutputJsonStructureType = OutputJsonStructureV49
        ExceptionManagerProto = ExceptionManagerProtocol
        SettingsProto = SettingsProtocol
        # ConfigProtocol はグローバルスコープまたは Part 4a で定義されているはずなので、ここで再定義は不要
    else:
        # ConfigProto = 'ConfigProtocol' # この行は削除またはコメントアウトしても良い
        StateType = 'GeneratorStateV49'
        StructuredErrorType = 'StructuredErrorV49'
        OutputJsonStructureType = 'OutputJsonStructureV49'
        ExceptionManagerProto = 'ExceptionManagerProtocol'
        SettingsProto = 'SettingsProtocol'

    # --- ▼▼▼ __init__ メソッドの型ヒントを修正 ▼▼▼ ---
    def __init__(self, job_id: str, config: 'ConfigProtocol', exception_manager: 'ExceptionManagerProto'): # type: ignore
    # --- ▲▲▲ ここまで修正 ▲▲▲ ---
        if not job_id: raise ValueError("job_id必須")
        
        self.config: ConfigProtocol = config # type: ignore
        self.exception_manager: ExceptionManagerProto = exception_manager # type: ignore
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        safe_job_id = sanitize_filename(job_id, max_length=80, replacement="-")
        if len(job_id) > 80:
            self.logger.warning(f"指定された job_id '{job_id}' は長いため、'{safe_job_id}' に短縮/サニタイズされました。")
        self.job_id = safe_job_id
        
        # fs は self.config.loaded_external_configs.file_settings を指す
        fs = getattr(config.loaded_external_configs, 'file_settings', None) if config.loaded_external_configs else None
        
        def get_setting(config_source: Optional[Any], attr_name: str, default_value_from_app_config: Any) -> Any:
            # config_source (通常は FileSettingsV49 インスタンス) から属性を取得しようと試みる
            # なければ default_value_from_app_config を返す
            if config_source and hasattr(config_source, attr_name):
                value = getattr(config_source, attr_name)
                if value is not None: # Noneでない明確な値が設定されていればそれを使用
                    return value
            return default_value_from_app_config

        # ▼▼▼ 修正箇所 ▼▼▼
        # self.base_output_dir = pathlib.Path(get_setting('base_output_dir', config.DEFAULT_BASE_OUTPUT_DIR)).resolve()
        # AppConfigインスタンスの base_output_dir (小文字、pathlib.Path型) をデフォルト値として使用
        # get_setting は FileSettingsV49 から 'base_output_dir' を取得しようとし、
        # なければ config.base_output_dir (AppConfigV49のインスタンス属性) を使用する。
        self.base_output_dir = pathlib.Path(
            get_setting(fs, 'base_output_dir', config.base_output_dir) # config.base_output_dir を参照
        ).resolve()
        # ▲▲▲ 修正箇所 ▲▲▲
        self.logger.debug(f"DialogueManager base_output_dir: '{self.base_output_dir}' (Length: {len(str(self.base_output_dir))})")
        
        self.filename_max_length = get_setting(fs, 'filename_max_length', config.filename_max_length) # config.filename_max_length を参照
        
        # 他の属性も同様に、AppConfigインスタンスの対応する小文字の属性をデフォルト値として参照するように修正
        self.resume_dir_name = get_setting(fs, 'resume_dir_name', AppConfigV49.RESUME_DIR_NAME_DEFAULT) # クラス定数から取得
        self.stats_dir_name = get_setting(fs, 'stats_dir_name', AppConfigV49.STATS_DIR_NAME_DEFAULT)
        self.prompt_dir_name = get_setting(fs, 'prompt_dir_name', AppConfigV49.PROMPT_DIR_NAME_DEFAULT)
        self.eval_dir_name = get_setting(fs, 'eval_dir_name', AppConfigV49.EVAL_DIR_NAME_DEFAULT)
        self.json_export_dir_name = get_setting(fs, 'json_export_dir_name', AppConfigV49.JSON_EXPORT_DIR_NAME_DEFAULT)
        self.rejected_dir_name = get_setting(fs, 'rejected_dir_name', AppConfigV49.REJECTED_DIR_NAME_DEFAULT)
        self.stats_filename = get_setting(fs, 'stats_filename', AppConfigV49.STATS_FILENAME_DEFAULT)
        self.lock_suffix = get_setting(fs, 'lock_suffix', AppConfigV49.LOCK_SUFFIX_DEFAULT)
        self.resume_suffix = get_setting(fs, 'resume_suffix', AppConfigV49.RESUME_SUFFIX_DEFAULT)
        
        self.job_output_dir = self.base_output_dir / self.job_id
        self.logger.debug(f"DialogueManager job_output_dir: '{self.job_output_dir}' (Length: {len(str(self.job_output_dir))})")
        
        self.resume_dir=self.job_output_dir/self.resume_dir_name; self.prompt_dir=self.job_output_dir/self.prompt_dir_name
        self.evaluation_dir=self.job_output_dir/self.eval_dir_name; self.json_export_dir=self.job_output_dir/self.json_export_dir_name
        self.rejected_dir=self.job_output_dir/self.rejected_dir_name; self.stats_dir=self.base_output_dir/self.stats_dir_name
        self.logger.debug(f"DialogueManager(Job:{self.job_id})初期化, Output Dir: {self.job_output_dir}")



    def initialize_directories(self, settings: SettingsProto) -> None:
        self.logger.debug("出力ディレクトリ初期化..."); dirs=[self.job_output_dir, self.resume_dir, self.stats_dir]
        if getattr(settings,'save_prompts',False): dirs.append(self.prompt_dir)
        if getattr(settings,'save_evaluations',False): dirs.append(self.evaluation_dir)
        if getattr(settings,'json_export_enabled',True): dirs.append(self.json_export_dir)
        if getattr(settings,'save_rejected_candidates',False): dirs.append(self.rejected_dir)
        AppCfg_cls = globals().get('AppConfigV49'); assert AppCfg_cls
        for d_path in dirs:
             self.logger.debug(f"  Ensuring directory: '{d_path}' (Length: {len(str(d_path.resolve()))})")
             try: AppCfg_cls._ensure_dir_exists(d_path)
             except Exception as e:
                 logf=self.logger.error if d_path in [self.job_output_dir, self.resume_dir, self.stats_dir] else self.logger.warning
                 logf(f"ディレクトリ作成エラー:{d_path}-{e}")
                 if d_path in [self.job_output_dir, self.resume_dir, self.stats_dir]:
                     raise RuntimeError(f"必須ディレクトリ '{d_path}' の作成に失敗しました。") from e

    def _get_path(self, dir_path: pathlib.Path, filename: str) -> pathlib.Path:
        sanitized_filename = sanitize_filename(filename, self.filename_max_length)
        full_path = dir_path / sanitized_filename
        self.logger.debug(f"_get_path: dir='{dir_path}', orig_filename='{filename}', sanitized_filename='{sanitized_filename}', full_path='{full_path}' (Full Path Length: {len(str(full_path.resolve()))})")
        return full_path

    def save_prompt(self, name_stem: str, text_content: str) -> None:
        """指定された名前でプロンプトテキストを保存します。"""
        if not isinstance(text_content, str):
            self.logger.error(f"プロンプト保存エラー ({name_stem}): コンテンツが文字列ではありません (型: {type(text_content)})。")
            return
        
        filename = f"{name_stem}.txt"
        prompt_filepath = self._get_path(self.prompt_dir, filename)
        
        op_ok, _, op_err = self.exception_manager.safe_file_operation(
            f"プロンプト保存 ({name_stem})",
            save_text,
            args=(prompt_filepath, text_content)
        )
        if not op_ok:
            self.logger.error(f"プロンプトファイル '{filename}' の保存に失敗しました。Path='{prompt_filepath}', Error='{op_err}'")

    def save_dialogue(self, version: int, text: str) -> None:
        p = self._get_path(self.job_output_dir, f"dialogue_v{version}.txt")
        op_ok, _, op_err = self.exception_manager.safe_file_operation(f"対話保存(v{version})", save_text, args=(p, text))
        if not op_ok: self.logger.error(f"対話ファイル(v{version})保存失敗: Path='{p}', Error='{op_err}'")

    def save_evaluation(self, version: int, text: str) -> None:
        p = self._get_path(self.evaluation_dir, f"evaluation_v{version}.txt")
        op_ok, _, op_err = self.exception_manager.safe_file_operation(f"評価保存(v{version})", save_text, args=(p, text))
        if not op_ok: self.logger.error(f"評価ファイル(v{version})保存失敗: Path='{p}', Error='{op_err}'")
    
    def save_rejected_candidate(self, cand_data: Dict[str, Any], reason: str) -> None:
        # ... (変更なし) ...
        v=cand_data.get('version',0); idx=cand_data.get('index',0); safe_r=sanitize_filename(reason,50); p=self._get_path(self.rejected_dir, f"rejected_v{v}_cand{idx}_{safe_r}.json")
        data={k:cand_data.get(k) for k in ["reason","index","version","scores","dfrs_scores","error"]}; data["text_snip"]=str(cand_data.get("text",""))[:500]+"..."; data["eval_snip"]=str(cand_data.get("evaluation",""))[:500]+"..." if cand_data.get("evaluation") else None
        self.exception_manager.safe_file_operation(f"不採用候補保存(v{v}_cand{idx})", save_json, args=(data, p))

    def save_resume_state(self, state: StateType) -> bool:
        # ... (変更なし) ...
        GenState_cls = globals().get('GeneratorStateV49'); assert GenState_cls
        if not isinstance(state, GenState_cls): self.logger.error("不正状態型"); return False
        p = self._get_path(self.resume_dir, f"{self.job_id}{self.resume_suffix}"); create_backup(p)
        
        self.logger.debug(f"--- Serializing state for Job ID: {state.job_id} ---")
        try:
            state_dict = state.model_dump(mode='python')
            for attr_name, attr_value in state_dict.items():
                self.logger.debug(f"  Attr: {attr_name}, Type: {type(attr_value)}")
                if attr_name == 'settings_snapshot' and isinstance(attr_value, dict):
                    self.logger.debug(f"    Checking settings_snapshot ({len(attr_value)} items):")
                    for k, v in attr_value.items():
                        if callable(v):
                            self.logger.warning(f"      !!! Method object found in settings_snapshot: Key='{k}', Type={type(v)}, Value={v} !!!")
        except Exception as debug_err:
            self.logger.error(f"Error during state debug logging: {debug_err}", exc_info=True)
        self.logger.debug(f"--- End of state serialization check ---")
        try:
            json_s = state.model_dump_json(indent=2, exclude_none=True, by_alias=True)
            ok, _, e = self.exception_manager.safe_file_operation(
                "レジューム保存", save_text, args=(p, json_s) # save_text の引数順序は (path, content)
            )
            return ok
        except Exception as e:
            self.logger.error(f"レジュームJSONシリアライズまたは保存呼び出しエラー:{e}", exc_info=True)
            return False

    # --- ▼▼▼ load_resume_state メソッドを追加 ▼▼▼ ---
    def load_resume_state(self) -> Optional['GeneratorStateV49']:
        """レジューム状態をJSONファイルから読み込み、検証する"""
        filepath = self._get_path(self.resume_dir, f"{self.job_id}{self.resume_suffix}")

        if not filepath.is_file():
            self.logger.info(f"レジュームファイルが見つかりません: {filepath}")
            return None

        load_ok, data, load_err = self.exception_manager.safe_file_operation(
            "レジューム読込",
            load_json, # utils_v49.py 相当の load_json 関数
            args=(filepath,)
        )

        if not load_ok or not isinstance(data, dict):
            self.logger.error(f"レジュームファイル読込/パース失敗: {filepath}, Error: {load_err}")
            return None

        GeneratorStateV49_cls = globals().get('GeneratorStateV49')
        if not GeneratorStateV49_cls:
            self.logger.error("GeneratorStateV49モデルクラス未定義。レジュームデータ検証不可。")
            return None

        try:
            state_model = GeneratorStateV49_cls.model_validate(data)
            self.logger.info(f"レジュームファイル読込・検証成功: {filepath}")
            return state_model
        except ValidationError as e:
            self.logger.error(f"レジュームデータ検証エラー ({filepath}):")
            try:
                for error_detail in e.errors():
                    loc_str = " -> ".join(map(str, error_detail.get('loc',[])))
                    self.logger.error(f"  - 場所: {loc_str}, エラー: {error_detail.get('msg')}")
            except Exception:
                self.logger.error(f"  - (エラー詳細の表示に失敗: {e})")
            self.logger.warning("レジュームファイルが破損しているか、バージョン不適合の可能性があります。破損ファイルとしてリネームを試みます。")
            try:
                corrupted_suffix = f"{filepath.suffix}.corrupted_{int(time.time())}"
                corrupted_path = filepath.with_suffix(corrupted_suffix)
                filepath.rename(corrupted_path)
                self.logger.info(f"破損レジュームファイルをリネームしました: {corrupted_path}")
            except Exception as rename_err:
                self.logger.error(f"破損レジュームファイルのリネームに失敗しました ({filepath}): {rename_err}")
            return None
        except Exception as e:
            self.logger.error(f"レジュームデータのPydantic処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return None
    # --- ▲▲▲ load_resume_state メソッドを追加 ▲▲▲ ---

    def append_stats(self, stats_data: Dict[str, Any]) -> None:
        # ... (変更なし) ...
        stats_filepath = self._get_path(self.stats_dir, self.stats_filename)
        lock_timeout = 5.0
        lock_context = nullcontext()

        if FILELOCK_AVAILABLE and FileLock:
            try:
                lock_filepath = stats_filepath.with_suffix(f"{stats_filepath.suffix}{self.lock_suffix}")
                lock_context = FileLock(str(lock_filepath), timeout=lock_timeout)
                self.logger.debug(f"FileLock準備完了: {lock_filepath}")
            except Exception as e:
                self.logger.warning(f"FileLockの初期化に失敗しました ({lock_filepath}): {e}。ロックなしで続行します。")
        elif not hasattr(self, '_warned_no_filelock_stats'):
            self.logger.warning("filelockライブラリが見つからないため、統計ファイル追記時のロックは行われません。")
            setattr(self, '_warned_no_filelock_stats', True)

        try:
            with lock_context:
                stats_filepath.parent.mkdir(parents=True, exist_ok=True)
                with stats_filepath.open('a', encoding='utf-8') as f:
                    json_line = json.dumps(stats_data, ensure_ascii=False, default=str)
                    f.write(json_line + '\n')
                self.logger.debug(f"統計データ追記成功: {stats_filepath}")
        except Timeout:
            self.logger.warning(f"統計ファイル'{stats_filepath}'のロック取得がタイムアウトしました ({lock_timeout}秒)。追記できませんでした。")
        except Exception as e:
            struct_err = StructuredErrorV49(
                f"統計ファイル追記またはJSONシリアライズ中にエラーが発生しました: {e}",
                "FILE.STATS_WRITE_ERROR",
                "DialogueManager.append_stats",
                original_exception=e
            )
            self.exception_manager.log_error(struct_err)

    def save_final_results(self, final_state: StateType, report_type: str) -> None:
        self.logger.info("最終結果保存開始...");
        if not final_state: self.logger.error("最終状態None。保存不可。"); return
        
        json_p_str:Optional[str] = None
        json_enabled = getattr(self.config.feature_flags,'json_export_enabled',True) if self.config.feature_flags else True
        if json_enabled:
             exported_path_obj = self._export_to_json(final_state)
             if exported_path_obj:
                 json_p_str=str(exported_path_obj)
                 final_state.final_output_json_path = json_p_str
             else:
                 self.logger.error("最終JSONエクスポート失敗。")
        try:
             md_content_str = self._generate_markdown_summary(final_state, report_type)
             report_path_obj = self._get_path(self.job_output_dir, "final_report.md")
             # --- ▼▼▼ save_text の呼び出し引数順序を修正 (path, content) ▼▼▼ ---
             ok,_,e = self.exception_manager.safe_file_operation(
                 "Markdown保存",
                 save_text,
                 args=(report_path_obj, md_content_str) # 修正: 第1引数 path, 第2引数 content
             )
             # --- ▲▲▲ ここまで修正 ▲▲▲ ---
             if ok: self.logger.info(f"Markdownレポート保存成功:{report_path_obj}")
             else: self.logger.error(f"Markdownレポート保存失敗: {e}")
        except Exception as e_md:
            self.logger.error(f"Markdown生成/保存処理中に予期せぬエラー:{e_md}", exc_info=True)

        try:
            stats_summary_dict = self._prepare_stats_summary(final_state)
            if stats_summary_dict:
                 self.append_stats(stats_summary_dict)
            else:
                 self.logger.warning("統計サマリーが空のため、追記をスキップしました。")
        except Exception as e_stats:
            self.logger.error(f"統計サマリ生成/追記処理中に予期せぬエラー:{e_stats}", exc_info=True)
        
        final_state.complete=True
        self.save_resume_state(final_state)
        self.logger.info("最終結果保存完了。")

    def _prepare_stats_summary(self, state: StateType) -> Dict[str, Any]:
        # ... (変更なし) ...
        if not state: return {"job_id": self.job_id, "status": "Error", "error_code": "STATE.MISSING"}
        final_eval=state.final_evaluation_summary; final_stats=state.final_generation_stats; settings=state.settings_snapshot or {}
        def get_s(metric):
             score=None; DFRS=globals().get('DFRSMetricsV49'); LLM_cls=globals().get('ScoreKeys', type('',(),{'LLM':None})).LLM
             if final_eval:
                 if LLM_cls and isinstance(metric, LLM_cls) and final_eval.llm_scores: score=getattr(final_eval.llm_scores, metric.value.lower(), None)
                 elif DFRS and isinstance(metric, DFRS) and final_eval.dfrs_scores: score=getattr(final_eval.dfrs_scores, metric.value.lower(), None)
             if score is None and isinstance(state.final_dfrs_scores,dict): score=state.final_dfrs_scores.get(metric.value)
             return float(score) if isinstance(score,(int,float)) else None
        err_dict = state.last_error if isinstance(state.last_error, dict) else {}
        err_code = err_dict.get('error_code')
        status="Error" if err_code else "Success"
        DFRS_cls=globals().get('DFRSMetricsV49'); LLMScoreKeys_cls=globals().get('ScoreKeys', type('',(),{'LLM':None})).LLM
        if not (DFRS_cls and LLMScoreKeys_cls): return {"job_id": self.job_id, "status": "Error", "error_code": "INTERNAL.MISSING_MODEL"}
        stats={"job_id":state.job_id, "timestamp_utc":datetime.now(timezone.utc).isoformat()+'Z', "system_version":state.system_version,
               "model_used":state.model_name, "status":status, "error_code":err_code, "total_loops":state.current_loop, "final_version":state.final_version,
               "final_llm_score":round(state.final_score_llm,3) if state.final_score_llm is not None else None,
               "final_eodf_v49":round(s,3) if (s:=get_s(DFRS_cls.FINAL_EODF_V49)) is not None else None,
               "final_subjectivity":round(s,3) if (s:=get_s(DFRS_cls.SUBJECTIVITY_SCORE)) is not None else None,
               "final_fluctuation":round(s,3) if (s:=get_s(DFRS_cls.FLUCTUATION_INTENSITY)) is not None else None,
               "final_richness":round(s,3) if (s:=get_s(DFRS_cls.EXPRESSION_RICHNESS)) is not None else None,
               "final_novelty":round(s,3) if (s:=get_s(DFRS_cls.CONTENT_NOVELTY)) is not None else None,
               "duration_seconds":getattr(final_stats,'duration_seconds',None), "error_count":getattr(final_stats,'error_count',len(state.error_records)),
               "settings_summary":{k:settings.get(k) for k in ['dialogue_mode','style_template','subjective_focus','subjective_intensity','adaptation_strategy_type','feedback_strategy_type','dfrs_evaluation_enabled','feedback_loops']}}
        return {k:v for k,v in stats.items() if v is not None and (not isinstance(v,dict) or v)}

    def _generate_markdown_summary(self, state: 'StateType', report_type: str) -> str:
        if not state:
            return "# エラー: 最終状態データがありません"

        final_v_num = state.final_version if state.final_version is not None else 0
        final_txt = "(最終バージョンのテキストが見つかりません)"
        final_v_model: Optional['VersionStateV49'] = None

        if state.versions and 0 < final_v_num <= len(state.versions):
            final_v_model = state.versions[final_v_num - 1]
        elif state.versions:
            final_v_model = state.versions[-1]
            self.logger.warning(f"最終バージョン番号({final_v_num})が不正か、versionsリストの範囲外です。リストの最後のバージョン(v{getattr(final_v_model, 'version_id', 'N/A')})をレポートに使用します。")
            if final_v_model:
                 final_v_num = final_v_model.version_id
                 final_txt = final_v_model.generated_text if final_v_model.generated_text is not None else "(生成テキストなし)"
            else:
                 final_v_num = 0
        else:
            self.logger.warning("versionsリストが空のため、最終対話テキストを取得できません。")
            final_v_num = 0

        if final_v_model and final_v_model.generated_text is not None:
            final_txt = final_v_model.generated_text
        elif final_v_model and final_v_model.generated_text is None:
             final_txt = "(生成テキストなし)"
        
        # --- ▼▼▼ md リストの初期化と文字列結合を確実に ▼▼▼ ---
        md_parts: List[str] = []
        md_parts.append(f"# NDGS {state.system_version} Report")
        md_parts.append(f"- **Job ID:** `{state.job_id}`")
        md_parts.append(f"- **Completed:** {state.completion_time.isoformat() if state.completion_time else 'N/A'}")
        md_parts.append(f"- **Model Used:** `{state.model_name}`")
        md_parts.append(f"- **Final Version:** `v{final_v_num}` (out of `{state.current_loop}` loops)")

        status = "Success"
        err_dict = state.last_error
        if err_dict and isinstance(err_dict, dict):
            error_code = err_dict.get('error_code', 'UNKNOWN_ERROR')
            error_msg = err_dict.get('error_message', '詳細不明')
            status = f"**Error** (`{error_code}`)"
            md_parts.append(f"- **Status:** {status}")
            md_parts.append(f"  - **Error Detail:** {error_msg}")
        else:
            md_parts.append(f"- **Status:** {status}")

        md_parts.append("\n## Final Evaluation Summary (Scale: 0-5)")
        final_eval = state.final_evaluation_summary
        if not final_eval:
            md_parts.append("N/A (評価データがありません)\n")
        else:
            LLMKeys_cls = globals().get('ScoreKeys.LLM')
            DFRSMetrics_cls = globals().get('DFRSMetricsV49')
            fmt_func = globals().get('fmt', str)

            llm_o = "N/A"
            if final_eval.llm_scores and LLMKeys_cls:
                llm_overall_value = getattr(final_eval.llm_scores, LLMKeys_cls.OVERALL.value.lower(), None)
                llm_o = fmt_func(llm_overall_value)

            eodf_score_value = getattr(final_eval, 'final_eodf_v49', None)
            eodf = fmt_func(eodf_score_value)

            dfrs_s = final_eval.dfrs_scores
            subj = "N/A"; fluc = "N/A"; rich = "N/A"; novel = "N/A"; pa = "N/A"; ta = "N/A"
            if dfrs_s and DFRSMetrics_cls and dfrs_s.scores:
                subj = fmt_func(getattr(dfrs_s, 'subjectivity_score', None))
                fluc = fmt_func(getattr(dfrs_s, 'fluctuation_intensity', None))
                rich = fmt_func(getattr(dfrs_s, 'expression_richness', None))
                novel = fmt_func(getattr(dfrs_s, 'content_novelty', None))
                pa = fmt_func(getattr(dfrs_s, 'phase_alignment', None))
                ta = fmt_func(getattr(dfrs_s, 'tone_alignment', None))

            md_parts.append("| Eval Category       | LLM Overall | DFRS Overall(v4.9) | Subjectivity | Fluctuation | Richness | Novelty | Phase Align | Tone Align |")
            md_parts.append("|-------------------|-------------|--------------------|--------------|-------------|----------|---------|-------------|------------|")
            md_parts.append(f"| **Main Scores** | {llm_o:<11} | **{eodf:<18}** | {subj:<12} | {fluc:<11} | {rich:<8} | {novel:<7} | {pa:<11} | {ta:<10} |")
            md_parts.append("")

            if report_type == 'full':
                md_parts.append("### Detailed DFRS Scores")
                if dfrs_s and DFRSMetrics_cls and dfrs_s.scores:
                    md_parts.append("| Metric | Score |")
                    md_parts.append("|---|---|")
                    for metric_enum_member in DFRSMetrics_cls:
                        score_val = dfrs_s.scores.get(metric_enum_member.value)
                        md_parts.append(f"| {metric_enum_member.value} | {fmt_func(score_val)} |")
                    md_parts.append("")
                else:
                    md_parts.append("N/A (詳細スコアデータなし)")

                if final_eval.evaluation_feedback:
                    md_parts.append("\n#### Final LLM Evaluation Feedback:\n```\n")
                    md_parts.append(final_eval.evaluation_feedback)
                    md_parts.append("```\n")
                else:
                    md_parts.append("\n#### Final LLM Evaluation Feedback: N/A\n")

        md_parts.append(f"\n## Final Dialogue (Version {final_v_num})\n\n```text")
        md_parts.append(final_txt)
        md_parts.append("```\n")

        if state.error_records:
            md_parts.append("\n## Error Records")
            md_parts.append("| Loop | Source | Code | Message |")
            md_parts.append("|---|---|---|---|")
            for err_rec in state.error_records:
                 if isinstance(err_rec, dict):
                     loop = err_rec.get('loop_number', 'N/A')
                     source = err_rec.get('error_source', 'N/A')
                     code = err_rec.get('error_code', 'N/A')
                     msg_raw = err_rec.get('error_message', 'N/A')
                     msg = str(msg_raw).replace('\n',' ').replace('|',' ')[:100] + ('...' if len(str(msg_raw)) > 100 else '')
                     md_parts.append(f"| {loop} | {source} | {code} | {msg} |")
            md_parts.append("")
        
        return "\n".join(md_parts) # md_parts を文字列として結合して返す
        # --- ▲▲▲ md リストの初期化と文字列結合を確実に ▲▲▲ ---

    def _export_to_json(self, final_state: StateType) -> Optional[pathlib.Path]:
        # ... (既存の _export_to_json メソッドは変更なし、ただし OutputDialogue が None にならないように注意) ...
        self.logger.info("最終結果JSONエクスポート処理開始...")

        OutJSON_cls = globals().get('OutputJsonStructureV49')
        OutMeta_cls = globals().get('OutputMetadataV49')
        SettingsMeta_cls = globals().get('SettingsMetadataV49')
        InCtx_cls = globals().get('InputContextV49')
        OutCharCtx_cls = globals().get('OutputCharacterContextV49')
        OutSceneCtx_cls = globals().get('OutputSceneContextV49')
        OutDiag_cls = globals().get('OutputDialogueV49')
        SpBlock_cls = globals().get('SpeechBlockV49')
        DescBlock_cls = globals().get('DescriptionBlockV49')

        required_classes_map = {
            'OutputJsonStructureV49': OutJSON_cls, 'OutputMetadataV49': OutMeta_cls,
            'SettingsMetadataV49': SettingsMeta_cls, 'InputContextV49': InCtx_cls,
            'OutputCharacterContextV49': OutCharCtx_cls, 'OutputSceneContextV49': OutSceneCtx_cls,
            'OutputDialogueV49': OutDiag_cls, 'SpeechBlockV49': SpBlock_cls,
            'DescriptionBlockV49': DescBlock_cls
        }
        missing_classes = [name for name, cls in required_classes_map.items() if cls is None]
        if missing_classes:
            self.logger.error(f"JSONエクスポートに必要なモデルクラスが見つかりません: {', '.join(missing_classes)}。処理を中止します。")
            return None
        if not final_state:
            self.logger.error("最終状態データ (final_state) がNoneのため、JSONエクスポートを中止します。")
            return None

        try:
            metadata_obj: Optional[OutputMetadataV49] = None
            input_context_obj: Optional[InputContextV49] = None
            output_dialogue_obj: Optional[OutputDialogueV49] = None

            # --- Metadata 構築 ---
            settings_meta = None
            if SettingsMeta_cls and final_state.settings_snapshot:
                try:
                    settings_meta = SettingsMeta_cls.model_validate(final_state.settings_snapshot)
                except Exception as e_inner:
                    self.logger.warning(f"SettingsMetadataの検証中にエラーが発生しました: {e_inner}。デフォルト値を使用します。")
                    settings_meta = SettingsMeta_cls()
            if OutMeta_cls:
                try:
                    metadata_obj = OutMeta_cls(
                        job_id=final_state.job_id,
                        generation_time=final_state.completion_time or datetime.now(timezone.utc),
                        model_used=final_state.model_name,
                        settings=settings_meta,
                        system_version=final_state.system_version,
                        generation_stats=final_state.final_generation_stats
                    )
                except Exception as e:
                    self.logger.error(f"OutputMetadataV49の作成中にエラー: {e}", exc_info=True)
                    metadata_obj = None

            # --- Input Context 構築 ---
            if InCtx_cls and OutCharCtx_cls and OutSceneCtx_cls and final_state.input_data:
                try:
                    char_a_ctx = OutCharCtx_cls.model_validate(final_state.input_data.characterA.model_dump())
                    char_b_ctx = OutCharCtx_cls.model_validate(final_state.input_data.characterB.model_dump())
                    scene_ctx = OutSceneCtx_cls.model_validate(final_state.input_data.sceneInfo.model_dump())
                    input_context_obj = InCtx_cls(character_sheets=[char_a_ctx, char_b_ctx], scene_sheet=scene_ctx)
                except Exception as e:
                    self.logger.error(f"InputContextV49の作成中にエラー: {e}", exc_info=True)
                    input_context_obj = None

            # --- Output Dialogue 構築 ---
            final_v_state_for_output: Optional['VersionStateV49'] = None
            if final_state.final_version is not None and 0 < final_state.final_version <= len(final_state.versions):
                final_v_state_for_output = final_state.versions[final_state.final_version - 1]
            elif final_state.versions:
                final_v_state_for_output = final_state.versions[-1]
                self.logger.warning(f"JSONエクスポート: 最終バージョン番号({final_state.final_version})不正。リスト最後のv{getattr(final_v_state_for_output,'version_id','N/A')}を使用。")

            if OutDiag_cls and SpBlock_cls and DescBlock_cls and final_v_state_for_output:
                final_txt_for_output = final_v_state_for_output.generated_text or ""
                blocks_validated: List[Union['SpeechBlockV49', 'DescriptionBlockV49']] = []

                analyzer_res_for_output = final_v_state_for_output.analyzer_results
                if isinstance(analyzer_res_for_output, dict):
                    output_content = analyzer_res_for_output.get("output_dialogue_content", {})
                    if isinstance(output_content, dict):
                        blocks_raw = output_content.get("blocks", [])
                        if isinstance(blocks_raw, list):
                            for b_dict in blocks_raw:
                                if isinstance(b_dict, dict):
                                    block_type = b_dict.get("type")
                                    target_model_cls = SpBlock_cls if block_type == "speech" else DescBlock_cls if block_type == "description" else None
                                    if target_model_cls:
                                        try:
                                            validated_block = target_model_cls.model_validate(b_dict)
                                            blocks_validated.append(validated_block)
                                        except Exception: pass
                try:
                    output_dialogue_obj = OutDiag_cls(dialogue_blocks=blocks_validated, total_length=len(final_txt_for_output))
                except Exception as e:
                    self.logger.error(f"OutputDialogueV49の作成中にエラー: {e}", exc_info=True)
                    output_dialogue_obj = None # エラー時はNone
            elif OutDiag_cls:
                 output_dialogue_obj = OutDiag_cls(dialogue_blocks=[], total_length=0) # フォールバック

            if not (metadata_obj and input_context_obj and output_dialogue_obj): # output_dialogue_obj もチェック
                missing_parts = []
                if not metadata_obj: missing_parts.append("Metadata")
                if not input_context_obj: missing_parts.append("InputContext")
                if not output_dialogue_obj: missing_parts.append("OutputDialogue") # ここで不足と判断される可能性
                if missing_parts:
                    raise ValueError(f"JSONエクスポートに必要な構成要素を作成できませんでした: {', '.join(missing_parts)}")
            
            final_json_model = OutJSON_cls(
                metadata=metadata_obj,
                input_context=input_context_obj,
                output_dialogue=output_dialogue_obj,
                evaluation=final_state.final_evaluation_summary,
                error_records=final_state.error_records
            )

            json_filepath = self._get_path(self.json_export_dir, f"{self.job_id}_output_v49.json")
            json_data_str = final_json_model.model_dump_json(indent=2, by_alias=True, exclude_none=True)

            save_ok, _, err = self.exception_manager.safe_file_operation("最終JSON保存", save_text, args=(json_filepath, json_data_str))

            if save_ok:
                self.logger.info(f"最終結果JSONのエクスポート成功: {json_filepath}")
                return json_filepath
            else:
                self.logger.error(f"最終結果JSONのファイル保存に失敗しました: {err}")
                return None

        except Exception as e:
            self.logger.error(f"JSONエクスポートプロセス全体で予期せぬエラーが発生しました: {e}", exc_info=True)
            return None

    def get_final_json_path(self) -> Optional[pathlib.Path]:
        ff = self.config.feature_flags
        json_enabled = getattr(ff, 'json_export_enabled', True) if ff else True
        if not json_enabled: return None
        p = self._get_path(self.json_export_dir, f"{self.job_id}_output_v49.json")
        return p if p.is_file() else None

    def save_text_in_rejected_dir(self, filename_stem: str, text: str) -> None:
        p = self._get_path(self.rejected_dir, f"{filename_stem}.txt")
        self.exception_manager.safe_file_operation(f"不採用候補テキスト保存({filename_stem})", save_text, args=(p, text))
# -----------------------------------------------------------------------------
# -- Argument Parser (v4.9α - 改善版)
# -----------------------------------------------------------------------------
def create_argument_parser_v49() -> argparse.ArgumentParser:
    AppCfg_cls = globals().get('AppConfigV49'); sys_ver = getattr(AppCfg_cls,'SYSTEM_VERSION','v4.9α') if AppCfg_cls else 'v4.9α'
    parser = argparse.ArgumentParser(description=f'NDGS {sys_ver}', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    app = parser.add_argument_group('AppConfig Overrides'); job = parser.add_argument_group('DialogueSettings Overrides'); exe = parser.add_argument_group('Execution Mode')
    app.add_argument('--out-dir',dest='DEFAULT_BASE_OUTPUT_DIR'); app.add_argument('--config-dir',dest='CONFIG_DIR'); app.add_argument('--log-level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL']); app.add_argument('--log-file'); app.add_argument('--model',dest='DEFAULT_MODEL'); app.add_argument('--clear-cache',action='store_true'); app.add_argument('--cache-vacuum',action='store_true'); app.add_argument('--external-config',dest='EXTERNAL_CONFIG_ENABLED',action=argparse.BooleanOptionalAction,default=True); app.add_argument('--app-config-yaml',dest='APP_CONFIG_YAML_PATH')
    job.add_argument('data_file',nargs='?',default=None); job.add_argument('--data','-d',dest='data_file_opt'); job.add_argument('--job-id','-j'); job.add_argument('--length','-l',type=int); job.add_argument('--loops','-n',dest='feedback_loops',type=int); job.add_argument('--min-loops',dest='min_feedback_loops',type=int); job.add_argument('--min-score','-m',dest='min_score_threshold',type=float); job.add_argument('--dialogue-mode',choices=["normal","delayed","mixed","auto"]); job.add_argument('--dfrs',dest='dfrs_evaluation_enabled',action=argparse.BooleanOptionalAction,default=None); job.add_argument('--dfrs-all',dest='dfrs_evaluate_all_loops',action=argparse.BooleanOptionalAction,default=None); job.add_argument('--nlp',dest='advanced_nlp_enabled',action=argparse.BooleanOptionalAction,default=None); job.add_argument('--ml-emotion',action=argparse.BooleanOptionalAction,default=None); job.add_argument('--subjective-focus',dest='subjective_focus',action=argparse.BooleanOptionalAction,default=None); job.add_argument('--adaptation',dest='adaptation_strategy_enabled',action=argparse.BooleanOptionalAction,default=None); job.add_argument('--log-transitions',action=argparse.BooleanOptionalAction,default=None); job.add_argument('--style',dest='style_template');
    SubjIntens_cls=globals().get('SubjectiveIntensityLevel'); int_choices=[lvl.value for lvl in SubjIntens_cls] if SubjIntens_cls else []
    job.add_argument('--intensity',dest='subjective_intensity',choices=int_choices); job.add_argument('--adapt-strat',dest='adaptation_strategy_type',choices=['simple_threshold','probabilistic_history','advanced_rl']); job.add_argument('--feedback-strat',dest='feedback_strategy_type',choices=['composite','phase_tone_only','subjectivity_only','quality_only','context_aware','fluctuation_only']); job.add_argument('--dfrs-weights'); job.add_argument('--final-weights'); job.add_argument('--initial-weights'); job.add_argument('--normalize-weights',dest='auto_normalize_weights',action=argparse.BooleanOptionalAction,default=None); job.add_argument('--save-prompts',action=argparse.BooleanOptionalAction,default=None); job.add_argument('--save-evals',dest='save_evaluations',action=argparse.BooleanOptionalAction,default=None); job.add_argument('--save-rejected',dest='save_rejected_candidates',action=argparse.BooleanOptionalAction,default=None)
    exe.add_argument('--check-only','-c',action='store_true'); exe.add_argument('--run-tests',action='store_true'); exe.add_argument('--report','-r',choices=['full','light'],default='light')
    return parser
# =============================================================================
# -- Part 15 終了点
# =============================================================================
# =============================================================================
# -- Part 16: Unit Tests (v4.9α - 最適化・Enum移行対応版)
# =============================================================================
# NDGSコンポーネントのユニットテスト。
# PsychologicalPhaseV49 Enum (11コアメンバー) および関連する
# Pydanticモデルの変更に対応するようにテストデータとアサーションを更新。

import unittest
import unittest.mock as mock # unittest.mock を mock としてインポート
from typing import TYPE_CHECKING, TypeVar, Any, Dict, List, Tuple, Optional, Union, Callable, Type # 必要な型を追加
import enum # 標準ライブラリ
import pathlib # 標準ライブラリ
from datetime import datetime, timezone # 標準ライブラリ
from collections import deque # 標準ライブラリ
# import logging # Part 0でインポート済み想定
# import random # Part 0でインポート済み想定
# import re # Part 0でインポート済み想定
# import json # Part 0でインポート済み想定
# import hashlib # Part 0でインポート済み想定 (DFRSキャッシュキー生成で使用)

# --- 必要なグローバル定義の取得 (Part 0, 1, 2, 3, 4 などで定義済みと仮定) ---
# (実際のテスト環境では、テスト対象モジュールから直接インポートする)
if TYPE_CHECKING:
    from __main__ import (
        setup_logging, logger, # Part 0
        PsychologicalPhaseV49, EmotionalToneV49, SubjectiveIntensityLevel, # Part 1
        ScoreKeys, DFRSMetricsV49, SubjectivityCategoryV49, FluctuationCategoryV49,
        AppConfigV49, ConfigProtocol, load_json, save_json, create_backup, # Part 2
        PersistentCache, # Part 2c
        # Part 3 Models
        ExternalConfigsV49, APISettingsV49, FileSettingsV49, FeatureFlagConfigV49,
        DFRSWeightsConfigV49, TemperatureStrategyConfigV49, AdaptationStrategyConfigV49,
        FeedbackStrategyConfigV49, PhaseToneFeedbackParams, # Part 3a
        SubjectivityKeywordsFileV49, SubjectivityKeywordEntryV49,
        FluctuationPatternsFileV49, FluctuationPatternEntryV49, # Part 3b
        FeedbackContextV49, PhaseTransitionRecordV49, GeneratorStateV49,
        DialogueBlockV49, SpeechBlockV49, DescriptionBlockV49, BlockAnalysisTagsV49,
        EmotionCurvePointV49, PhaseTimelinePointV49, InputDataV49, CharacterV49, SceneInfoV49,
        VersionStateV49, # Part 3c
        # Part 4 Protocols & Classes
        SettingsProtocol, ApiClientProtocol, ScorerProtocol, AnalyzerProtocol,
        EvaluatorProtocol, PromptBuilderProtocol, AdaptationStrategyProtocol,
        FeedbackStrategyProtocol, FeedbackManagerProtocol, StyleManagerProtocol,
        DialogueManagerProtocol, ExceptionManagerProtocol, ApiClientV49,
        # Part 5 Classes
        PromptBuilderV49,
        # Part 6 Classes (変更なしと判断されたが、テストでは参照される可能性)
        SubjectivityFluctuationScorerV49,
        # Part 7 Classes
        EnhancedDialogFlowEvaluatorV49,
        # Part 7b Classes
        AdvancedDialogueAnalyzerV49,
        # Part 8 Classes
        PhaseToneAdaptationStrategyV49,
        # Part 9 Classes
        PhaseToneFeedbackStrategyV49, SubjectivityFeedbackStrategyV49,
        FluctuationFeedbackStrategyV49, QualityFeedbackStrategyV49,
        CompositeFeedbackStrategyV49, FeedbackManagerV49,
        # Part 10 Classes
        DialogueSettingsV49, DialogStyleManagerV49,
        # Part 11 Classes
        StructuredErrorV49, ExceptionManagerV49,
        # Part 12, 13, 14 (DialogueGeneratorV49関連)
        DialogueGeneratorV49, GeneratorDependenciesV49,
        # Part 15 (main_v49)
        main_v49
    )
    # 型エイリアス
    PsychologicalPhaseV49EnumType: TypeAlias = PsychologicalPhaseV49
    EmotionalToneV49EnumType: TypeAlias = EmotionalToneV49
    FeedbackContextV49Type: TypeAlias = FeedbackContextV49
    GeneratorStateV49Type: TypeAlias = GeneratorStateV49
else:
    # 実行時は globals().get() で取得
    setup_logging = globals().get('setup_logging', lambda **kwargs: logging.getLogger())
    logger = globals().get('logger', logging.getLogger())
    ConfigProtocol = 'ConfigProtocol' # 文字列リテラル
    SettingsProtocol = 'SettingsProtocol'
    PsychologicalPhaseV49EnumType = globals().get('PsychologicalPhaseV49', enum.Enum)
    EmotionalToneV49EnumType = globals().get('EmotionalToneV49', enum.Enum)
    SubjectiveIntensityLevel = globals().get('SubjectiveIntensityLevel', enum.Enum)
    ScoreKeys = globals().get('ScoreKeys')
    DFRSMetricsV49 = globals().get('DFRSMetricsV49', enum.Enum)
    FeedbackContextV49Type = globals().get('FeedbackContextV49', dict)
    GeneratorStateV49Type = globals().get('GeneratorStateV49', dict)
    AppConfigV49 = globals().get('AppConfigV49') # テストヘルパーで使用
    # 他のクラスも同様に globals().get() で取得する想定


# --- テスト用ヘルパー関数 (v4.9α - Enum移行対応版) ---
def create_test_config_v49() -> ConfigProtocol: # type: ignore
    """テスト用のConfigProtocolモックオブジェクトを作成します。"""
    mock_config = mock.Mock(spec=ConfigProtocol) # type: ignore
    
    # AppConfigV49 の主要な属性とプロパティのデフォルト値を設定
    # Part 2 での AppConfigV49 の変更に合わせて調整
    mock_config.SYSTEM_VERSION = "TestNDGS_v4.9_Optimized"
    mock_config.API_KEY_ENV_VAR = "TEST_GENAI_API_KEY"
    mock_config.DEFAULT_MODEL = "models/gemini-2.5-flash-preview-04-17-thinking"
    mock_config.RPM_LIMIT = 1000 # テストでは高めに設定
    mock_config.MAX_RETRIES = 1
    mock_config.BASE_RETRY_DELAY = 0.01
    mock_config.MAX_RETRY_DELAY = 0.1
    mock_config.RATE_LIMIT_DELAY = 0.05
    mock_config.API_TIMEOUT = 10 # 秒
    mock_config.EVALUATION_TEMPERATURE = 0.1
    mock_config.INITIAL_CANDIDATE_COUNT = 2
    
    mock_config.base_output_dir = pathlib.Path("./test_output_v49_alpha")
    mock_config.config_dir = pathlib.Path("./test_configs")
    mock_config.resources_dir = mock_config.config_dir / "resources"
    mock_config.cache_dir = pathlib.Path("./test_cache/ndgs_v49_alpha")
    mock_config.persistent_cache_dir = mock_config.cache_dir / "persistent"
    
    mock_config.log_filename = "test_ndgs.log"
    mock_config.log_max_bytes = 1024 * 1024 # 1MB
    mock_config.log_backup_count = 1
    mock_config.filename_max_length = 100
    mock_config.enable_filelock = False # テスト中は通常False

    # プロパティのモック (Pydanticモデルを返すものは、そのモックを返す)
    # ExternalConfigsV49 のモック (Part 3で定義)
    mock_ext_configs = mock.Mock(spec=globals().get('ExternalConfigsV49'))
    mock_ext_configs.api_settings = mock.Mock(spec=globals().get('APISettingsV49'))
    mock_ext_configs.file_settings = mock.Mock(spec=globals().get('FileSettingsV49'))
    # ... 他のネストされた設定モデルのモックも同様に ...
    # feature_flags はブール値を持つPydanticモデルを想定
    mock_feature_flags = mock.Mock(spec=globals().get('FeatureFlagConfigV49'))
    for flag_attr in [f.name for f in getattr(globals().get('FeatureFlagConfigV49', object), 'model_fields', {}).values()]: # Pydantic V2
        setattr(mock_feature_flags, flag_attr, False) # デフォルトはFalse
    mock_feature_flags.dfrs_evaluation_enabled = True # テストで必要なものはTrueに
    mock_feature_flags.phase_tone_analysis_enabled = True
    mock_ext_configs.feature_flags = mock_feature_flags
    
    mock_ext_configs.dfrs_weights_config = mock.Mock(spec=globals().get('DFRSWeightsConfigV49'))
    mock_ext_configs.dfrs_weights_config.weights = {} # 空の重み
    
    # phase_tone_prompt_templates は PsychologicalPhaseV49.value をキーとする辞書
    # テスト用に最小限のテンプレートを設定
    phase_cls_for_test = globals().get('PsychologicalPhaseV49')
    tone_cls_for_test = globals().get('EmotionalToneV49')
    if phase_cls_for_test and tone_cls_for_test:
        mock_ext_configs.phase_tone_prompt_templates = {
            phase_cls_for_test.INTRODUCTION.value: { # type: ignore
                tone_cls_for_test.NEUTRAL.value: "Test intro neutral prompt." # type: ignore
            },
            phase_cls_for_test.DEVELOPMENT.value: { # type: ignore
                 tone_cls_for_test.TENSE.value: "Test development tense prompt." # type: ignore
            }
        }
        # phase_transition_matrix も PsychologicalPhaseV49.value をキーとする
        mock_ext_configs.phase_transition_matrix = {
            phase_cls_for_test.INTRODUCTION.value: {phase_cls_for_test.DEVELOPMENT.value: 0.8} # type: ignore
        }
    else:
        mock_ext_configs.phase_tone_prompt_templates = {}
        mock_ext_configs.phase_transition_matrix = {}

    mock_config.loaded_external_configs = mock_ext_configs
    
    # プロパティの返り値を設定
    type(mock_config).api_key = mock.PropertyMock(return_value="test_api_key_from_config")
    type(mock_config).generation_config = mock.PropertyMock(return_value={"temperature": 0.7, "candidate_count": 1})
    type(mock_config).safety_settings = mock.PropertyMock(return_value=None) # デフォルト
    type(mock_config).dfrs_weights = mock.PropertyMock(return_value={}) # 空の重み
    type(mock_config).subjectivity_keywords = mock.PropertyMock(return_value={}) # 空のデータ
    type(mock_config).fluctuation_patterns = mock.PropertyMock(return_value={}) # 空のデータ
    mock_config.analyzer_keywords_data = {"emotion_keywords": {}, "phase_keywords": {}}

    type(mock_config).temperature_config = mock.PropertyMock(return_value=mock.Mock(spec=globals().get('TemperatureStrategyConfigV49')))
    type(mock_config).feature_flags = mock.PropertyMock(return_value=mock_feature_flags) # 上で作成したモック
    type(mock_config).adaptation_config = mock.PropertyMock(return_value=mock.Mock(spec=globals().get('AdaptationStrategyConfigV49')))
    type(mock_config).feedback_config = mock.PropertyMock(return_value=mock.Mock(spec=globals().get('FeedbackStrategyConfigV49')))
    
    # メソッドのモック
    mock_config.load_external_configs = mock.MagicMock(return_value=True) # 成功を返す
    mock_config.update_from_args = mock.MagicMock()
    mock_config.initialize_base_directories = mock.MagicMock()
    
    mock_config._exception_manager = mock.Mock(spec=globals().get('ExceptionManagerV49'))
    return mock_config # type: ignore

def create_mock_settings_v49(config: Optional[ConfigProtocol] = None) -> SettingsProtocol: # type: ignore
    """テスト用のSettingsProtocolモックオブジェクトを作成します。"""
    mock_config_to_use = config if config else create_test_config_v49()
    mock_settings = mock.Mock(spec=SettingsProtocol) # type: ignore
    
    # DialogueSettingsV49 の主要な属性のデフォルト値を設定
    mock_settings.dialogue_mode = "auto"
    mock_settings.style_template = "standard"
    mock_settings.custom_style_file_path = None
    mock_settings.subjective_focus = True
    # subjective_intensity は SubjectiveIntensityLevel Enumメンバーを期待
    intensity_cls = globals().get('SubjectiveIntensityLevel')
    mock_settings.subjective_intensity = intensity_cls.MEDIUM if intensity_cls else "medium" # type: ignore
    
    mock_settings.dfrs_evaluation_enabled = True
    mock_settings.dfrs_evaluate_all_loops = True
    mock_settings.advanced_nlp_enabled = False # テストでは無効化推奨
    mock_settings.nlp_model_name = "ja_core_news_sm" # テスト用軽量モデル
    # ... (DialogueSettingsV49の他の属性も同様に設定) ...
    mock_settings.feedback_loops = 2
    mock_settings.min_feedback_loops = 1
    mock_settings.min_score_threshold = 4.0
    mock_settings.target_length = 500
    mock_settings.dfrs_for_initial_selection = True

    # プロパティのモック
    type(mock_settings).config = mock.PropertyMock(return_value=mock_config_to_use)
    type(mock_settings).enhanced_dfrs_weights = mock.PropertyMock(return_value={}) # 空の重み
    type(mock_settings).final_selection_weights = mock.PropertyMock(return_value={})
    type(mock_settings).initial_candidate_weights = mock.PropertyMock(return_value={})
    
    # メソッドのモック
    mock_settings.update_settings_based_on_mode = mock.MagicMock()
    mock_settings.update_from_args = mock.MagicMock()
    return mock_settings # type: ignore

def create_mock_dependencies_v49(
    config: Optional[ConfigProtocol] = None,
    settings: Optional[SettingsProtocol] = None
) -> 'GeneratorDependenciesV49': # type: ignore
    """テスト用のGeneratorDependenciesV49モックオブジェクトを作成します。"""
    mock_config_dep = config if config else create_test_config_v49()
    mock_settings_dep = settings if settings else create_mock_settings_v49(mock_config_dep)

    # GeneratorDependenciesV49 クラスを取得
    GeneratorDependencies_cls: Optional[Type['GeneratorDependenciesV49']] = _get_global_type('GeneratorDependenciesV49', type) # type: ignore
    if not GeneratorDependencies_cls:
        raise ImportError("GeneratorDependenciesV49 class not found for mocking.")

    # --- 依存コンポーネントのモック作成 ---
    # ExceptionManagerV49
    ExceptionManager_cls: Optional[Type['ExceptionManagerV49']] = _get_global_type('ExceptionManagerV49', type) # type: ignore
    mock_exception_manager = mock.Mock(spec=ExceptionManager_cls) if ExceptionManager_cls else mock.Mock()
    StructuredError_cls: Optional[Type['StructuredErrorV49']] = _get_global_type('StructuredErrorV49', type(Exception)) # type: ignore
    def safe_op_side_effect(op_name: str, func_to_run: Callable[..., Any], args: Optional[Tuple[Any, ...]]=None, kwargs: Optional[Dict[str, Any]]=None):
        try:
            return True, func_to_run(*(args or ()), **(kwargs or {})), None
        except Exception as e:
            err_obj = StructuredError_cls(original_exception=e, source=op_name) if StructuredError_cls else e # type: ignore
            return False, None, err_obj
    mock_exception_manager.safe_file_operation.side_effect = safe_op_side_effect
    mock_exception_manager.safe_api_call.side_effect = safe_op_side_effect
    mock_exception_manager.safe_nlp_processing.side_effect = safe_op_side_effect
    mock_exception_manager.log_error.side_effect = lambda e, *a, **kw: e if StructuredError_cls and isinstance(e, StructuredError_cls) else (StructuredError_cls(e) if StructuredError_cls else e) # type: ignore
    mock_exception_manager.is_retryable.return_value = False

    # ApiClientV49
    mock_api_client = mock.Mock(spec=globals().get('ApiClientProtocol'))
    mock_api_client.api_available = True # テストではAPI利用可能とする
    mock_api_client.generate_content.return_value = "Mocked API Response Text"
    mock_api_client.generate_content_with_candidates.return_value = ["Candidate A text", "Candidate B text"]

    # AdvancedDialogueAnalyzerV49
    mock_analyzer = mock.Mock(spec=globals().get('AnalyzerProtocol'))
    # analyze_and_get_results の返り値は、テストケースに応じて具体的に設定する
    mock_analyzer.analyze_and_get_results.return_value = {
        "analysis_summary_partial": {"dominant_phase": "development", "dominant_tone": "neutral"},
        "output_dialogue_content": {"blocks": []},
        "evaluation_metrics_partial": {"emotion_curve": [], "phase_timeline": []}
    }
    
    # SubjectivityFluctuationScorerV49
    mock_scorer = mock.Mock(spec=globals().get('ScorerProtocol'))
    mock_scorer.calculate_subjectivity_score.return_value = (0.5, {}) # (score, category_hits)
    mock_scorer.calculate_fluctuation_intensity.return_value = (0.2, {}) # (score, category_hits)

    # EnhancedDialogFlowEvaluatorV49
    mock_evaluator = mock.Mock(spec=globals().get('EvaluatorProtocol'))
    mock_evaluator.get_dfrs_scores_v49.return_value = {"csr": 4.0, "final_eodf_v49": 3.8} # 例

    # PhaseToneAdaptationStrategyV49
    mock_adaptation_strategy = mock.Mock(spec=globals().get('AdaptationStrategyProtocol'))
    mock_adaptation_strategy.enabled = False # デフォルトでは無効
    mock_adaptation_strategy.suggest_next_state.return_value = (None, None) # (phase, tone)

    # FeedbackManagerV49
    mock_feedback_manager = mock.Mock(spec=globals().get('FeedbackManagerProtocol'))
    mock_feedback_manager.get_feedback.return_value = "Mocked feedback message."
    mock_feedback_manager.composite_strategy = mock.Mock(spec=globals().get('FeedbackStrategyProtocol'))

    # PromptBuilderV49
    mock_prompt_builder = mock.Mock(spec=globals().get('PromptBuilderProtocol'))
    mock_prompt_builder.create_dialogue_prompt.return_value = "Mocked initial dialogue prompt."
    mock_prompt_builder.create_improvement_prompt.return_value = "Mocked improvement prompt."
    mock_prompt_builder.create_evaluation_prompt.return_value = "Mocked evaluation prompt."
    # extract_scores は静的メソッドなので、モックの仕方を変えるか、クラス自体をモックする必要がある場合も
    mock_prompt_builder.extract_scores = mock.Mock(return_value={}) # 空のスコア辞書

    # DialogStyleManagerV49
    mock_style_manager = mock.Mock(spec=globals().get('StyleManagerProtocol'))
    mock_style_manager.get_style_prompt_addition_text.return_value = " (Mocked style addition)"

    # DialogueManagerV49
    mock_dialogue_manager = mock.Mock(spec=globals().get('DialogueManagerProtocol'))
    mock_dialogue_manager.job_id = "test_job_123"
    mock_dialogue_manager.job_output_dir = mock_config_dep.base_output_dir / "test_job_123"
    mock_dialogue_manager.base_output_dir = mock_config_dep.base_output_dir
    mock_dialogue_manager.load_resume_state.return_value = None # デフォルトは再開データなし
    
    return GeneratorDependencies_cls( # type: ignore
        config=mock_config_dep,
        settings=mock_settings_dep,
        api_client=mock_api_client,
        analyzer=mock_analyzer,
        scorer=mock_scorer,
        evaluator=mock_evaluator,
        adaptation_strategy=mock_adaptation_strategy,
        feedback_manager=mock_feedback_manager,
        prompt_builder=mock_prompt_builder,
        style_manager=mock_style_manager,
        dialogue_manager=mock_dialogue_manager,
        exception_manager=mock_exception_manager
    )

def create_mock_feedback_context_v49(version: int = 1, **kwargs) -> FeedbackContextV49Type: # type: ignore
    """テスト用のFeedbackContextV49モックデータを作成します。PsychologicalPhaseV49に対応。"""
    FeedbackContext_cls: Optional[Type[FeedbackContextV49Type]] = _get_global_type('FeedbackContextV49', BaseModel if PYDANTIC_AVAILABLE else dict) # type: ignore
    Phase_cls: Optional[Type[PsychologicalPhaseV49EnumType]] = _get_global_type('PsychologicalPhaseV49', enum.EnumMeta) # type: ignore
    Tone_cls: Optional[Type[EmotionalToneV49EnumType]] = _get_global_type('EmotionalToneV49', enum.EnumMeta) # type: ignore
    LLMKeys_cls: Optional[Type[ScoreKeysLLMEnumType]] = getattr(_get_global_type('ScoreKeys'), 'LLM', None) # type: ignore
    DFRSMetrics_cls: Optional[Type[DFRSMetricsV49EnumType]] = _get_global_type('DFRSMetricsV49', enum.EnumMeta) # type: ignore

    if not (FeedbackContext_cls and Phase_cls and Tone_cls and LLMKeys_cls and DFRSMetrics_cls):
        raise ImportError("FeedbackContextV49または関連Enumのロードに失敗しました。")

    # デフォルトの位相とトーンを新しいEnumメンバーで設定
    default_phase = kwargs.get("p_int", Phase_cls.DEVELOPMENT) # 例: DEVELOPMENT
    default_tone = kwargs.get("t_int", Tone_cls.NEUTRAL)
    default_inferred_phase = kwargs.get("p_inf", Phase_cls.DEVELOPMENT)
    default_inferred_tone = kwargs.get("t_inf", Tone_cls.NEUTRAL)

    # 文字列で渡された場合もEnumメンバーに変換 (安全のため)
    if isinstance(default_phase, str): default_phase = Phase_cls(default_phase)
    if isinstance(default_tone, str): default_tone = Tone_cls(default_tone)
    if isinstance(default_inferred_phase, str): default_inferred_phase = Phase_cls(default_inferred_phase)
    if isinstance(default_inferred_tone, str): default_inferred_tone = Tone_cls(default_inferred_tone)

    data_for_model: Dict[str, Any] = {
        "version": version,
        "dialogue_text": kwargs.get("text", f"Test dialogue text for version {version}."),
        "intended_phase": default_phase,
        "intended_tone": default_tone,
        "inferred_phase": default_inferred_phase,
        "inferred_tone": default_inferred_tone,
        "dfrs_scores": kwargs.get("dfrs", {DFRSMetrics_cls.FINAL_EODF_V49.value: 3.5}),
        "llm_scores": kwargs.get("llm", {LLMKeys_cls.OVERALL.value: 3.5}) # type: ignore
    }
    return FeedbackContext_cls.model_validate(data_for_model) # type: ignore

# --- Test Classes (v4.9α - PsychologicalPhaseV49移行対応版) ---

class TestAppConfigV49(unittest.TestCase):
    # (既存のテストケースを維持しつつ、必要に応じてAppConfigV49の変更点をテスト)
    def test_config_initialization_and_defaults(self):
        app_config = AppConfigV49() # type: ignore
        self.assertIsNotNone(app_config)
        self.assertEqual(app_config.DEFAULT_MODEL, "models/gemini-2.5-flash-preview-04-17-thinking") # Part 2のデフォルト値
        # 他のデフォルト値もテスト

    def test_load_external_configs_from_mock_yaml(self):
        # YAMLロードのテストは、実際のYAMLファイルまたはモックされた `load_yaml_file` を使用
        pass # 実装は省略

class TestPromptBuilderV49(unittest.TestCase):
    def setUp(self):
        self.mock_config = create_test_config_v49()
        self.mock_feedback_strategy = mock.Mock(spec=globals().get('FeedbackStrategyProtocol'))
        self.mock_style_manager = mock.Mock(spec=globals().get('StyleManagerProtocol'))
        self.builder = PromptBuilderV49(self.mock_config, self.mock_feedback_strategy, self.mock_style_manager) # type: ignore
        self.phase_cls = self.builder.PsychologicalPhaseV49_cls
        self.tone_cls = self.builder.EmotionalToneV49_cls
        self.intensity_cls = self.builder.SubjectiveIntensityLevel_cls

        # テスト用の設定 (phase_tone_prompt_templates)
        if self.mock_config.loaded_external_configs and self.phase_cls and self.tone_cls:
            self.mock_config.loaded_external_configs.phase_tone_prompt_templates = { # type: ignore
                self.phase_cls.INTRODUCTION.value: { # type: ignore
                    self.tone_cls.NEUTRAL.value: "導入部、ニュートラルな指示です。" # type: ignore
                },
                self.phase_cls.CONFLICT_CRISIS.value: { # type: ignore
                    self.tone_cls.TENSE.value: "葛藤/危機、緊張感のある指示です。" # type: ignore
                }
            }
        # テスト用のキーワードデータ (subjectivity_keywords)
        if self.mock_config.loaded_external_configs and self.builder.SubjectivityKeywordEntry_cls and self.builder.SubjectivityCategoryV49_cls:
            cat_emo_pos = self.builder.SubjectivityCategoryV49_cls.EMOTION_POSITIVE # type: ignore
            self.mock_config.subjectivity_keywords = { # type: ignore
                cat_emo_pos: [ # type: ignore
                    self.builder.SubjectivityKeywordEntry_cls(keyword="嬉しい", intensity=0.9, category=cat_emo_pos, related_phases=[self.phase_cls.RESOLUTION_CONCLUSION]), # type: ignore
                    self.builder.SubjectivityKeywordEntry_cls(keyword="楽しい", intensity=0.8, category=cat_emo_pos, related_phases=[self.phase_cls.DEVELOPMENT]) # type: ignore
                ]
            }


    def test_get_enum_member_from_value_psychological_phase(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49_cls is not loaded.")
        # 正しい値
        self.assertEqual(self.builder._get_enum_member_from_value(self.phase_cls, "introduction"), self.phase_cls.INTRODUCTION) # type: ignore
        self.assertEqual(self.builder._get_enum_member_from_value(self.phase_cls, "CONFLICT_CRISIS"), self.phase_cls.CONFLICT_CRISIS) # type: ignore
        # _missing_ で処理されるべき値 (小文字、ハイフンなど)
        self.assertEqual(self.builder._get_enum_member_from_value(self.phase_cls, "internal-processing"), self.phase_cls.INTERNAL_PROCESSING) # type: ignore
        self.assertEqual(self.builder._get_enum_member_from_value(self.phase_cls, "climax_turning_point"), self.phase_cls.CLIMAX_TURNING_POINT) # type: ignore
        # 存在しない値 -> UNKNOWN にフォールバック
        self.assertEqual(self.builder._get_enum_member_from_value(self.phase_cls, "non_existent_phase"), self.phase_cls.UNKNOWN) # type: ignore
        # Enumメンバーインスタンスを渡した場合
        self.assertEqual(self.builder._get_enum_member_from_value(self.phase_cls, self.phase_cls.RESOLUTION_CONCLUSION), self.phase_cls.RESOLUTION_CONCLUSION) # type: ignore

    def test_select_relevant_keywords_with_new_phases(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49_cls is not loaded.")
        # RESOLUTION_CONCLUSION にマッチするキーワードが選択されるはず
        selected = self.builder._select_relevant_keywords(self.phase_cls.RESOLUTION_CONCLUSION, None, top_k=1) # type: ignore
        self.assertEqual(len(selected), 1)
        if selected: self.assertEqual(selected[0].keyword, "嬉しい")

    def test_get_phase_tone_instruction_v49_with_new_phases(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49_cls is not loaded.")
        self.assertIsNotNone(self.tone_cls, "EmotionalToneV49_cls is not loaded.")
        # INTRODUCTION と NEUTRAL の組み合わせ
        instruction = self.builder._get_phase_tone_instruction_v49(self.phase_cls.INTRODUCTION, self.tone_cls.NEUTRAL) # type: ignore
        self.assertIn("導入部、ニュートラルな指示です。", instruction)
        # CONFLICT_CRISIS と TENSE の組み合わせ
        instruction_conflict = self.builder._get_phase_tone_instruction_v49(self.phase_cls.CONFLICT_CRISIS, self.tone_cls.TENSE) # type: ignore
        self.assertIn("葛藤/危機、緊張感のある指示です。", instruction_conflict)
        # 存在しない組み合わせ
        instruction_none = self.builder._get_phase_tone_instruction_v49(self.phase_cls.ACTION_EVENT, self.tone_cls.CALM) # type: ignore
        self.assertEqual(instruction_none.strip(), "")

    def test_create_dialogue_prompt_with_new_phase(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49_cls is not loaded.")
        # (テストデータ準備)
        char_a = {"name": "Alice"}
        char_b = {"name": "Bob"}
        scene = {"name": "Test Scene"}
        mock_settings = create_mock_settings_v49(self.mock_config)
        mock_settings.subjective_focus = True # 主観描写を有効にする
        
        prompt = self.builder.create_dialogue_prompt(
            char_a, char_b, scene, 500, mock_settings,
            phase_val=self.phase_cls.DEVELOPMENT, # type: ignore
            tone_val=self.tone_cls.TENSE # type: ignore
        )
        self.assertIsInstance(prompt, str)
        self.assertIn(self.phase_cls.DEVELOPMENT.value, prompt.lower()) # type: ignore # プロンプト内に位相名が含まれるか（デバッグ用）
        self.assertIn("development tense prompt", prompt) # _get_phase_tone_instruction_v49 からの指示が含まれるか

    def test_create_improvement_prompt_with_new_phase_in_context(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49_cls is not loaded.")
        mock_settings = create_mock_settings_v49(self.mock_config)
        # FeedbackContextV49 に新しいEnumメンバーを設定
        feedback_context_data = create_mock_feedback_context_v49(
            p_int=self.phase_cls.CLIMAX_TURNING_POINT, # type: ignore
            t_int=self.tone_cls.EXCITED # type: ignore
        )
        self.mock_feedback_strategy.generate.return_value = "Mocked strategy feedback." # type: ignore
        
        prompt = self.builder.create_improvement_prompt(
            "Previous dialogue.", "Previous evaluation.", feedback_context_data, mock_settings
        )
        self.assertIsInstance(prompt, str)
        self.assertIn(self.phase_cls.CLIMAX_TURNING_POINT.value, prompt) # type: ignore
        self.assertIn(self.tone_cls.EXCITED.value, prompt) # type: ignore

class TestSubjectivityFluctuationScorerV49(unittest.TestCase):
    # このクラスは PsychologicalPhaseV49 を直接使用しないため、既存のテストを維持。
    # ただし、依存するキーワードファイルの内容が最新のEnum体系を反映しているかは間接的に影響する。
    # (テストケースの例 - 変更なし)
    def setUp(self):
        self.mock_config = create_test_config_v49()
        # テスト用にキーワードデータをモックコンフィグに設定
        # subjectivity_keywords と fluctuation_patterns を設定
        subj_cat_cls = _get_global_type('SubjectivityCategoryV49', enum.EnumMeta)
        subj_entry_cls = _get_global_type('SubjectivityKeywordEntryV49', BaseModel if PYDANTIC_AVAILABLE else type)
        fluc_cat_cls = _get_global_type('FluctuationCategoryV49', enum.EnumMeta)
        fluc_entry_cls = _get_global_type('FluctuationPatternEntryV49', BaseModel if PYDANTIC_AVAILABLE else type)

        if subj_cat_cls and subj_entry_cls:
            self.mock_config.subjectivity_keywords = { # type: ignore
                subj_cat_cls.EMOTION_POSITIVE: [ # type: ignore
                    subj_entry_cls(keyword="嬉しい", intensity=0.9, category=subj_cat_cls.EMOTION_POSITIVE) # type: ignore
                ]
            }
        if fluc_cat_cls and fluc_entry_cls:
             self.mock_config.fluctuation_patterns = { # type: ignore
                fluc_cat_cls.VERBAL_HESITATION: [ # type: ignore
                    fluc_entry_cls(pattern="えっと", intensity=0.4, category=fluc_cat_cls.VERBAL_HESITATION, use_regex=False) # type: ignore
                ]
            }
        self.scorer = SubjectivityFluctuationScorerV49(self.mock_config) # type: ignore

    def test_calculate_subjectivity_score(self):
        score, _ = self.scorer.calculate_subjectivity_score("本当に嬉しい")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

class TestEnhancedDialogFlowEvaluatorV49(unittest.TestCase):
    def setUp(self):
        self.mock_config = create_test_config_v49()
        self.mock_scorer = mock.Mock(spec=globals().get('ScorerProtocol'))
        self.evaluator = EnhancedDialogFlowEvaluatorV49(self.mock_config, self.mock_scorer) # type: ignore
        self.phase_cls = self.evaluator.PsychologicalPhase_cls # Evaluator内でロードされるはず
        self.tone_cls = globals().get('EmotionalToneV49')

        # テスト用の位相遷移マトリックス (新しいEnumの.value文字列をキーとして使用)
        if self.mock_config.loaded_external_configs and self.phase_cls:
            self.mock_config.loaded_external_configs.phase_transition_matrix = { # type: ignore
                self.phase_cls.INTRODUCTION.value: {self.phase_cls.DEVELOPMENT.value: 0.9}, # type: ignore
                self.phase_cls.DEVELOPMENT.value: {self.phase_cls.CONFLICT_CRISIS.value: 0.8}, # type: ignore
                # ... 他の必要な遷移 ...
            }

    def test_get_dfrs_scores_v49_with_new_phase(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49 class not loaded in Evaluator.")
        # analyzer_results のモック (phase_timeline_data に新しいEnumの.value文字列を含む)
        mock_analyzer_results = {
            "output_dialogue_content": {"blocks": [{"type": "description", "text": "テストテキスト"}]},
            "evaluation_metrics_partial": {
                "phase_timeline": [
                    {"block_index": 0, "phase": self.phase_cls.INTRODUCTION.value, "confidence": 0.8} # type: ignore
                ],
                "emotion_curve": []
            }
        }
        scores = self.evaluator.get_dfrs_scores_v49(
            dialogue_text="テストテキスト",
            analyzer_results=mock_analyzer_results,
            intended_phase=self.phase_cls.INTRODUCTION # type: ignore
        )
        self.assertIsInstance(scores, dict)
        # キャッシュキーが新しいEnumメンバーの.valueで生成されることの検証は難しいが、
        # 少なくともエラーなく実行されることを確認
        self.assertIn(DFRSMetricsV49.FINAL_EODF_V49.value, scores) # type: ignore

    def test_calculate_phase_transition_naturalness_with_new_phases(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49 class not loaded in Evaluator.")
        # 位相タイムライン (新しいEnumの.value文字列を使用)
        phase_timeline = [
            {"phase": self.phase_cls.INTRODUCTION.value, "confidence": 0.9}, # type: ignore
            {"phase": self.phase_cls.DEVELOPMENT.value, "confidence": 0.8}, # type: ignore
            {"phase": self.phase_cls.CONFLICT_CRISIS.value, "confidence": 0.85} # type: ignore
        ]
        ptn_score = self.evaluator.calculate_phase_transition_naturalness(phase_timeline)
        self.assertGreaterEqual(ptn_score, 0.0)
        self.assertLessEqual(ptn_score, 1.0)
        # マトリックスに基づく具体的なスコアをアサーションすることも可能

    def test_calculate_phase_tone_alignment_with_new_phase(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49 class not loaded in Evaluator.")
        self.assertIsNotNone(self.tone_cls, "EmotionalToneV49 class not loaded.")
        phase_timeline = [{"phase": self.phase_cls.DEVELOPMENT.value, "confidence": 0.9}] # type: ignore
        emotion_curve = [{"tone": self.tone_cls.TENSE.value, "strength": 0.8}] # type: ignore
        
        phase_align, tone_align = self.evaluator._calculate_phase_tone_alignment(
            intended_phase=self.phase_cls.DEVELOPMENT, # type: ignore
            intended_tone=self.tone_cls.TENSE, # type: ignore
            phase_timeline_data=phase_timeline,
            emotion_curve_data=emotion_curve
        )
        self.assertEqual(phase_align, 1.0)
        self.assertEqual(tone_align, 1.0)

class TestPhaseToneAdaptationStrategyV49(unittest.TestCase):
    def setUp(self):
        self.mock_config = create_test_config_v49()
        self.mock_exception_manager = mock.Mock(spec=globals().get('ExceptionManagerV49'))
        self.strategy = PhaseToneAdaptationStrategyV49(self.mock_config, self.mock_exception_manager) # type: ignore
        self.phase_cls = self.strategy.PsychologicalPhase_cls
        self.tone_cls = self.strategy.EmotionalTone_cls

    def test_get_state_key_with_new_phases(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49_cls is not loaded in Strategy.")
        key = self.strategy._get_state_key(self.phase_cls.INTERNAL_PROCESSING, self.tone_cls.NEUTRAL) # type: ignore
        self.assertEqual(key, f"{self.phase_cls.INTERNAL_PROCESSING.value}:{self.tone_cls.NEUTRAL.value}") # type: ignore

    def test_parse_state_key_with_new_phase_values(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49_cls is not loaded in Strategy.")
        # 新しいEnumの .value 文字列からのパース
        phase, tone = self.strategy._parse_state_key(f"{self.phase_cls.ACTION_EVENT.value}:{self.tone_cls.DETERMINED.value}") # type: ignore
        self.assertEqual(phase, self.phase_cls.ACTION_EVENT) # type: ignore
        self.assertEqual(tone, self.tone_cls.DETERMINED) # type: ignore
        # _missing_ で処理される可能性のある古い形式のキーもテスト (Enum側の_missing_実装に依存)
        phase_old, tone_old = self.strategy._parse_state_key("intro:neutral") # 仮の古いキー
        self.assertIsNotNone(phase_old) # UNKNOWN または INTRODUCTION になるはず
        self.assertIsNotNone(tone_old)  # NEUTRAL になるはず

    def test_get_random_phase_tone_excludes_unknown_phase(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49_cls is not loaded in Strategy.")
        # 多数回実行して UNKNOWN が選択されないことを確認 (確率的テスト)
        for _ in range(50):
            p, _ = self.strategy._get_random_phase_tone()
            if p: # Noneでない場合
                self.assertNotEqual(p, self.phase_cls.UNKNOWN) # type: ignore

class TestFeedbackStrategiesAndManagerV49(unittest.TestCase):
    def setUp(self):
        self.mock_config = create_test_config_v49()
        # FeedbackContextV49 で使用するEnumクラス
        self.phase_cls = _get_global_type('PsychologicalPhaseV49', enum.EnumMeta)
        self.tone_cls = _get_global_type('EmotionalToneV49', enum.EnumMeta)
        # DFRSMetricsV49 も必要
        self.dfrs_metrics_cls = _get_global_type('DFRSMetricsV49', enum.EnumMeta)


    def test_phase_tone_feedback_strategy_with_new_phase(self):
        self.assertIsNotNone(self.phase_cls, "PsychologicalPhaseV49 class not loaded.")
        self.assertIsNotNone(self.tone_cls, "EmotionalToneV49 class not loaded.")
        self.assertIsNotNone(self.dfrs_metrics_cls, "DFRSMetricsV49 class not loaded.")

        strategy = PhaseToneFeedbackStrategyV49(self.mock_config) # type: ignore
        # FeedbackContextV49 に新しいEnumメンバーを設定
        # (create_mock_feedback_context_v49 が新しいEnumを返すように修正済みと仮定)
        context = create_mock_feedback_context_v49(
            p_int=self.phase_cls.CONFLICT_CRISIS, # type: ignore
            p_inf=self.phase_cls.DEVELOPMENT, # type: ignore
            t_int=self.tone_cls.ANGRY, # type: ignore
            t_inf=self.tone_cls.TENSE, # type: ignore
            dfrs={
                self.dfrs_metrics_cls.PHASE_ALIGNMENT.value: 1.5, # 低いスコア (0-5スケール) type: ignore
                self.dfrs_metrics_cls.TONE_ALIGNMENT.value: 2.0   # 低いスコア type: ignore
            }
        )
        feedback = strategy.generate(context)
        self.assertIn(self.phase_cls.CONFLICT_CRISIS.value, feedback) # type: ignore
        self.assertIn(self.phase_cls.DEVELOPMENT.value, feedback) # type: ignore
        self.assertIn("位相のズレ", feedback)

class TestDialogueManagerV49(unittest.TestCase):
    # (既存のテストケースを維持しつつ、GeneratorStateV49内の位相情報が
    #  新しいEnumメンバーで正しくシリアライズ・デシリアライズされるか再確認)
    def setUp(self):
        self.mock_config = create_test_config_v49()
        # テスト用の一時ディレクトリを作成
        self.test_dir = pathlib.Path("./temp_test_dialogue_manager_output")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.mock_config.base_output_dir = self.test_dir # モックコンフィグの出力先を変更
        
        DialogueManager_cls: Optional[Type['DialogueManagerV49']] = _get_global_type('DialogueManagerV49', type) # type: ignore
        ExceptionManager_cls: Optional[Type['ExceptionManagerV49']] = _get_global_type('ExceptionManagerV49', type) # type: ignore
        if not DialogueManager_cls or not ExceptionManager_cls:
            self.skipTest("DialogueManagerV49 or ExceptionManagerV49 class not found.")
        
        self.mock_exception_manager = mock.Mock(spec=ExceptionManager_cls)
        # safe_file_operation のモック (create_mock_dependencies_v49から流用)
        def safe_op_mock(name, func, args=None, kwargs=None): # type: ignore
            try: return True, func(*(args or ()), **(kwargs or {})), None
            except Exception as e_safe: return False, None, e_safe
        self.mock_exception_manager.safe_file_operation.side_effect=safe_op_mock
        self.mock_exception_manager.log_error.side_effect = lambda e, *a, **kw: e

        self.job_id = "test_job_for_dm"
        self.manager = DialogueManager_cls(self.job_id, self.mock_config, self.mock_exception_manager) # type: ignore
        self.mock_settings = create_mock_settings_v49(self.mock_config)
        self.manager.initialize_directories(self.mock_settings) # type: ignore

    def tearDown(self):
        # テスト用ディレクトリをクリーンアップ
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)

    def test_save_load_resume_state_with_new_phase(self):
        GenState_cls: Optional[Type[GeneratorStateV49Type]] = _get_global_type('GeneratorStateV49', type(BaseModel) if PYDANTIC_AVAILABLE else dict) # type: ignore
        InData_cls = _get_global_type('InputDataV49', type(BaseModel) if PYDANTIC_AVAILABLE else dict)
        Char_cls = _get_global_type('CharacterV49', type(BaseModel) if PYDANTIC_AVAILABLE else dict)
        Scene_cls = _get_global_type('SceneInfoV49', type(BaseModel) if PYDANTIC_AVAILABLE else dict)
        VerState_cls = _get_global_type('VersionStateV49', type(BaseModel) if PYDANTIC_AVAILABLE else dict)
        Phase_cls_for_test_dm = _get_global_type('PsychologicalPhaseV49', enum.EnumMeta)

        if not (GenState_cls and InData_cls and Char_cls and Scene_cls and VerState_cls and Phase_cls_for_test_dm):
            self.skipTest("Required Pydantic models or Enums for GeneratorStateV49 test not found.")

        char_a_data = Char_cls(name="Character Alpha") # type: ignore
        char_b_data = Char_cls(name="Character Beta") # type: ignore
        scene_data = Scene_cls(name="Test Scene Alpha") # type: ignore
        input_data_obj = InData_cls(characterA=char_a_data, characterB=char_b_data, sceneInfo=scene_data) # type: ignore
        
        original_state = GenState_cls( # type: ignore
            job_id=self.job_id,
            system_version="test_v1",
            model_name="test_model_dm",
            start_time=datetime.now(timezone.utc),
            input_data=input_data_obj,
            target_length=1000,
            settings_snapshot={"test_setting": "value"},
            current_loop=2,
            current_intended_phase=Phase_cls_for_test_dm.RESOLUTION_CONCLUSION, # 新しいEnumメンバー
            versions=[
                VerState_cls(version_id=0, generated_text="Version 0 text"), # type: ignore
                VerState_cls(version_id=1, generated_text="Version 1 text") # type: ignore
            ]
        )
        
        save_ok = self.manager.save_resume_state(original_state)
        self.assertTrue(save_ok, "Resume state save failed.")
        
        loaded_state = self.manager.load_resume_state()
        self.assertIsNotNone(loaded_state, "Loaded resume state is None.")
        if loaded_state: # mypyのためのNoneチェック
            self.assertEqual(loaded_state.job_id, self.job_id)
            self.assertEqual(loaded_state.current_loop, 2)
            self.assertIsInstance(loaded_state.versions, list)
            self.assertEqual(len(loaded_state.versions), 2)
            self.assertEqual(loaded_state.versions[1].generated_text, "Version 1 text")
            self.assertEqual(loaded_state.current_intended_phase, Phase_cls_for_test_dm.RESOLUTION_CONCLUSION) # Enumメンバーとして復元される

class TestExceptionManagerV49(unittest.TestCase):
    # このクラスは PsychologicalPhaseV49 を直接使用しないため、変更なし。
    pass

def run_tests_v49() -> int:
    """ユニットテスト実行関数 (v4.9α 最適化・Enum移行対応版)"""
    # setup_logging は Part 0 で定義済みと仮定
    global logger # type: ignore
    if 'logger' not in globals() or not isinstance(globals()['logger'], logging.Logger) or not globals()['logger'].hasHandlers(): # type: ignore
        logger = setup_logging(level=logging.DEBUG, debug_mode=True) # テスト時はデバッグモード推奨
    
    logger.info("NDGS v4.9α (Optimized) Unit Tests Starting...")
    
    test_suite = unittest.TestSuite()
    test_loader = unittest.TestLoader()
    
    # テスト対象のクラスリスト
    # プロジェクトの構造に合わせて、実際のテストクラスを動的に見つける方が良い場合もある
    test_classes_to_run = [
        TestAppConfigV49,
        TestPromptBuilderV49,
        TestSubjectivityFluctuationScorerV49,
        TestEnhancedDialogFlowEvaluatorV49,
        TestPhaseToneAdaptationStrategyV49,
        TestFeedbackStrategiesAndManagerV49,
        TestDialogueManagerV49,
        TestExceptionManagerV49,
        # TODO: TestDialogueGeneratorV49 も PsychologicalPhaseV49 に関連するテストケースがあれば追加・修正
    ]
    
    for test_class in test_classes_to_run:
        # globals() にクラスが存在するか確認 (単一ファイル実行の場合)
        if test_class.__name__ in globals() and isinstance(globals()[test_class.__name__], type) and issubclass(globals()[test_class.__name__], unittest.TestCase):
            test_suite.addTest(test_loader.loadTestsFromTestCase(test_class))
        else:
            logger.warning(f"Test class '{test_class.__name__}' not found or not a TestCase in current scope. Skipping.")
            
    test_runner = unittest.TextTestRunner(verbosity=2) # verbosity=2 で詳細表示
    test_result = test_runner.run(test_suite)
    
    logger.info("NDGS v4.9α (Optimized) Unit Tests Finished.")
    return 0 if test_result.wasSuccessful() else 1

# =============================================================================
# -- Part 16 終了点
# =============================================================================
# =============================================================================
# -- Part 17: Main Execution Function (v4.9α - Gemini提案FIX版)
# =============================================================================
def main_v49() -> int:
    """メイン処理関数"""
    start_time = datetime.now(timezone.utc)
    exit_code = 0
    
    # setup_logging は Part 0 または Part 1a で定義済みと仮定
    # create_argument_parser_v49 も同様
    # AppConfigV49, ExceptionManagerV49 なども同様

    parser = create_argument_parser_v49() # type: ignore
    args = parser.parse_args()

    # 初期ロガー設定 (AppConfig ロード前)
    initial_log_level_str = args.log_level or os.environ.get('LOG_LEVEL', 'INFO')
    initial_log_level = logging.getLevelName(initial_log_level_str.upper())
    if not isinstance(initial_log_level, int):
        # この段階では main logger がまだ設定されていない可能性があるため、標準エラー出力
        print(f"警告: 不正なログレベル指定 '{initial_log_level_str}'。INFOを使用します。", file=sys.stderr)
        initial_log_level = logging.INFO
    
    initial_log_file_arg = args.log_file # コマンドライン引数から取得
    # この時点では AppConfigV49 の LOG_FILENAME_DEFAULT_BOOTSTRAP を使用
    # AppConfigV49 がロードされた後、再度設定される
    
    global logger # グローバルスコープの logger を更新
    logger = setup_logging( # type: ignore
        level=initial_log_level,
        log_file=initial_log_file_arg if initial_log_file_arg is not None else LOG_FILENAME_DEFAULT_BOOTSTRAP, # type: ignore
        max_bytes=LOG_MAX_BYTES_DEFAULT_BOOTSTRAP, # type: ignore
        backup_count=LOG_BACKUP_COUNT_DEFAULT_BOOTSTRAP, # type: ignore
        console_debug_mode=(initial_log_level == logging.DEBUG), # 初期はコマンドライン引数レベルに合わせる
        force_reload_handlers=True # ★★★修正点★★★
    )

    system_version_for_log = getattr(globals().get('AppConfigV49'), 'SYSTEM_VERSION', "NDGS v4.9α (Optimized)")
    logger.info(f"=== {system_version_for_log} 実行開始 ===")

    if args.run_tests:
        logger.info("テスト実行モード...")
        run_tests_func = globals().get('run_tests_v49')
        if callable(run_tests_func):
            exit_code = run_tests_func() # type: ignore
        else:
            logger.error("テスト実行関数 'run_tests_v49' が見つかりません。")
            exit_code = 1
        logger.info(f"テスト終了 (Exit Code: {exit_code})。")
        logging.shutdown()
        return exit_code

    app_config: Optional['AppConfigV49'] = None # type: ignore
    exception_manager: Optional['ExceptionManagerV49'] = None # type: ignore
    job_settings: Optional['DialogueSettingsV49'] = None # type: ignore
    validated_input_data: Optional['InputDataV49'] = None # type: ignore
    dialogue_manager_instance: Optional['DialogueManagerV49'] = None # type: ignore
    api_instance: Optional['ApiClientV49'] = None # type: ignore
    scorer_instance: Optional['SubjectivityFluctuationScorerV49'] = None # type: ignore
    analyzer_instance: Optional['AdvancedDialogueAnalyzerV49'] = None # type: ignore
    evaluator_instance: Optional['EnhancedDialogFlowEvaluatorV49'] = None # type: ignore
    adaptation_instance: Optional['PhaseToneAdaptationStrategyV49'] = None # type: ignore
    feedback_manager_instance: Optional['FeedbackManagerV49'] = None # type: ignore
    style_manager_instance: Optional['DialogStyleManagerV49'] = None # type: ignore
    prompt_builder_instance: Optional['PromptBuilderV49'] = None # type: ignore
    dependencies: Optional['GeneratorDependenciesV49'] = None # type: ignore
    generator: Optional['DialogueGeneratorV49'] = None # type: ignore

    try:
        # 1. AppConfig と ExceptionManager
        try:
            AppConfig_cls = globals().get('AppConfigV49')
            ExceptionManager_cls = globals().get('ExceptionManagerV49')
            if not AppConfig_cls or not ExceptionManager_cls:
                logger.critical("AppConfigV49 または ExceptionManagerV49 クラスが見つかりません。")
                return 1
            app_config = AppConfig_cls() # type: ignore
            exception_manager = ExceptionManager_cls(app_config, logger) # type: ignore
            app_config._exception_manager = exception_manager # type: ignore
        except Exception as e_core_init:
            logger.critical(f"コア設定初期化エラー: {e_core_init}", exc_info=True)
            return 1

        # 2. 設定ファイルのロードと適用
        try:
            logger.info("設定ファイルのロードと適用...")
            if app_config.external_config_enabled: # type: ignore
                ok_load_ext, _, err_load_ext = exception_manager.safe_file_operation( # type: ignore
                    "外部設定ロード", app_config.load_external_configs # type: ignore
                )
                if not ok_load_ext: logger.error(f"外部設定ロード失敗: {err_load_ext}。")
            else: logger.info("外部設定ロードは無効。")
            
            app_config.update_from_args(args) # type: ignore

            # AppConfig の値に基づいてロガーを再設定
            log_level_str_after_config = args.log_level if args.log_level is not None else app_config.log_level_str # type: ignore
            final_log_level = logging.getLevelName(log_level_str_after_config.upper())
            if not isinstance(final_log_level, int): final_log_level = logger.getEffectiveLevel()
            
            logger = setup_logging( # type: ignore
                level=final_log_level,
                log_file=str(app_config.log_filename), # type: ignore
                max_bytes=app_config.log_max_bytes, # type: ignore
                backup_count=app_config.log_backup_count, # type: ignore
                console_debug_mode=(final_log_level == logging.DEBUG), # 再設定時は最終レベルに合わせる
                force_reload_handlers=True # ★★★修正点★★★
            )
            logger.info(f"ロガー再設定完了 (Level: {logging.getLevelName(final_log_level)}, File: {app_config.log_filename})。") # type: ignore

            DialogueSettings_cls = globals().get('DialogueSettingsV49')
            if not DialogueSettings_cls:
                logger.critical("DialogueSettingsV49 クラスが見つかりません。")
                return 1
            job_settings = DialogueSettings_cls(app_config) # type: ignore
            job_settings.update_from_args(args) # type: ignore

            base_output_dir_override = pathlib.Path(args.appconfig_base_output_dir).resolve() if hasattr(args, 'appconfig_base_output_dir') and args.appconfig_base_output_dir is not None else None
            app_config.initialize_base_directories(base_output_dir_override=base_output_dir_override) # type: ignore
            logger.info("設定初期化完了。")
        except Exception as e_settings_apply:
            logger.critical(f"設定ロード・適用エラー: {e_settings_apply}", exc_info=True)
            if exception_manager: exception_manager.log_error(e_settings_apply, "main_settings_apply_critical")
            return 1

        # 3. 入力データのロードと検証
        input_data_json_path_str = args.data_file_opt or args.data_file
        if not input_data_json_path_str:
            logger.error("入力データファイル未指定。")
            parser.print_usage(sys.stderr)
            return 1
        input_data_file = pathlib.Path(input_data_json_path_str).resolve()
        if not input_data_file.is_file():
            logger.error(f"入力ファイル未発見: {input_data_file}")
            return 1
        
        InputDataV49_cls = globals().get('InputDataV49')
        if not InputDataV49_cls:
            logger.critical("InputDataV49 クラスが見つかりません。")
            return 1
        try:
            logger.info(f"入力ファイル検証: {input_data_file}")
            ok_load, raw_data, err_load = exception_manager.safe_file_operation( # type: ignore
                "入力JSONロード", load_json_file, args=(input_data_file,) # load_json_file はPart1aで定義
            )
            if not (ok_load and raw_data is not None):
                logger.error(f"入力JSON読込失敗: {err_load}")
                return 1
            validated_input_data = InputDataV49_cls.model_validate(raw_data) # type: ignore
            logger.info("入力データ検証完了。")
        except ValidationError as val_err: # type: ignore
            logger.error(f"入力データ検証失敗: {val_err}", exc_info=True)
            return 1
        except Exception as e_input:
            logger.critical(f"入力データ処理中エラー: {e_input}", exc_info=True)
            if exception_manager: exception_manager.log_error(e_input, "main_input_data_critical")
            return 1

        if args.check_only:
            logger.info("データチェックのみモード終了。")
            return 0

        # 4. 依存コンポーネントのインスタンス化
        try:
            logger.info("依存コンポーネント初期化...")
            job_id_base = args.job_id or f"{app_config.DEFAULT_JOB_ID_PREFIX}{int(start_time.timestamp())}" # type: ignore

            # 各コンポーネントクラスの取得
            DialogueManager_cls = globals().get('DialogueManagerV49')
            ApiClient_cls = globals().get('ApiClientV49')
            Scorer_cls = globals().get('SubjectivityFluctuationScorerV49')
            Analyzer_cls = globals().get('AdvancedDialogueAnalyzerV49')
            Evaluator_cls = globals().get('EnhancedDialogFlowEvaluatorV49')
            Adaptation_cls = globals().get('PhaseToneAdaptationStrategyV49')
            FeedbackManager_cls = globals().get('FeedbackManagerV49')
            StyleManager_cls = globals().get('DialogStyleManagerV49')
            PromptBuilder_cls = globals().get('PromptBuilderV49')
            GeneratorDeps_cls = globals().get('GeneratorDependenciesV49')
            DialogueGenerator_cls = globals().get('DialogueGeneratorV49')

            missing_deps = [name for name, cls in {
                "DialogueManagerV49": DialogueManager_cls, "ApiClientV49": ApiClient_cls,
                "SubjectivityFluctuationScorerV49": Scorer_cls, "AdvancedDialogueAnalyzerV49": Analyzer_cls,
                "EnhancedDialogFlowEvaluatorV49": Evaluator_cls, "PhaseToneAdaptationStrategyV49": Adaptation_cls,
                "FeedbackManagerV49": FeedbackManager_cls, "DialogStyleManagerV49": StyleManager_cls,
                "PromptBuilderV49": PromptBuilder_cls, "GeneratorDependenciesV49": GeneratorDeps_cls,
                "DialogueGeneratorV49": DialogueGenerator_cls
            }.items() if cls is None]
            if missing_deps:
                logger.critical(f"依存コンポーネントのクラス定義が見つかりません: {', '.join(missing_deps)}")
                return 1

            dialogue_manager_instance = DialogueManager_cls(job_id_base, app_config, exception_manager) # type: ignore
            dialogue_manager_instance.initialize_directories(job_settings) # type: ignore
            api_instance = ApiClient_cls(app_config, exception_manager) # type: ignore
            scorer_instance = Scorer_cls(app_config) # type: ignore
            analyzer_instance = Analyzer_cls(app_config, scorer_instance) # type: ignore
            evaluator_instance = Evaluator_cls(app_config, scorer_instance) # type: ignore
            adaptation_instance = Adaptation_cls(app_config, exception_manager) # type: ignore
            feedback_manager_instance = FeedbackManager_cls(app_config) # type: ignore
            style_manager_instance = StyleManager_cls(app_config) # type: ignore
            prompt_builder_instance = PromptBuilder_cls(app_config, feedback_manager_instance.composite_strategy, style_manager_instance) # type: ignore
            
            logger.debug("Re-attempting to rebuild GeneratorDependenciesV49 before instantiation in main_v49...")
            try:
                if hasattr(GeneratorDeps_cls, 'model_rebuild'):
                    GeneratorDeps_cls.model_rebuild(force=True) # type: ignore
                    logger.info("GeneratorDependenciesV49.model_rebuild() called successfully from main_v49.")
                else:
                    logger.warning("GeneratorDependenciesV49 does not have model_rebuild method.")
            except PydanticUndefinedAnnotation as e_rebuild_undefined: # type: ignore
                logger.error(f"PydanticUndefinedAnnotation during model_rebuild: {e_rebuild_undefined}", exc_info=True)
            except Exception as e_rebuild_other:
                logger.error(f"Error during model_rebuild: {e_rebuild_other}", exc_info=True)
            
            # デバッグログは削除（Pydantic V2 の挙動に依存する部分は最小限に）

            dependencies = GeneratorDeps_cls( # type: ignore
                config=app_config,
                settings=job_settings,
                api_client=api_instance,
                analyzer=analyzer_instance,
                scorer=scorer_instance,
                evaluator=evaluator_instance,
                adaptation_strategy=adaptation_instance,
                feedback_manager=feedback_manager_instance,
                prompt_builder=prompt_builder_instance,
                style_manager=style_manager_instance,
                dialogue_manager=dialogue_manager_instance,
                exception_manager=exception_manager
            )
            logger.info("依存コンポーネント初期化完了。")
            
            generator = DialogueGenerator_cls(job_id_base, dependencies) # type: ignore
            if dialogue_manager_instance: # job_id を generator のものに同期
                 dialogue_manager_instance.job_id = generator.job_id
            
        except Exception as e_comp_init:
            logger.critical(f"依存コンポーネント初期化中にエラー: {e_comp_init}", exc_info=True)
            if exception_manager:
                exception_manager.log_error(e_comp_init, "main_component_init_critical")
            return 1
            
        # 5. 対話生成ループの実行
        if not generator or not validated_input_data:
            logger.critical("ジェネレータまたは入力データがnull。処理中止。")
            return 1

        logger.info(f"=== ジョブ '{generator.job_id}' 対話生成開始 ===")
        target_length = args.length if args.length is not None else job_settings.target_length # type: ignore
        
        final_state = generator.execute_generation_loops(
            character_a_input=validated_input_data.characterA.model_dump(by_alias=True),
            character_b_input=validated_input_data.characterB.model_dump(by_alias=True),
            scene_info_input=validated_input_data.sceneInfo.model_dump(by_alias=True),
            target_length=target_length,
            report_type=args.report
        )
        logger.info(f"=== ジョブ '{generator.job_id}' 対話生成終了 ===")
        
        exit_code = 0
        if final_state and final_state.last_error: # type: ignore
            error_code = final_state.last_error.get('error_code', 'UNKNOWN.GENERIC') # type: ignore
            if exception_manager:
                severity = exception_manager.ERROR_SEVERITY.get(error_code, 'FATAL')
                if severity != 'WARNING_ONLY': exit_code = 1
            else: # exception_managerがない場合はデフォルトでエラーとする
                exit_code = 1
            logger.log(logging.WARNING if exit_code == 1 else logging.INFO, f"ジョブ'{generator.job_id}'は{'エラーで' if exit_code == 1 else '警告付きで'}終了/完了。")
        elif not final_state:
            logger.error(f"ジョブ'{generator.job_id}'の最終状態取得不可。")
            exit_code = 1
            
    except SystemExit as se:
        exit_code = se.code if isinstance(se.code, int) else 1
        if exit_code != 0: logger.error(f"処理が SystemExit (Code: {exit_code}) で中断。")
    except KeyboardInterrupt:
        logger.warning("処理がユーザーにより中断 (KeyboardInterrupt)。")
        exit_code = 130
    except Exception as main_err:
        logger.critical(f"メイン処理ループ中に予期せぬ致命的エラー: {main_err}", exc_info=True)
        if exception_manager: exception_manager.log_error(main_err, "main_loop_uncaught")
        exit_code = 1
        
    finally:
        duration = datetime.now(timezone.utc) - start_time
        logger.info(f"総処理時間: {duration.total_seconds()/60:.2f}分 ({duration.total_seconds():.1f}秒)")
        final_system_version = getattr(app_config, 'SYSTEM_VERSION', system_version_for_log) if app_config else system_version_for_log
        logger.info(f"=== {final_system_version} 実行終了 (Exit Code: {exit_code}) ===")
        logging.shutdown()
    
    return exit_code

# =============================================================================
# -- Part 17 終了点 (main_v49 関数の定義終了)
# =============================================================================

if __name__ == "__main__":
    final_exit_code = 1
    try:
        final_exit_code = main_v49()
    except Exception as e_very_top_level:
        # この段階ではloggerが機能しているか不明なため、標準エラー出力も使用
        print(f"CRITICAL UNHANDLED EXCEPTION AT SCRIPT TOP LEVEL: {e_very_top_level}", file=sys.stderr)
        if logger and logger.handlers: # loggerが利用可能ならログにも記録
             logger.critical(f"CRITICAL UNHANDLED EXCEPTION AT SCRIPT TOP LEVEL: {e_very_top_level}", exc_info=True)
        else: # loggerが利用できない場合はトレースバックを標準エラーに出力
            import traceback
            traceback.print_exc()
        final_exit_code = 2
    finally:
        sys.exit(final_exit_code)
