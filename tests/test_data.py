"""Unit tests for src.data module."""

import math

import numpy as np
import pandas as pd
import pytest

from src.data import (
    address_entropy,
    engineer_features,
    get_port_range,
    prepare_xy,
    safe_divide,
)


class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10, 2) == 5.0

    def test_zero_denominator(self):
        assert safe_divide(10, 0) == 0

    def test_zero_numerator(self):
        assert safe_divide(0, 5) == 0.0

    def test_float_division(self):
        assert abs(safe_divide(1, 3) - 1 / 3) < 1e-10


class TestAddressEntropy:
    def test_known_ip(self):
        entropy = address_entropy("192.168.1.1")
        assert entropy > 0

    def test_single_char(self):
        assert address_entropy("a") == 0

    def test_empty_string(self):
        assert address_entropy("") == 0

    def test_uniform_distribution(self):
        # "abcd" has 4 unique chars, each appearing once -> entropy = log2(4) = 2.0
        entropy = address_entropy("abcd")
        assert abs(entropy - 2.0) < 1e-10

    def test_all_same_chars(self):
        # "aaaa" has entropy 0
        assert address_entropy("aaaa") == 0.0


class TestGetPortRange:
    def test_well_known(self):
        assert get_port_range(80) == 'WellKnown'
        assert get_port_range(0) == 'WellKnown'
        assert get_port_range(1023) == 'WellKnown'

    def test_registered(self):
        assert get_port_range(1024) == 'Registered'
        assert get_port_range(8080) == 'Registered'
        assert get_port_range(49151) == 'Registered'

    def test_ephemeral(self):
        assert get_port_range(49152) == 'Ephemeral'
        assert get_port_range(65535) == 'Ephemeral'

    def test_unknown_string(self):
        assert get_port_range('abc') == 'Unknown'

    def test_unknown_none(self):
        assert get_port_range(None) == 'Unknown'

    def test_string_number(self):
        assert get_port_range('443') == 'WellKnown'


class TestEngineerFeatures:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'StartTime': ['2011/08/18 15:40:53'],
            'Dur': [2.0],
            'Proto': ['tcp'],
            'SrcAddr': ['192.168.1.1'],
            'Sport': ['80'],
            'Dir': ['->'],
            'DstAddr': ['10.0.0.1'],
            'Dport': ['443'],
            'State': ['S_RA'],
            'sTos': [0],
            'dTos': [0],
            'TotPkts': [10],
            'TotBytes': [1000],
            'SrcBytes': [500],
        })

    def test_creates_expected_columns(self, sample_df):
        result = engineer_features(sample_df)
        expected_cols = ['BytesPerSecond', 'PktsPerSecond', 'SrcAddrEntropy',
                         'DstAddrEntropy', 'SportRange', 'DportRange',
                         'DurCategory', 'BytePktRatio']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_drops_unnecessary_columns(self, sample_df):
        result = engineer_features(sample_df)
        for col in ['sTos', 'dTos', 'StartTime', 'Sport', 'Dport']:
            assert col not in result.columns

    def test_bytepktratio_value(self, sample_df):
        result = engineer_features(sample_df)
        assert result['BytePktRatio'].iloc[0] == 100.0  # 1000 / 10

    def test_bytespersecond_value(self, sample_df):
        result = engineer_features(sample_df)
        assert result['BytesPerSecond'].iloc[0] == 500.0  # 1000 / 2.0


class TestPrepareXY:
    @pytest.fixture
    def engineered_df(self):
        return pd.DataFrame({
            'Dur': [1.0, 2.0, 3.0],
            'Proto': ['tcp', 'udp', 'tcp'],
            'Dir': ['->', '->', '->'],
            'State': ['S_RA', 'S_RA', 'S_RA'],
            'SrcAddr': ['192.168.1.1', '10.0.0.1', '172.16.0.1'],
            'DstAddr': ['10.0.0.1', '192.168.1.1', '8.8.8.8'],
            'TotPkts': [10, 20, 30],
            'TotBytes': [100, 200, 300],
            'SrcBytes': [50, 100, 150],
            'BytesPerSecond': [100.0, 100.0, 100.0],
            'PktsPerSecond': [10.0, 10.0, 10.0],
            'BytePktRatio': [10.0, 10.0, 10.0],
            'SrcAddrEntropy': [2.5, 2.5, 2.5],
            'DstAddrEntropy': [2.5, 2.5, 2.5],
            'DurCategory': pd.Categorical(['very_short', 'short', 'short']),
            'Botnet': [0, 1, 0],
        })

    def test_output_shapes(self, engineered_df):
        X, y = prepare_xy(engineered_df)
        assert len(y) == 3
        assert 'Botnet' not in X.columns

    def test_ip_columns_dropped(self, engineered_df):
        X, y = prepare_xy(engineered_df)
        assert 'SrcAddr' not in X.columns
        assert 'DstAddr' not in X.columns

    def test_categoricals_encoded(self, engineered_df):
        X, y = prepare_xy(engineered_df)
        for col in ['Proto', 'Dir', 'State']:
            if col in X.columns:
                assert X[col].dtype in [np.int8, np.int16, np.int32, np.int64]

    def test_missing_botnet_raises(self):
        df = pd.DataFrame({'Dur': [1.0], 'TotPkts': [10]})
        with pytest.raises(ValueError, match="No 'Botnet' column"):
            prepare_xy(df)
