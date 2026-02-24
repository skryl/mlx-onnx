import pytest

from _op_harness import run_case


@pytest.mark.parametrize('case_id', ['ArgReduceMin', 'ArgReduceMax'])
def test_argreduce_mlx_to_onnx_and_parity(case_id):
    run_case(case_id)
