"""Test using the ONNX handler with Lazyscribe."""

import zoneinfo
from datetime import datetime

import time_machine
from lazyscribe import Project
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
)


@time_machine.travel(
    datetime(2025, 1, 20, 13, 23, 30, tzinfo=zoneinfo.ZoneInfo("UTC")), tick=False
)
def test_onnx_project_write(tmp_path):
    """Test logging an artifact using the ONNX handler."""
    location = tmp_path / "my-project-location"
    location.mkdir()

    project = Project(fpath=location / "project.json", mode="w")
    with project.log("My experiment") as exp:
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])

        graph = make_graph([node1, node2], "lr", [X, A, B], [Y])

        onnx_model = make_model(graph)

        exp.log_artifact(name="My model", value=onnx_model, handler="onnx")

    project.save()

    assert (
        location / "my-experiment-20250120132330" / "my-model-20250120132330.onnx"
    ).is_file()

    project_r = Project(fpath=location / "project.json", mode="r")
    out = project_r["my-experiment"].load_artifact(name="My model")

    check_model(out)

    assert onnx_model == out
