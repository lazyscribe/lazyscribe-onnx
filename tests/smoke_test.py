"""Check that basic features work.

Used in our publishing pipeline.
"""

import tempfile
from pathlib import Path

from lazyscribe import Project
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
)

with tempfile.TemporaryDirectory() as tmpdir:
    # Create a project
    project = Project(Path(tmpdir) / "project.json", mode="w")
    with project.log(name="ONNX experiment") as exp:
        # Create a fake object and log it
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

        node1 = make_node("MatMul", ["X", "A"], ["XA"])
        node2 = make_node("Add", ["XA", "B"], ["Y"])

        graph = make_graph([node1, node2], "lr", [X, A, B], [Y])

        onnx_model = make_model(graph)

        exp.log_artifact(name="model", value=onnx_model, handler="onnx")

    project.save()

    exp = project["onnx-experiment"]
    data_onnx_ = exp.load_artifact(name="model")

    check_model(data_onnx_)
    assert onnx_model == data_onnx_
