"""Test the custom ONNX handler."""

import zoneinfo
from datetime import datetime

import time_machine
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
)

from lazyscribe_onnx import ONNXArtifact


@time_machine.travel(
    datetime(2025, 1, 20, 13, 23, 30, tzinfo=zoneinfo.ZoneInfo("UTC")), tick=False
)
def test_onnx_handler(tmp_path):
    """Test reading and writing ONNX files with the handler."""
    location = tmp_path / "my-location"
    location.mkdir()

    # Create the ONNX model based on their tutorial here:
    # https://onnx.ai/onnx/intro/python.html

    # 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])

    # outputs, the shape is left undefined

    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

    # nodes

    # It creates a node defined by the operator type MatMul,
    # 'X', 'A' are the inputs of the node, 'XA' the output.
    node1 = make_node("MatMul", ["X", "A"], ["XA"])
    node2 = make_node("Add", ["XA", "B"], ["Y"])

    # from nodes to graph
    # the graph is built from the list of nodes, the list of inputs,
    # the list of outputs and a name.

    graph = make_graph(
        [node1, node2],  # nodes
        "lr",  # a name
        [X, A, B],  # inputs
        [Y],
    )  # outputs

    # onnx graph
    # there is no metadata in this case.

    onnx_model = make_model(graph)

    # Let's check the model is consistent,
    # this function is described in section
    # Checker and Shape Inference.
    check_model(onnx_model)

    handler = ONNXArtifact.construct(name="My output file")

    assert handler.fname == "my-output-file-20250120132330.onnx"

    with open(location / handler.fname, "wb") as buf:
        handler.write(onnx_model, buf)

    assert (location / handler.fname).is_file()

    with open(location / handler.fname, "rb") as buf:
        out = handler.read(buf)

    check_model(out)

    assert onnx_model == out
